import pygame
import sys
import random
import torch
import gc

# Set print options
torch.set_printoptions(profile="full")

from map_data.constants import CIRCLE_RADIUS, TAN, COLOR_NAMES, DARK_GREEN, INPUT_SIZE, OUTPUT_SIZE
# from map_data.map_attributes import Map, City, Upgrade, Office, Route
from ai.ai_model import HansaNN
from ai.game_state import BoardData
from ai.action_options import perform_action_from_index, masking_out_invalid_actions
from game.game_info import Game
from game.game_actions import claim_post_action, displace_action, move_action, displace_claim, assign_new_bonus_marker_on_route, claim_route_for_office, claim_route_for_upgrade, claim_route_for_points, buy_tile
from drawing.drawing_utils import redraw_window, draw_end_game

def end_game(winning_players):
    draw_end_game(win, winning_players)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_RETURN:
                    pygame.quit()
                    sys.exit()

def check_bounds(item, pos):
    return (item.x_pos < pos[0] < item.x_pos + item.width and
            item.y_pos < pos[1] < item.y_pos + item.height)

def check_if_post_clicked(pos, button):
    route, post = find_post_by_position(pos)

    if post:
        if button == 1:  # left click
            if post.can_be_claimed_by("square"):
                claim_post_action(game, route, post, "square")
            elif post.is_owned() and post.owner != game.current_player:
                displace_action(game, post, route, "square")
            elif post.is_owned() and post.owner == game.current_player:
                handle_move(pos, button)
        elif button == 3:  # right click
            if post.can_be_claimed_by("circle"):
                claim_post_action(game, route, post, "circle")
            elif post.is_owned() and post.owner != game.current_player:
                displace_action(game, post, route, "circle")
        elif button == 2:  # middle click
            post.DEBUG_print_post_details()
            return
        # game.switch_player_if_needed()
        return

def handle_move(pos, button):
    # Find the clicked post based on position
    route, post = find_post_by_position(pos)
    shape_clicked = 'circle' if button == 3 else 'square'

    if post is None:
        print("No valid post was clicked.")
        return
    # Check if selecting an empty post when no pieces are held
    elif not game.current_player.holding_pieces and not post.is_owned():
        print("Cannot select an empty post, when holding no pieces.")
        return

    move_action(game, route, post, shape_clicked)

def find_clicked_city(cities, pos):
    for city in cities:
        if check_bounds(city, pos):
            # print(f"Clicked on city: {city.name}")
            return city
    return None

def check_if_route_claimed(pos, button):
    current_player = game.current_player

    if current_player.actions <= 0:
        return

    city = find_clicked_city(game.selected_map.cities, pos)
    if city:
        controlled_routes = [route for route in city.routes if route.is_controlled_by(current_player)]
        if len(controlled_routes) > 1:
            # More than one route controlled by the player, need to choose
            print(f"Choose a Route to claim the city {city.name} with.")
            selected_route = get_route_choice(controlled_routes)
        elif controlled_routes:
            # Only one route, proceed with it
            selected_route = controlled_routes[0]
        else:
            print(f"No controlled routes available for city: {city.name}")
            return
        
        if selected_route.is_controlled_by(current_player):
            if button == 1: # leftclick
                claim_route_for_office(game, city, selected_route)
            elif button == 2: #middle click
                upgrade_choice = None
                if not city.upgrade_city_type:
                    print(f"City: {city.name} does not contain any upgrades.")
                elif len(city.upgrade_city_type) > 1:
                    print(f"Waiting for player to click on a valid upgrade in the city: {city.name}")
                    upgrades_for_city = [upgrade for upgrade in game.selected_map.upgrade_cities if upgrade.city_name == city.name]
                    upgrade_choice = get_upgrade_choice(upgrades_for_city)
                else:
                    upgrade_choice = city.upgrade_city_type[0]  # Select the single upgrade type
                claim_route_for_upgrade(game, city, selected_route, upgrade_choice)
            elif button == 3: #right click
                claim_route_for_points(game, selected_route)
            else:
                print(f"Invalid scenario with {city.name}")
            if selected_route.permanent_bonus_marker:
                handle_permanent_bonus_marker(selected_route.permanent_bonus_marker.type)
            check_if_game_over()
            return
        else:
            print(f"Route(s) not controlled by current_player: {COLOR_NAMES[current_player.color]}")

def check_if_game_over():
    highest_scoring_players = game.check_for_game_end()
    if highest_scoring_players:
        redraw_window(win, game)
        end_game(highest_scoring_players)

def get_upgrade_choice(upgrade_options):
    waiting_for_click = True
    while waiting_for_click:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                for upgrade in upgrade_options:
                    if check_bounds(upgrade, pygame.mouse.get_pos()) and game.current_player.actions > 0:
                        return upgrade.upgrade_type
                print (f"Please click on either {upgrade_options[0].upgrade_type} or {upgrade_options[1].upgrade_type}")
        pygame.time.wait(10)  # Wait for a short period to prevent high CPU usage

    return None

def get_route_choice(route_options):
    waiting_for_click = True
    while waiting_for_click:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                for route in route_options:
                    route, post = find_post_by_position(pygame.mouse.get_pos())
                    if route:
                        return route
                print("Please click on one of the available routes.")
        pygame.time.wait(10)  # Wait for a short period to prevent high CPU usage

    return None

def check_if_income_clicked(pos):
    # Check if any player board's Income Action button was clicked
    board = game.current_player.board
    if hasattr(board, 'circle_buttons'):
        for idx, button_rect in enumerate(board.circle_buttons):
            if button_rect.collidepoint(pos):
                board.income_action_based_on_circle_count(idx)
                if game.OneIncomeIfOthersIncomeOwner:
                    if game.current_player != game.OneIncomeIfOthersIncomeOwner:
                        game.OneIncomeIfOthersIncomeOwner.add_1_income()
                return

def handle_bonus_marker(player, route, reset_pieces):
    if route.bonus_marker:
        route.bonus_marker.owner = player
        player.bonus_markers.append(route.bonus_marker)
        route.bonus_marker = None
        game.replace_bonus_marker += 1
    elif route.permanent_bonus_marker:
        print(f"Waiting for Player to handle {route.permanent_bonus_marker.type} BM")
        handle_permanent_bonus_marker(route.permanent_bonus_marker.type, reset_pieces)

def handle_permanent_bonus_marker(perm_bm_type):
    waiting_for_click = True

    while waiting_for_click:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_position = pygame.mouse.get_pos()
                if perm_bm_type == 'MoveAny2':
                    handle_move(mouse_position, event.button)
                    if (not game.current_player.pieces_to_pickup and
                        not game.current_player.holding_pieces):
                        waiting_for_click = False
                        game.waiting_for_bm_move_any_2 = False
                elif perm_bm_type == 'ClaimGreenCity':
                    if claim_green_city_with_bm(mouse_position):
                        waiting_for_click = False
                elif perm_bm_type == 'Place2TradesmenFromRoute':
                    handle_place_two_tradesmen_from_route(mouse_position, event.button)
                    if not game.current_player.pieces_to_place:
                        print(f"Finished placing pieces on valid posts!")
                        waiting_for_click = False
                        game.waiting_for_bm_move_any_2 = False
                elif perm_bm_type == 'Place2ScotlandOrWales':
                    handle_place_two_tradesmen_from_route(mouse_position, event.button)
                    if not game.current_player.pieces_to_place:
                        print(f"Finished placing pieces on valid posts INSIDE Scotland or Wales!")
                        waiting_for_click = False
                        game.waiting_for_place2_in_scotland_or_wales = False
                else:
                    print("Invalid scenario.")
        redraw_window(win, game)
        pygame.display.flip()  # Update the screen
        pygame.time.wait(100)
    return None

def claim_green_city_with_bm(pos):
    for city in game.selected_map.cities:
        if check_bounds(city, pos) and city.color == DARK_GREEN:
            if city.claim_green_city(game):
                return True
    print("Please click on a GREEN City!")
    return False

def handle_place_two_tradesmen_from_route(pos, button):
    _, post = find_post_by_position(pos)

    # Check if a post was found
    if post is None:
        print("No valid post was clicked.")
        return

    reset_pieces = game.current_player.holding_pieces

    if game.waiting_for_place2_in_scotland_or_wales:
        if post.region != 'Scotland' and post.region != 'Wales':
            print("Please click on a post in Scotland or Wales!")
            return False
            
    # Check if a post was clicked and is empty
    if post and not post.is_owned():
        shape_clicked = 'circle' if button == 3 else 'square'
        
        # Find the piece in reset_pieces that matches the shape_clicked
        piece_to_remove = next((piece for piece in reset_pieces if piece[0] == shape_clicked), None)
        
        if piece_to_remove and (not post.required_shape or post.required_shape == shape_clicked):
            post.claim(game.current_player, shape_clicked)
            reset_pieces.remove(piece_to_remove)
            game.current_player.pieces_to_place -= 1

            # Update the general stock for the current player
            if shape_clicked == 'circle':
                game.current_player.general_stock_circles -= 1
            else:
                game.current_player.general_stock_squares -= 1

            remaining_pieces = ', '.join(piece[0] for piece in reset_pieces)
            print(f"BM: Please place {game.current_player.pieces_to_place} more piece(s) of {remaining_pieces} on valid posts!")
        else:
            print(f"ERROR: Cannot place {shape_clicked}. Available pieces: {', '.join(piece[0] for piece in reset_pieces)}.")
    elif not post:
        print("No valid post was clicked.")

def check_if_bm_clicked(pos):
    for bm in game.current_player.bonus_markers:
        if bm.is_clicked(pos):
            print(f"Clicked on bonus marker: {bm.type}")
            return use_bonus_marker(bm)
    return False

def use_bonus_marker(bm):
    player = game.current_player
    
    if bm.type == 'PlaceAdjacent':
        print ("Only can be done if route is full.")
        print ("If route is full, clicking on the city will automatically handle this.")
        return True
    elif bm.type == 'SwapOffice':
        print("Click a City to swap offices on.")
    elif bm.type == 'Move3':
        bm.handle_move3(game)
    elif bm.type == 'UpgradeAbility':
        print("Please click on an upgrade to choose it.")
    elif bm.type == '3Actions':
        player.actions_remaining += 3
        player.used_bonus_markers.append(bm)
        player.bonus_markers.remove(bm)
        return True
    elif bm.type == '4Actions':
        player.actions_remaining += 4
        player.used_bonus_markers.append(bm)
        player.bonus_markers.remove(bm)
        return True
    elif bm.type == 'ExchangeBonusMarker':
        opponents_used_bms_exist = False
        for opponents in game.players:
            if opponents != game.current_player and len(opponents.used_bonus_markers) > 0:
                opponents_used_bms_exist = True
                break
        if not opponents_used_bms_exist:
            print("No opponents have used bonus markers to exchange.")
            return False

        print("Please click on a Used Bonus Marker - from any other player - to exchange this with it.")
    elif bm.type == 'Tribute4EstablishingTP' or bm.type == 'BlockTradeRoute':
        if player.personal_supply_squares <= 0:
            print("You have no squares in your personal supply to place on the board.")
            return False
        else:
            print("Please click a trade route to establish this Bonus Marker.")
    else:
        return False
    
    player.used_bonus_markers.append(bm)
    player.bonus_markers.remove(bm)

    waiting_for_click = True
    while waiting_for_click:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_position = pygame.mouse.get_pos()
                if bm.type == 'SwapOffice':
                    for city in game.selected_map.cities:
                        if check_bounds(city, mouse_position):
                            print(f"Clicked on city: {city.name}")
                            if bm.handle_swap_office(city, player):
                                waiting_for_click = False
                elif bm.type == 'Move3':
                    handle_move(mouse_position, event.button)
                    if (not game.current_player.pieces_to_place and
                        not game.current_player.holding_pieces):
                        waiting_for_click = False
                        game.waiting_for_bm_move3 = False
                elif bm.type == 'UpgradeAbility':
                    for upgrade in game.selected_map.upgrade_cities:
                        if check_bounds(upgrade, mouse_position):
                            if bm.handle_upgrade_ability(upgrade, player):
                                waiting_for_click = False
                elif bm.type == 'Tribute4EstablishingTP':
                    route, _ = find_post_by_position(mouse_position)
                    if route:
                        print(f"Clicked on route between cities of {route.cities[0].name} and {route.cities[1].name}")
                        if bm.handle_tribute4_establishing_tp(route, player):
                            waiting_for_click = False
                elif bm.type == 'BlockTradeRoute':
                    route, _ = find_post_by_position(mouse_position)
                    if route:
                        print(f"Clicked on route between cities of {route.cities[0].name} and {route.cities[1].name}")
                        if bm.handle_block_trade_route(route, player):
                            waiting_for_click = False
                elif bm.type == 'ExchangeBonusMarker':
                    for opponent_bm, rect in game.opponents_used_bms_rects.items():
                        if rect.collidepoint(mouse_position):
                            print(f"Used BM {opponent_bm.type} was clicked.")
                            print(f"Exchanging {COLOR_NAMES[game.current_player.color]}'s BM 'ExchangeBonusMarker' with {COLOR_NAMES[opponent_bm.owner.color]}'s BM {opponent_bm.type}.")
                            opponent_bm.handle_exchange_bonus_marker(game)
                            waiting_for_click = False
                else:
                    print("Invalid scenario.")
        redraw_window(win, game)
        pygame.display.flip()  # Update the screen
        pygame.time.wait(100)
    return True

def check_if_tile_clicked(pos):
    bm_payment1 = None
    bm_payment2 = None
    for tile, rect in game.tile_rects.items():
        if rect.collidepoint(pos):
            print(f"Tile {tile} was clicked.")
            if len(game.current_player.bonus_markers) > 2:
                print(f"Player {COLOR_NAMES[game.current_player.color]} has more than 2 bonus markers, please select 2 to use to play for the tile.")
                waiting_for_click = True

                while waiting_for_click:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.MOUSEBUTTONUP:
                            mouse_position = pygame.mouse.get_pos()
                            for bm in game.current_player.bonus_markers:
                                if bm.is_clicked(mouse_position):
                                    print(f"Selected BM: {bm.type}")
                                    if bm_payment1 is None:
                                        bm_payment1 = bm
                                    elif bm_payment2 is None:
                                        bm_payment2 = bm
                                        waiting_for_click = False   
                    pygame.display.flip()  # Update the screen
                    pygame.time.wait(100)
            buy_tile(game, tile, bm_payment1, bm_payment2)
            redraw_window(win, game)

def handle_click(pos, button):
    check_if_post_clicked(pos, button)
    check_if_route_claimed(pos,button)
    check_if_income_clicked(pos)
    check_if_tile_clicked(pos)                        

def displaced_click(pos, button):
    route, post = find_post_by_position(pos) 

    if game.DisplaceAnywhereOwner == game.displaced_player.player:
        print (f"Displaced Player: {COLOR_NAMES[game.displaced_player.player.color]}")
        print (f"Owner: {COLOR_NAMES[game.DisplaceAnywhereOwner.color]} can displace to any post.")
    elif post not in game.all_empty_posts and not game.displaced_player.is_general_stock_empty() and not game.displaced_player.is_personal_supply_empty():
        return

    # Determine which shape the player wants to place based on the button they clicked
    desired_shape = 'circle' if button == 3 else 'square'
    
    displace_claim(game, post, desired_shape)

def is_route_filled(route):
    """Checks if all posts on a route are filled."""
    for post in route.posts:
        if post.owner is None:
            return False
    return True

def find_post_by_position(pos):
    for route in routes:
        for post in route.posts:
            if abs(post.pos[0] - pos[0]) < CIRCLE_RADIUS and abs(post.pos[1] - pos[1]) < CIRCLE_RADIUS:
                return route, post
    return None, None

def handle_end_turn_click(pos):
    if (game.selected_map.map_width+415 < pos[0] < game.selected_map.map_width+415 + 75 and
        game.selected_map.map_height-170 < pos[1] < game.selected_map.map_height-170 + 170):
        print ("End Turn clicked!")
        game.current_player.ending_turn = True
    else:
        print("Please click End Turn, or a Bonus Marker that you can use!")

def handle_get_game_state(pos):
    if (game.selected_map.map_width+415 < pos[0] < game.selected_map.map_width+415 + 75 and
        game.selected_map.map_height-100 < pos[1] < game.selected_map.map_height):
        return True

def check_if_player_has_usable_BMs():
    return game.current_player.bonus_markers and not all(bm.type == 'PlaceAdjacent' for bm in game.current_player.bonus_markers)

def handle_shift_click(mouse_position):
    for upgrade in game.selected_map.upgrade_cities:
        if check_bounds(upgrade, mouse_position):
            if game.current_player.perform_upgrade(upgrade.upgrade_type):
                print(f"ADMIN: Clicked on upgrade: {upgrade.upgrade_type}")
                return True
            else:
                print("ADMIN: Invalid click when Upgrading")
                return False
    for tile, rect in game.tile_rects.items():
        if rect.collidepoint(mouse_position):
            print(f"ADMIN: Tile {tile} was clicked.")
            game.tile_pool.remove(tile)
            game.current_player.tiles.append(tile)
            return True            
            
game = Game(map_num=1, num_players=3)
board_data = BoardData()
game_state_tensor = board_data.get_game_state(game)
hansa_nn = HansaNN(board_data.all_game_state_size, OUTPUT_SIZE, model_file=f"hansa_nn_model.pth")
# game = board_data.load_game_from_file('game_states_for_training.txt')

# # Print initial weights
# # print("Initial weights of layer1:")
# # game.players[game.active_player].hansa_nn.print_weights('layer1')
# initial_weights = game.players[0].hansa_nn.layer1.weight.data.cpu().numpy().flatten()
# initial_weights_layer3 = game.players[0].hansa_nn.layer3.weight.data.cpu().numpy().flatten()
epsilon_start = 1.0
epsilon_end = 0.1
decay_rate = 0.005  # Adjust this to control how quickly epsilon decays
epsilon = epsilon_start

for j in range(1):
    game = Game(map_num=random.randint(1, 2), num_players=random.randint(3, 5))
    for i in range(1):
        active_player = game.players[game.active_player] #declaring this variable now to prevent the active player variable from being overwritten

        # Get the current game state tensor
        game_state_tensor = board_data.get_game_state(game)
        game_state_tensor = game_state_tensor.float()   # Convert to float if not already

        # Forward pass through the neural network to get the action probabilities
        action_probabilities = hansa_nn(game_state_tensor.unsqueeze(0))  # Adding an extra dimension for batch
        # print(f"action_probabilities: {action_probabilities}")

        # Apply masking to filter out invalid actions
        valid_actions = masking_out_invalid_actions(game)
        masked_action_probabilities = action_probabilities * valid_actions
        masked_action_probabilities = masked_action_probabilities.flatten()

        # Get the top 3 indices with highest probabilities
        topk_values, topk_indices = torch.topk(masked_action_probabilities, 3)

        topk_values = topk_values.flatten()
        top3_choices = topk_indices.flatten()

        # Filter out indices that correspond to zero probabilities
        valid_top3_choices = [index for index, value in zip(top3_choices, topk_values) if value > 0]

        # If there are no valid choices (all probabilities are zero), handle this case separately
        if not valid_top3_choices:
            print("No valid actions available")
            print("action_probabilities:")
            print(f"{action_probabilities}")
            print("masked_action_probabilities:")
            print(f"{masked_action_probabilities}")
            print("No valid actions available")
            break
        else:
            if random.random() < epsilon:
                # Exploration: Randomly select from valid actions
                selected_index = random.choice(valid_top3_choices)
                print(f"Exploration: Selected random action: Index {selected_index}")
            else:
                # Exploitation: Select the best action
                selected_index = valid_top3_choices[0]  # Assuming top3_choices are sorted by probability
                print(f"Exploitation: Selected best action: Index {selected_index}")

            # Perform the action corresponding to the selected index
            perform_action_from_index(game, selected_index)

            # Calculate loss and update network
            hansa_nn.optimizer.zero_grad()
            prob = masked_action_probabilities[selected_index].item()
            if prob > 0:
                log_prob = torch.log(masked_action_probabilities[selected_index])
                # if active_player.reward == 0:
                #     active_player.reward = 0.1
                loss = -log_prob * active_player.reward
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(hansa_nn.parameters(), max_norm=1.0)

                hansa_nn.optimizer.step()
                print(f"[{COLOR_NAMES[active_player.color]}] Reward: {active_player.reward}, Log Prob: {log_prob}, Loss: {loss.item()}")
                
            else:
                print("Invalid probability value, skipping update")
                break
            active_player.reward = 0
            epsilon = max(epsilon_end, epsilon - decay_rate)

        print(f"Saving model for Player: {active_player.order} as hansa_nn_model.pth")
        torch.save(hansa_nn.state_dict(), f"hansa_nn_model.pth")
    gc.collect()

# final_weights = game.players[0].hansa_nn.layer1.weight.data.cpu().numpy().flatten()
# final_weights_layer3 = game.players[0].hansa_nn.layer3.weight.data.cpu().numpy().flatten()
# # Plot histograms
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.hist(initial_weights_layer3, bins=50, alpha=0.7)
# plt.title("Initial Weights Distribution")
# plt.subplot(1, 2, 2)
# plt.hist(final_weights_layer3, bins=50, alpha=0.7)
# plt.title("Final Weights Distribution")
# plt.show()

WIDTH = game.selected_map.map_width+800
HEIGHT = game.selected_map.map_height
cities = game.selected_map.cities
routes = game.selected_map.routes

win = pygame.display.set_mode((WIDTH, HEIGHT))
# viewable_window = pygame.display.set_mode((1800, 1350))
pygame.display.set_caption('Hansa Teutonica Sample Game')
win.fill(TAN)

# exit()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_position = pygame.mouse.get_pos()
            
            # Admin Mode
            mods = pygame.key.get_mods()  # Get the current state of all keyboard modifiers
            if mods & pygame.KMOD_SHIFT:
                # Shift was held down when the mouse button was released
                handle_shift_click(mouse_position)

            elif handle_get_game_state(mouse_position):
                board_data.save_game_state_to_file(game)

            elif(check_if_bm_clicked(mouse_position)):
                print(f"Bonus Marker was used!")
            # while game_not_over:
            elif game.waiting_for_displaced_player:
                displaced_click(mouse_position, event.button)

            elif game.current_player.holding_pieces:
                handle_move(mouse_position, event.button)

            elif game.current_player.actions_remaining > 0:
                handle_click(mouse_position, event.button)

            elif game.current_player.actions_remaining == 0:
                 # Player has no actions remaining
                usable_BMs = check_if_player_has_usable_BMs()
                if (usable_BMs and not game.current_player.ending_turn):
                    # Player has usable BMs (not 'PlaceAdjacent'), wait for them to either use a BM or click "end turn"
                    handle_end_turn_click(mouse_position)
                    
                elif ((game.current_player.ending_turn and game.replace_bonus_marker > 0) or
                      (game.replace_bonus_marker > 0 and not usable_BMs)):
                    # Player clicked "end turn" and needs to replace bonus marker(s)
                    route, post = find_post_by_position(mouse_position)
                    assign_new_bonus_marker_on_route(game, route)
                    if game.replace_bonus_marker == 0:
                        # After placing BM, switch player and reset ending_turn
                        game.current_player.ending_turn = False
                        game.switch_player_if_needed()
                        break

                elif not usable_BMs and game.replace_bonus_marker == 0:
                    # Player has no BMs to use or only has 'PlaceAdjacent', switch player
                    game.switch_player_if_needed()
                    break
            
            if (game.current_player.actions_remaining == 0):
                if check_if_player_has_usable_BMs() and not game.current_player.ending_turn:
                    print ("Please use a BM or Select End Turn")
                elif game.replace_bonus_marker > 0:
                    print (f"Please Select a Route to place a BM. {game.replace_bonus_marker} remaining.")
                elif ((not check_if_player_has_usable_BMs() and game.replace_bonus_marker == 0) or
                      (game.current_player.ending_turn and game.replace_bonus_marker == 0)):
                    # Player has no BMs to use or only has 'PlaceAdjacent', switch player
                    game.switch_player_if_needed()
                    break

    redraw_window(win, game)
    # Scale down the large surface to fit into the display surface
    # scaled_surface = pygame.transform.scale(win, (1800, 1350))
    # Blit the scaled surface onto the display surface
    # viewable_window.blit(scaled_surface, (0, 0))
    # viewable_window.blit(win, (0, 0))

    pygame.display.flip()  # Update the screen
    pygame.time.delay(50)