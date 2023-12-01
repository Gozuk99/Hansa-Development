import pygame
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F

from map_data.constants import CIRCLE_RADIUS, TAN, COLOR_NAMES, DARK_GREEN
from map_data.map_attributes import Map, City, Upgrade, Office, Route
from ai.ai_model import HansaNN
from ai.game_state import get_available_actions, get_game_state
from ai.action_options import perform_action_from_index, masking_out_invalid_actions
from game.game_info import Game
from game.game_actions import claim_post_action, displace_action, gather_empty_posts, move_action, displace_claim, assign_new_bonus_marker_on_route, score_route, claim_route_for_office, claim_route_for_upgrade, claim_route_for_points
from drawing.drawing_utils import redraw_window, draw_end_game

game = Game(map_num=1, num_players=5)

WIDTH = game.selected_map.map_width+800
HEIGHT = game.selected_map.map_height
cities = game.selected_map.cities
routes = game.selected_map.routes

win = pygame.display.set_mode((WIDTH, HEIGHT))
# viewable_window = pygame.display.set_mode((1800, 1350))
pygame.display.set_caption('Hansa Sample Game')
win.fill(TAN)

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

    move_action(game, post, shape_clicked)

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
            print("No controlled routes available for this city.")
            return
        
        if selected_route.is_controlled_by(current_player):
            if button == 1: # leftclick
                claim_route_for_office(game, city, selected_route)
            elif button == 2: #middle click
                upgrade_choice = None
                if len(city.upgrade_city_type) > 1:
                    print(f"Waiting for player to click on a valid upgrade in the city: {city.name}")
                    upgrades_for_city = [upgrade for upgrade in game.selected_map.upgrade_cities if upgrade.city_name == city.name]
                    upgrade_choice = get_upgrade_choice(upgrades_for_city)
                else:
                    upgrade_choice = city.upgrade_city_type[0]  # Select the single upgrade type
                claim_route_for_upgrade(game, city, selected_route, upgrade_choice)
            elif button == 3: #right click
                claim_route_for_points(selected_route)
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
                # game.switch_player_if_needed()
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
                    if (not game.current_player.pieces_to_place and
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
                    return
                else:
                    print("Invalid scenario.")
        redraw_window(win, game)
        pygame.display.flip()  # Update the screen
        pygame.time.wait(100)
    return None

def claim_green_city_with_bm(pos):
    for city in game.selected_map.cities:
        if check_bounds(city, pos) and city.color == DARK_GREEN:
            if game.current_player.personal_supply_squares == 0 and game.current_player.personal_supply_circles == 0:
                print(f"Cannot claim GREEN City: {city.name}, because you have no Tradesmen in your Personal Supply")
            else:
                if city.city_all_offices_occupied():
                    #create a new office
                    #append it to city.offices
                    city.add_office(Office("square", "WHITE", 0))

                city.update_next_open_office_ownership(game)
                
                # Remove a square if available, otherwise remove a circle
                if game.current_player.personal_supply_squares > 0:
                    game.current_player.personal_supply_squares -= 1
                elif game.current_player.personal_supply_circles > 0:
                    game.current_player.personal_supply_circles -= 1
                
                print(f"Claimed office in GREEN City: {city.name}")

                game.check_for_east_west_connection()
                check_if_game_over()
            return True
    print("Please click on a GREEN City!")
    return False

def handle_place_two_tradesmen_from_route(pos, button):
    _, post = find_post_by_position(pos)
    reset_pieces = game.current_player.holding_pieces

    # Check if a post was clicked and is empty
    if post and not post.is_owned():
        shape_clicked = 'circle' if button == 3 else 'square'

        if shape_clicked in game.current_player.holding_pieces:
            post.claim(game.current_player, shape_clicked)
            reset_pieces.remove(shape_clicked)
            game.current_player.pieces_to_place -= 1

            # Update the general stock for the current player
            if shape_clicked == 'circle':
                game.current_player.general_stock_circles -= 1
            else:
                game.current_player.general_stock_squares -= 1

            remaining_pieces = ', '.join(reset_pieces)
            print(f"BM: Please place {game.current_player.pieces_to_place} more piece(s) of {remaining_pieces} on valid posts!")
        else:
            print(f"Cannot place {shape_clicked}. Available pieces: {', '.join(reset_pieces)}.")
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
    waiting_for_click = True
    if bm.type == 'PlaceAdjacent':
        print ("Only can be done if route is full.")
        print ("If route is full, clicking on the city will automatically handle this.")
        return True
    
    player.used_bonus_markers.append(bm)
    player.bonus_markers.remove(bm)

    if bm.type == 'SwapOffice':
        print("Click a City to swap offices on.")
    elif bm.type == 'Move3':
        player.pieces_to_place = 3  # Set the pieces to move to 3 as per the bonus marker
        print("You can now move up to 3 opponent's pieces. Click on an opponent's piece to move it.")
        game.waiting_for_bm_move3 = True
    elif bm.type == 'UpgradeAbility':
        print("Please click on an upgrade to choose it.")
    elif bm.type == '3Actions':
        player.actions_remaining += 3
        return True
    elif bm.type == '4Actions':
        player.actions_remaining += 4
        return True
    else:
        return False

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
                else:
                    print("Invalid scenario.")
        redraw_window(win, game)
        pygame.display.flip()  # Update the screen
        pygame.time.wait(100)
    return True

def handle_click(pos, button):
    check_if_post_clicked(pos, button)
    check_if_route_claimed(pos,button)
    check_if_income_clicked(pos)                        

def displaced_click(pos, button):
    route, post = find_post_by_position(pos)  

    if post not in game.all_empty_posts:
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
        
def reset_valid_posts_to_displace_to():
    for post in game.all_empty_posts:
        post.reset_post()
    game.all_empty_posts.clear()

def handle_end_turn_click(pos):
    if (game.selected_map.map_width+300 < pos[0] < game.selected_map.map_width+300 + 200 and
        game.selected_map.map_height-170 < pos[1] < game.selected_map.map_height-170 + 170):
        print ("End Turn clicked!")
        game.current_player.ending_turn = True
    else:
        print("Please click End Turn, or a Bonus Marker that you can use!")

def check_if_player_has_usable_BMs():
    return game.current_player.bonus_markers and not all(bm.type == 'PlaceAdjacent' for bm in game.current_player.bonus_markers)

# Constants for input and output sizes
INPUT_SIZE = 2309
OUTPUT_SIZE = 671

# Create the neural network instance
hansa_nn = HansaNN(INPUT_SIZE, OUTPUT_SIZE)

# Define the optimizer
optimizer = optim.Adam(hansa_nn.parameters(), lr=0.001)

# Loop 5 times
for _ in range(10):
    # Get the current game state tensor
    game_state_tensor = get_game_state(game)
    game_state_tensor = game_state_tensor.float()   # Convert to float if not already

    # Forward pass through the neural network to get the action probabilities
    action_probabilities = hansa_nn(game_state_tensor.unsqueeze(0))  # Adding an extra dimension for batch

    # Apply masking to filter out invalid actions
    valid_actions = masking_out_invalid_actions(game)
    masked_action_probabilities = action_probabilities * valid_actions

    # Print the action probabilities after masking
    print("Action probabilities after masking:")
    print(masked_action_probabilities)

    # Find the index of the maximum value in the masked probabilities
    max_prob_index = torch.argmax(masked_action_probabilities).item()

    # Perform the action corresponding to the maximum probability index
    print(f"Iteration {_ + 1}, Action with highest probability index: {max_prob_index}")
    perform_action_from_index(game, max_prob_index)

    # Calculate reward (or penalty) for the action
    # reward = ... (This should be determined based on your game's mechanics)

    # Loss calculation (using MSE loss as an example)
    # Note: You might need to adjust this depending on how you define rewards and outputs
    # loss = F.mse_loss(masked_action_probabilities, reward)

    # # Zero gradients, perform a backward pass, and update the weights.
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # Print information for each iteration

# exit()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONUP:
            get_available_actions(game)
            mouse_position = pygame.mouse.get_pos()
            if(check_if_bm_clicked(mouse_position)):
                print(f"Bonus Marker was used!")
            # while game_not_over:
            elif game.waiting_for_displaced_player:
                displaced_click(mouse_position, event.button)
                if not game.all_empty_posts:
                    print("No empty posts found initially. Searching for adjacent routes...")  # Debugging log
                    game.all_empty_posts = gather_empty_posts(game.original_route_of_displacement)
                    if not game.all_empty_posts:
                        print("No empty posts found in adjacent routes either!")  # Debugging log
                    
                    post_count = 0  # Counter for the number of posts processed
                    for post in game.all_empty_posts:
                        post.valid_post_to_displace_to()
                        post_count += 1
                    print(f"Processed {post_count} posts.")  # Debugging log
                if game.displaced_player.all_pieces_placed():
                    reset_valid_posts_to_displace_to()
                    game.original_route_of_displacement = None
                    game.displaced_player.reset_displaced_player()
                    game.waiting_for_displaced_player = False
                    game.current_player.actions_remaining -= 1
                    # game.switch_player_if_needed()

            elif game.current_player.holding_pieces :
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
                elif not check_if_player_has_usable_BMs() and game.replace_bonus_marker == 0:
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