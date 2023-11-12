import pygame
import sys
from map_data.map1 import Map1
from player_info.player_attributes import Player, DisplacedPlayer, PlayerBoard, UPGRADE_METHODS_MAP, UPGRADE_MAX_VALUES
from map_data.constants import WHITE, GREEN, BLUE, PURPLE, RED, YELLOW, BLACK, CIRCLE_RADIUS, TAN, COLOR_NAMES, PRIVILEGE_COLORS
from map_data.map_attributes import Route, Post, City
from game_info.game_attributes import Game
from drawing.drawing_utils import draw_line, redraw_window, draw_scoreboard, draw_end_game, draw_bonus_markers, draw_upgrades

game = Game(map_num=1, num_players=5)
WIDTH = game.selected_map.map_width+800
HEIGHT = game.selected_map.map_height
cities = game.selected_map.cities
routes = game.selected_map.routes
upgrade_cities = game.selected_map.upgrades
specialprestigepoints_city = game.selected_map.specialprestigepoints

pygame.init()

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Hansa Sample Game')
win.fill(TAN)

displaced_player = game.displaced_player
player_boards = [PlayerBoard(WIDTH-800, i * 220, player) for i, player in enumerate(game.players)]

def end_game(winning_player):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_RETURN:
                    pygame.quit()
                    sys.exit()

        draw_end_game(win, winning_player)

def score_route(route):
    # Allocate points
    for city in route.cities:
        player = city.get_controller()
        if player is not None:
            player.score += 1
            print(f"Player {COLOR_NAMES[player.color]} scored for controlling {city.name}, total score: {player.score}")

def check_bounds(item, pos):
    return (item.x_pos < pos[0] < item.x_pos + item.width and
            item.y_pos < pos[1] < item.y_pos + item.height)

def check_if_post_clicked(pos, button):
    route, post = find_post_by_position(pos)
    if post:
        if button == 1:  # left click
            if post.can_be_claimed_by("square"):
                claim_post(post, game.current_player, "square")
            elif post.is_owned() and post.owner != game.current_player:
                handle_displacement(post, route, "square")
            elif post.is_owned() and post.owner == game.current_player:
                handle_move(pos, button)
        elif button == 3:  # right click
            if post.can_be_claimed_by("circle"):
                claim_post(post, game.current_player, "circle")
            elif post.is_owned() and post.owner != game.current_player:
                handle_displacement(post, route, "circle")
        elif button == 2:  # middle click
            post.DEBUG_print_post_details()
            return
        game.switch_player_if_needed()
        return

def assign_new_bonus_marker_on_route(pos, button):
    # Ensure left click
    if button != 1:  # Assuming 1 is the left click
        print("No action taken: Click was not a left click.")
        return

    # Find the post and route by position
    route, post = find_post_by_position(pos)

    if not route:
        print("Invalid BM Placement: Clicked position does not correspond to any route.")
        return

    if route.bonus_marker:
        print(f"Invalid BM Placement: Route between {route.cities[0].name} and {route.cities[1].name} already has a bonus marker.")
        return

    if route.has_tradesmen():
        print(f"Invalid BM Placement: Route between {route.cities[0].name} and {route.cities[1].name} has tradesmen on it.")
        return

    if not route.has_empty_office_in_cities():
        print(f"Invalid BM Placement: Both cities at the ends of the route between {route.cities[0].name} and {route.cities[1].name} have no empty offices.")
        return

    if game.selected_map.bonus_marker_pool:
        bm_type = game.selected_map.bonus_marker_pool.pop()
        route.assign_map_new_bonus_marker(bm_type)  # Create a new BonusMarker instance with the type
        print(f"Bonus marker '{bm_type}' has been placed on the route between {route.cities[0].name} and {route.cities[1].name}.")
        # Set the flag to False after placing the new bonus marker
        game.replace_bonus_marker -= 1
        # Now you might want to update the display or the route to show the bonus marker
        # draw_bonus_marker_on_route(route)  # Implement this function as needed
    else:
        print("No action taken: No bonus markers available in the pool.")

def handle_move(pos, button):
    # Find the clicked post based on position
    route, post = find_post_by_position(pos)

    # If no post was found, simply return without doing anything
    if post is None:
        return

    # If the player is picking up pieces
    if button == 1 and post.owner == game.current_player:
        # If we have not started a move yet, start one
        if game.current_player.pieces_to_place is None:
            game.current_player.start_move()
        # Attempt to pick up a piece if within the pieces to move limit (book)
        game.current_player.pick_up_piece(post)

    # If the player has pieces in hand to place
    elif game.current_player.holding_pieces:
        if not post.is_owned():
            game.current_player.place_piece(post, button)

        # If no pieces are left to place, finish the move
        if not game.current_player.holding_pieces:
            game.current_player.finish_move()
            game.switch_player_if_needed()  # Check if out of actions

# And the corresponding changes in the handle_move_opponent function:
def handle_move_opponent(pos, button):
    route, post = find_post_by_position(pos)

    if post is None:
        return

    if button == 1 and post.is_owned() and post.owner != game.current_player:
        if game.current_player.pieces_to_place > 0:
            game.current_player.pick_up_piece(post)

    elif game.current_player.holding_pieces:
        if not post.is_owned():
            game.current_player.place_piece(post, button)
            if not game.current_player.holding_pieces:
                game.current_player.finish_move()
                game.waiting_for_bm_move3_choice = False
        else:
            print("Cannot place a piece here. The post is already occupied.")

def find_clicked_city(cities, pos):
    for city in cities:
        if check_bounds(city, pos):
            print(f"Clicked on city: {city.name}")
            return city
    return None

def finalize_route_claim(route, placed_piece_shape):
    update_stock_and_reset(route, game.current_player, placed_piece_shape)
    handle_bonus_marker(game.current_player, route)
    game.current_player.actions_remaining -= 1
    game.check_for_east_west_connection()
    game.switch_player_if_needed()
    check_for_game_end()

def check_if_route_claimed(pos, button):
    placed_piece_shape = None

    if game.current_player.actions <= 0:
        return

    city = find_clicked_city(game.selected_map.cities, pos)

    if city:
        print(f"Clicked on city: {city.name}")
        for route in city.routes:
            if route.is_controlled_by(game.current_player):
                if button == 1: # leftclick
                    next_open_office_color = city.get_next_open_office_color()
                    print(f"Next open office color: {next_open_office_color}")

                    if game.current_player.player_can_claim_office(next_open_office_color):
                        if not city.has_required_piece_shape(game.current_player, route, city):
                            required_shape = city.get_next_open_office_shape()
                            print(f"{COLOR_NAMES[game.current_player.color]} tried to claim an office in {city.name} but doesn't have the required {required_shape} shape on the route.")
                        else:
                            score_route(route)
                            placed_piece_shape = city.get_next_open_office_shape()
                            print(f"{COLOR_NAMES[game.current_player.color]} placed a {placed_piece_shape.upper()} into an office of {city.name}.")
                            city.update_office_ownership(game.current_player, game.current_player.color)
                            finalize_route_claim(route, placed_piece_shape)
                    elif 'PlaceAdjacent' in (bm.type for bm in game.current_player.bonus_markers) or (city.city_is_full()):
                        score_route(route)
                        city.claim_office_with_bonus_marker(game.current_player)
                        print(f"{COLOR_NAMES[game.current_player.color]} placed a square into a NEW office of {city.name}.")
                        finalize_route_claim(route, "square")
                    else:
                        print(f"{COLOR_NAMES[game.current_player.color]} doesn't have the correct privilege - {game.current_player.privilege} - to claim an office in {city.name}.")

                elif button == 2: #middle click
                    if city.upgrade_city_type == "SpecialPrestigePoints":
                        if specialprestigepoints_city.claim_highest_prestige(game.current_player):
                            specialprestigepoints_city.draw_special_prestige_points(win)
                            score_route(route)
                            finalize_route_claim(route, placed_piece_shape)
                    elif city.upgrade_city_type in ["Keys", "Privilege", "Book", "Actions", "Bank"]:
                        if game.current_player.perform_upgrade(city.upgrade_city_type):
                            score_route(route)
                            finalize_route_claim(route, placed_piece_shape)
                    else:
                        print(f"Cannot upgrade an ability by Middle clicking {city.name}")
                elif button == 3: #right click
                    score_route(route)
                    finalize_route_claim(route, placed_piece_shape)
                else:
                    print(f"Invalid scenario with {city.name}")
                return
            else:
                print(f"Route(s) not controlled by current_player: {COLOR_NAMES[game.current_player.color]}")

# def upgrade_clicked(pos):

# def claim_office_clicked(pos):

# def claim_route_for_points_clicked(pos):

# def special_bonus_points_clicked(pos):

def check_if_income_clicked(pos):
    # Check if any player board's Income Action button was clicked
    for board in player_boards:
        if hasattr(board, 'circle_buttons'):
            for idx, button_rect in enumerate(board.circle_buttons):
                if button_rect.collidepoint(pos):
                    board.income_action_based_on_circle_count(idx)
                    game.current_player.actions_remaining -= 1
                    game.switch_player_if_needed()
                    return

def handle_bonus_marker(player, route):
    if route.bonus_marker:
        route.bonus_marker.owner = player
        player.bonus_markers.append(route.bonus_marker)
        route.bonus_marker = None
        game.replace_bonus_marker += 1

# def check_if_route_claimed(pos, button):
#     # if button == 1:
#     #     if upgrade_clicked(pos):
#     #         return
#     #     if claim_office_clicked(pos):
#     #         return
#     #     if special_bonus_points_clicked(pos):
#     #         return
#     # elif button == 3:
#     #     if claim_route_for_points_clicked(pos):
#     #         return
#     claim_route_clicked(pos, button)

def check_if_bm_clicked(pos):
    for bm in game.current_player.bonus_markers:
        if bm.is_clicked(pos):
            print(f"Clicked on bonus marker: {bm.type}")
            bm.use_bm(game)

def handle_click(pos, button):
    check_if_post_clicked(pos, button)
    check_if_route_claimed(pos,button)
    check_if_income_clicked(pos)            
    check_if_bm_clicked(pos)            

def try_claim_office(route, city, player):
    if not city.has_required_piece_shape(player, route, city):
        required_shape = city.get_next_open_office_shape()
        print(f"{COLOR_NAMES[player.color]} tried to claim an office in {city.name} but doesn't have the required {required_shape} shape on the route.")
        return False
    return True

def handle_route_for_city_claim(route, city, player):
    """Attempt to claim a route for a city and handle the action."""
    # if try_claim_office(route, city, player):
    if not city.has_required_piece_shape(player, route, city):
        required_shape = city.get_next_open_office_shape()
        print(f"{COLOR_NAMES[player.color]} tried to claim an office in {city.name} but doesn't have the required {required_shape} shape on the route.")
        return False
    else:
        score_route(route)
        placed_piece_shape = city.get_next_open_office_shape()
        print(f"{COLOR_NAMES[player.color]} placed a {placed_piece_shape.upper()} into an office of {city.name}.")
        update_stock_and_reset(route, player, placed_piece_shape)
        city.update_office_ownership(player, player.color)
        player.actions_remaining -= 1
        handle_bonus_marker(game.current_player, route)
        game.switch_player_if_needed()
        check_for_game_end()
        return True

def check_for_game_end():
    # Check if the bonus marker pool is empty
    if not game.selected_map.bonus_marker_pool:
        # Find the player with the highest score
        highest_scoring_player = max(game.players, key=lambda p: p.score)
        # End the game with the player who has the highest score
        end_game(highest_scoring_player)
    else:
        # If the bonus marker pool is not empty, check if any player has reached the score threshold
        for player in game.players:
            if player.score >= 3:
                end_game(player)
                break

def update_stock_and_reset(route, player, placed_piece_shape=None):
    """Update player's general stock based on pieces on the route and reset those posts."""
    circles_on_route = sum(1 for post in route.posts if post.owner == player and post.owner_piece_shape == "circle")
    squares_on_route = sum(1 for post in route.posts if post.owner == player and post.owner_piece_shape == "square")

    if placed_piece_shape == "circle":
        circles_on_route -= 1
    elif placed_piece_shape == "square":
        squares_on_route -= 1

    # Update the player's general supply
    player.general_stock_circles += circles_on_route
    player.general_stock_squares += squares_on_route

    for post in route.posts:
        if post.owner == player:
            post.reset_post()

def displaced_click(pos, button):
    route, post = find_post_by_position(pos)  

    if post not in game.all_empty_posts:
        return

    # Determine which shape the player wants to place based on the button they clicked
    if button == 1:
        desired_shape = "square"
    elif button == 3:
        desired_shape = "circle"
    else:
        return

    # Check if the player is forced to use the displaced shape
    must_use_displaced_piece = False
    if not displaced_player.played_displaced_shape:
        if displaced_player.displaced_shape == "circle" and displaced_player.total_pieces_to_place == 1:
            must_use_displaced_piece = True
        elif displaced_player.displaced_shape == "square" and displaced_player.total_pieces_to_place == 1:
            must_use_displaced_piece = True

    if must_use_displaced_piece and desired_shape != displaced_player.displaced_shape:
        print("Invalid action. Must use the displaced piece.")
        return

    wants_to_use_displaced_piece = (not displaced_player.played_displaced_shape) and (desired_shape == displaced_player.displaced_shape)
    
    if wants_to_use_displaced_piece:
        print(f"Attempting to place a {desired_shape} while Displaced Shape has NOT been played yet")
        displace_to(post, displaced_player, desired_shape, use_displaced_piece=True)
    else:
        displace_to(post, displaced_player, desired_shape)

def displace_to(post, displaced_player, shape, use_displaced_piece=False):
    if use_displaced_piece:
        print(f"Attempting to use displaced piece {shape} - no affect to the GS or PS")
        claim_and_update(post, displaced_player, shape, use_displaced_piece=True)
    else:
        if has_general_stock(displaced_player, shape):
            print(f"Attempting to place a {shape} from general_stock, because use_displaced is false")
            claim_and_update(post, displaced_player, shape)
        elif is_general_stock_empty(displaced_player) and has_personal_supply(displaced_player, shape):
            print(f"Attempting to place a {shape} from personal_supply, because use_displaced is false")
            claim_and_update(post, displaced_player, shape, from_personal_supply=True)
        else:
            print(f"Cannot place a {shape} because the general stock and personal supply are empty.")

def has_general_stock(displaced_player, shape):
    if shape == "square":
        return displaced_player.player.general_stock_squares > 0
    return displaced_player.player.general_stock_circles > 0

def is_general_stock_empty(displaced_player):
    return displaced_player.player.general_stock_squares == 0 and displaced_player.player.general_stock_circles == 0

def has_personal_supply(displaced_player, shape):
    if shape == "square":
        return displaced_player.player.personal_supply_squares > 0
    return displaced_player.player.personal_supply_circles > 0

def claim_and_update(post, displaced_player, shape, use_displaced_piece=False, from_personal_supply=False):
    post.claim(displaced_player.player, shape)
    game.all_empty_posts.remove(post)
    
    if use_displaced_piece:
        displaced_player.played_displaced_shape = True
    elif not from_personal_supply:
        if shape == "square":
            displaced_player.player.general_stock_squares -= 1
        else:
            displaced_player.player.general_stock_circles -= 1
    else:
        if shape == "square":
            displaced_player.player.personal_supply_squares -= 1
        else:
            displaced_player.player.personal_supply_circles -= 1
    
    displaced_player.total_pieces_to_place -= 1

def handle_displacement(post, route, displacing_piece_shape):
    # Determine the shape of the piece that will be displaced
    displaced_piece_shape = post.owner_piece_shape
    current_displaced_player = post.owner

    # Calculate the cost to displace
    cost = 2 if displaced_piece_shape == "square" else 3

    # Check if the current player has enough tradesmen
    if game.current_player.personal_supply_squares + game.current_player.personal_supply_circles < cost:
        return  # Not enough tradesmen, action cannot be performed

    # Before displacing with a square, check if the current player has a square in their personal supply
    if displacing_piece_shape == "square" and game.current_player.personal_supply_squares == 0:
        return  # Invalid move as there's no square in the personal supply

    # Before displacing with a circle, check if the current player has a circle in their personal supply
    if displacing_piece_shape == "circle" and game.current_player.personal_supply_circles == 0:
        return  # Invalid move as there's no circle in the personal supply

    # Handle the cost of displacement with priority to squares
    squares_to_pay = min(game.current_player.personal_supply_squares, cost - 1)  # Subtract 1 for the piece being placed
    circles_to_pay = cost - 1 - squares_to_pay

    game.current_player.personal_supply_squares -= squares_to_pay
    game.current_player.personal_supply_circles -= circles_to_pay

    game.current_player.general_stock_squares += squares_to_pay
    game.current_player.general_stock_circles += circles_to_pay

    # Handle the piece being placed
    if displacing_piece_shape == "square":
        game.current_player.personal_supply_squares -= 1
        post.circle_color = TAN
        post.square_color = game.current_player.color
    elif displacing_piece_shape == "circle":
        game.current_player.personal_supply_circles -= 1
        post.circle_color = game.current_player.color
        post.square_color = TAN
    else:
        sys.exit()

    post.owner_piece_shape = displacing_piece_shape
    post.owner = game.current_player

    # Find empty posts on adjacent routes
    game.all_empty_posts = gather_empty_posts(route)
    for post in game.all_empty_posts:
        post.valid_post_to_displace_to()
    # Check conditions based on displaced_piece_shape and number of game.all_empty_posts
    if ((displaced_piece_shape == "square" and len(game.all_empty_posts) == 1) or
        (displaced_piece_shape == "circle" and 1 <= len(game.all_empty_posts) <= 2)):
        game.original_route_of_displacement = route

    game.waiting_for_displaced_player = True
    displaced_player.populate_displaced_player(current_displaced_player, displaced_piece_shape)
    print(f"Waiting for Displaced Player {COLOR_NAMES[displaced_player.player.color]} to place {displaced_player.total_pieces_to_place} tradesmen (circle or square) from their general_stock, one must be {displaced_player.displaced_shape}.")

def gather_empty_posts(start_route):
    visited_routes = [start_route]  # Mark the start route as visited immediately
    queue = get_adjacent_routes(start_route)  # Start with the routes adjacent to the given route

    while queue:
        level_size = len(queue)
        empty_posts = []
        next_level_routes = []  # Routes for the next level
        
        # Process each route in the current level
        for i in range(level_size):
            current_route = queue.pop(0)
            if current_route in visited_routes:
                continue
            visited_routes.append(current_route)

            # Add routes connected to current_route to the next level
            adjacent_routes = get_adjacent_routes(current_route)
            for route in adjacent_routes:
                if route not in visited_routes and route not in next_level_routes:
                    next_level_routes.append(route)
            
            # Check for empty posts in the current route
            for post in current_route.posts:
                if not post.is_owned():
                    empty_posts.append(post)

        # If we've found empty posts at this level, return them
        if empty_posts:
            return empty_posts
        
        # Otherwise, continue to the next level
        queue.extend(next_level_routes)

    return []  # If no empty posts are found after all levels are traversed

def get_adjacent_routes(current_route):
    """Returns a list of routes adjacent to the given route."""
    adjacent_routes = []
    for city in current_route.cities:
        for adjacent_route in city.routes:
            if adjacent_route != current_route and adjacent_route not in adjacent_routes:
                adjacent_routes.append(adjacent_route)
    return adjacent_routes

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

def claim_post(post, player, piece_to_play):
    if piece_to_play == "square" and player.personal_supply_squares > 0:
        post.claim(player, "square")
        player.actions_remaining -= 1
        player.personal_supply_squares -= 1
    elif piece_to_play == "circle" and player.personal_supply_circles > 0:
        post.claim(player, "circle")
        player.actions_remaining -= 1
        player.personal_supply_circles -= 1
        
def reset_valid_posts_to_displace_to():
    for post in game.all_empty_posts:
        post.reset_post()
    game.all_empty_posts.clear()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONUP:
            # while game_not_over:
            if game.waiting_for_displaced_player:
                displaced_click(pygame.mouse.get_pos(), event.button)
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
                if displaced_player.all_pieces_placed():
                    reset_valid_posts_to_displace_to()
                    game.original_route_of_displacement = None
                    displaced_player.reset_displaced_player()
                    game.waiting_for_displaced_player = False
                    game.current_player.actions_remaining -= 1
                    game.switch_player_if_needed()

            elif game.waiting_for_bm_swap_office:
                for city in game.selected_map.cities:
                    if check_bounds(city, pygame.mouse.get_pos()):
                        print(f"Clicked on city: {city.name}")
                        if city.check_if_eligible_to_swap_offices(game.current_player):
                            print ("Valid City to swap offices")
                            city.swap_offices(game.current_player)
                            game.waiting_for_bm_swap_office = False
                        else:
                            print ("Invalid City to Swap offices, please try another city.")

            elif game.waiting_for_bm_move3_choice:
                handle_move_opponent(pygame.mouse.get_pos(), event.button)

            elif game.current_player.holding_pieces and not game.waiting_for_bm_move3_choice:
                handle_move(pygame.mouse.get_pos(), event.button)

            elif game.waiting_for_bm_upgrade_choice:
                for upgrade in upgrade_cities:
                    if check_bounds(upgrade, pygame.mouse.get_pos()):
                        if game.current_player.perform_upgrade(upgrade.upgrade_type):
                            print("Successfully used Upgrade BM")
                            game.waiting_for_bm_upgrade_choice = False
                        else:
                            print("Invalid click when Upgrading via BM")
                            
            elif game.current_player.actions_remaining == 0:
                if game.replace_bonus_marker > 0:
                    # If replace_bonus_marker > 0, the current player must place 1+ new bonus markers
                    assign_new_bonus_marker_on_route(pygame.mouse.get_pos(), event.button)
                game.switch_player_if_needed()

            else:
                handle_click(pygame.mouse.get_pos(), event.button)

    draw_bonus_markers(win, game.selected_map)
    draw_upgrades(win, game.selected_map)
    redraw_window(win, game.selected_map.cities, routes, game.current_player, game.waiting_for_displaced_player, displaced_player, WIDTH, HEIGHT)
    draw_scoreboard(win, game.players, WIDTH-200, HEIGHT-150)
    # In the game loop:
    for player_board in player_boards:
        player_board.draw(win, game.current_player)

    pygame.display.flip()  # Update the screen
    pygame.time.delay(100)
