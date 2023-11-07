import pygame
import sys
from map_data.map1 import Map1
from player_info.player_attributes import Player, DisplacedPlayer, PlayerBoard, UPGRADE_METHODS_MAP, UPGRADE_MAX_VALUES
from map_data.constants import WHITE, GREEN, BLUE, PURPLE, RED, YELLOW, BLACK, CIRCLE_RADIUS, TAN, COLOR_NAMES, PRIVILEGE_COLORS
from map_data.map_attributes import Post, City
from drawing.drawing_utils import draw_line, redraw_window, draw_scoreboard, draw_end_game

selected_map = Map1()
WIDTH = selected_map.map_width+800
HEIGHT = selected_map.map_height
cities = selected_map.cities
routes = selected_map.routes
upgrade_cities = selected_map.upgrades
specialprestigepoints_city = selected_map.specialprestigepoints

pygame.init()

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Hansa Sample Game')
win.fill(TAN)

for route in routes:
    draw_line(win, WHITE, route.cities[0].midpoint, route.cities[1].midpoint, 10, 2)
for upgrade_types in upgrade_cities:
    upgrade_types.draw_upgrades_on_map(win)
specialprestigepoints_city.draw_special_prestige_points(win)

# Define players
# players = [Player(GREEN, 1), Player(BLUE, 2), Player(PURPLE, 3)]
players = [Player(GREEN, 1), Player(BLUE, 2), Player(PURPLE, 3), Player(RED, 4), Player(YELLOW, 5)]
displaced_player = DisplacedPlayer()
player_boards = [PlayerBoard(WIDTH-800, i * 220, player) for i, player in enumerate(players)]
current_player = players[0]
current_player.actions_remaining = current_player.actions
waiting_for_displaced_player = False
original_route_of_displacement = None
all_empty_posts = []

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
    # Check for the game-ending condition after all points have been allocated
    check_for_game_end()

def check_bounds(item, pos):
    return (item.x_pos < pos[0] < item.x_pos + item.width and
            item.y_pos < pos[1] < item.y_pos + item.height)

def check_if_post_clicked(pos, button):
    route, post = find_post_by_position(pos)
    if post:
        if button == 1:  # left click
            if post.can_be_claimed_by("square"):
                claim_post(post, current_player, "square")
            elif post.is_owned() and post.owner != current_player:
                handle_displacement(post, route, "square")
            elif post.is_owned() and post.owner == current_player:
                handle_move(pos, button)
        elif button == 3:  # right click
            if post.can_be_claimed_by("circle"):
                claim_post(post, current_player, "circle")
            elif post.is_owned() and post.owner != current_player:
                handle_displacement(post, route, "circle")
        elif button == 2:  # middle click
            post.DEBUG_print_post_details()
            return
        switch_player_if_needed()
        return

def handle_move(pos, button):
    # Find the clicked post based on position
    route, post = find_post_by_position(pos)

    # If no post was found, simply return without doing anything
    if post is None:
        print("No post found at this position.")
        return

    # If the player is picking up pieces
    if button == 1 and post.owner == current_player:
        # If we have not started a move yet, start one
        if current_player.pieces_to_place is None:
            current_player.start_move()
        # Attempt to pick up a piece if within the pieces to move limit (book)
        current_player.pick_up_piece(post)

    # If the player has pieces in hand to place
    elif current_player.holding_pieces:
        # Left-click on an empty post to place a square
        if button == 1 and not post.is_owned():
            current_player.place_piece(post, 'square')
        # Right-click on an empty post to place a circle
        elif button == 3 and not post.is_owned():
            current_player.place_piece(post, 'circle')

        # If no pieces are left to place, finish the move
        if not current_player.holding_pieces:
            current_player.finish_move()
            switch_player_if_needed()  # Check if out of actions

def upgrade_clicked(pos):
    for upgrade in upgrade_cities:
        if check_bounds(upgrade, pos) and current_player.actions > 0:
            associated_city = next(city for city in cities if city.name == upgrade.city_name)

            if any(route.is_controlled_by(current_player) for route in associated_city.routes):
                method_name = UPGRADE_METHODS_MAP[upgrade.upgrade_type.lower()]
                current_value = getattr(current_player, upgrade.upgrade_type.lower())
                max_value = UPGRADE_MAX_VALUES.get(upgrade.upgrade_type.lower())

                if current_value != max_value:
                    upgrade_function = getattr(current_player, method_name)
                    upgrade_function()

                    if upgrade.upgrade_type.lower() in ["keys", "privilege", "actions", "bank"]:
                        current_player.personal_supply_squares += 1
                    elif upgrade.upgrade_type.lower() == "book":
                        current_player.personal_supply_circles += 1

                    for route in associated_city.routes:
                        if route.is_controlled_by(current_player):
                            score_route(route)
                            update_stock_and_reset(route, current_player)
                            current_player.actions_remaining -= 1
                            switch_player_if_needed()
                else:
                    print(f"{upgrade.upgrade_type} is already at its maximum value for player {COLOR_NAMES[current_player.color]}. Action is invalid.")
            else:
                print(f"Route(s) not controlled by current_player: {COLOR_NAMES[current_player.color]}")

def claim_office_clicked(pos):
    for city in cities:
        if city_was_clicked(city, pos) and current_player.actions > 0:
            print(f"Clicked on city: {city.name}")
            next_open_office_color = city.get_next_open_office_color()
            print(f"Next open office color: {next_open_office_color}")

            if player_can_claim_office(current_player, next_open_office_color):
                for route in city.routes:
                    handle_route_for_city_claim(route, city, current_player)
            else:
                print(f"{COLOR_NAMES[current_player.color]} doesn't have the correct privilege - {current_player.privilege} - to claim an office in {city.name}.")

def claim_route_for_points_clicked(pos):
    for city in cities:
        if city_was_clicked(city, pos) and current_player.actions > 0:
            print(f"Clicked on city: {city.name}")
            for route in city.routes:
                if route.is_controlled_by(current_player):
                    score_route(route)
                    update_stock_and_reset(route, current_player)
                    current_player.actions_remaining -= 1
                    switch_player_if_needed()
                    check_for_game_end()

def special_bonus_points_clicked(pos):
    if check_bounds(specialprestigepoints_city, pos) and current_player.actions > 0:
        # Get the city associated with the specialprestigepoints_city using city_name
        associated_city = next(city for city in cities if city.name == specialprestigepoints_city.city_name)
        
        # Directly check the single route associated with the city
        if associated_city.routes[0].is_controlled_by(current_player):
            specialprestigepoints_city.claim_highest_prestige(current_player)
            specialprestigepoints_city.draw_special_prestige_points(win)

def check_if_income_clicked(pos):
    # Check if any player board's Income Action button was clicked
    for board in player_boards:
        if hasattr(board, 'circle_buttons'):
            for idx, button_rect in enumerate(board.circle_buttons):
                if button_rect.collidepoint(pos):
                    board.income_action_based_on_circle_count(idx)
                    current_player.actions_remaining -= 1
                    switch_player_if_needed()
                    return

def check_if_route_claimed(pos, button):
    if button == 1:
        if upgrade_clicked(pos):
            return
        if claim_office_clicked(pos):
            return
        if special_bonus_points_clicked(pos):
            return
    elif button == 3:
        if claim_route_for_points_clicked(pos):
            return

def handle_click(pos, button):
    check_if_post_clicked(pos, button)
    check_if_route_claimed(pos,button)
    check_if_income_clicked(pos)            

def player_can_claim_office(player, office_color):
    """Check if a player can claim an office of the specified color."""
    allowed_office_colors = PRIVILEGE_COLORS[:PRIVILEGE_COLORS.index(player.privilege) + 1]
    return office_color in allowed_office_colors

def try_claim_office(route, city, player):
    if not city.has_required_piece_shape(player, route, city):
        required_shape = city.get_next_open_office_shape()
        print(f"{COLOR_NAMES[player.color]} tried to claim an office in {city.name} but doesn't have the required {required_shape} shape on the route.")
        return False
    return True

def handle_route_for_city_claim(route, city, player):
    """Attempt to claim a route for a city and handle the action."""
    if route.is_controlled_by(player):
        if try_claim_office(route, city, player):
            finalize_route_claim(route, city, player)

def finalize_route_claim(route, city, player):
    score_route(route)
    placed_piece_shape = city.get_next_open_office_shape()
    print(f"Shaped placed into office is {placed_piece_shape}")
    update_stock_and_reset(route, player, placed_piece_shape)
    city.update_office_ownership(player, player.color)
    player.actions_remaining -= 1
    switch_player_if_needed()
    check_for_game_end()

def check_for_game_end():
    for player in players:
        if player.score >= 3:
            end_game(player)

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

def city_was_clicked(city, pos):
    """Check if the city was clicked."""
    return city.pos[0] < pos[0] < city.pos[0] + city.width and city.pos[1] < pos[1] < city.pos[1] + city.height

def displaced_click(pos, button):
    route, post = find_post_by_position(pos)  

    if post not in all_empty_posts:
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
    all_empty_posts.remove(post)
    
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
    global waiting_for_displaced_player, all_empty_posts, original_route_of_displacement
    # Determine the shape of the piece that will be displaced
    displaced_piece_shape = post.owner_piece_shape
    current_displaced_player = post.owner

    # Calculate the cost to displace
    cost = 2 if displaced_piece_shape == "square" else 3

    # Check if the current player has enough tradesmen
    if current_player.personal_supply_squares + current_player.personal_supply_circles < cost:
        return  # Not enough tradesmen, action cannot be performed

    # Before displacing with a square, check if the current player has a square in their personal supply
    if displacing_piece_shape == "square" and current_player.personal_supply_squares == 0:
        return  # Invalid move as there's no square in the personal supply

    # Before displacing with a circle, check if the current player has a circle in their personal supply
    if displacing_piece_shape == "circle" and current_player.personal_supply_circles == 0:
        return  # Invalid move as there's no circle in the personal supply

    # Handle the cost of displacement with priority to squares
    squares_to_pay = min(current_player.personal_supply_squares, cost - 1)  # Subtract 1 for the piece being placed
    circles_to_pay = cost - 1 - squares_to_pay

    current_player.personal_supply_squares -= squares_to_pay
    current_player.personal_supply_circles -= circles_to_pay

    current_player.general_stock_squares += squares_to_pay
    current_player.general_stock_circles += circles_to_pay

    # Handle the piece being placed
    if displacing_piece_shape == "square":
        current_player.personal_supply_squares -= 1
        post.circle_color = TAN
        post.square_color = current_player.color
    elif displacing_piece_shape == "circle":
        current_player.personal_supply_circles -= 1
        post.circle_color = current_player.color
        post.square_color = TAN
    else:
        sys.exit()

    post.owner_piece_shape = displacing_piece_shape
    post.owner = current_player

    # Find empty posts on adjacent routes
    all_empty_posts = gather_empty_posts(route)
    for post in all_empty_posts:
        post.valid_post_to_displace_to()
    # Check conditions based on displaced_piece_shape and number of all_empty_posts
    if ((displaced_piece_shape == "square" and len(all_empty_posts) == 1) or
        (displaced_piece_shape == "circle" and 1 <= len(all_empty_posts) <= 2)):
        original_route_of_displacement = route

    waiting_for_displaced_player = True
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
        
def switch_player_if_needed():
    global current_player
    if current_player.actions_remaining == 0:
        current_player = players[(players.index(current_player) + 1) % len(players)]
        current_player.actions_remaining = current_player.actions

def reset_valid_posts_to_displace_to():
    for post in all_empty_posts:
        post.reset_post()
    all_empty_posts.clear()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONUP:
            # while game_not_over:
            if waiting_for_displaced_player:
                displaced_click(pygame.mouse.get_pos(), event.button)
                if not all_empty_posts:
                    print("No empty posts found initially. Searching for adjacent routes...")  # Debugging log
                    all_empty_posts = gather_empty_posts(original_route_of_displacement)
                    if not all_empty_posts:
                        print("No empty posts found in adjacent routes either!")  # Debugging log
                    
                    post_count = 0  # Counter for the number of posts processed
                    for post in all_empty_posts:
                        post.valid_post_to_displace_to()
                        post_count += 1
                    print(f"Processed {post_count} posts.")  # Debugging log
                if displaced_player.all_pieces_placed():
                    reset_valid_posts_to_displace_to()
                    original_route_of_displacement = None
                    displaced_player.reset_displaced_player()
                    waiting_for_displaced_player = False
                    current_player.actions_remaining -= 1
                    switch_player_if_needed()
            elif current_player.holding_pieces:
                handle_move(pygame.mouse.get_pos(), event.button)
            else:
                handle_click(pygame.mouse.get_pos(), event.button)

    redraw_window(win, cities, routes, current_player, waiting_for_displaced_player, displaced_player, WIDTH, HEIGHT)
    draw_scoreboard(win, players, WIDTH-200, HEIGHT-150)
    # In the game loop:
    for player_board in player_boards:
        player_board.draw(win, current_player)

    pygame.display.flip()  # Update the screen
    pygame.time.delay(100)
