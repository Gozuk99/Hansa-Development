import pygame
import sys
from map_data.map1 import Map1
from player_info.player_attributes import Player, PlayerBoard
from map_data.constants import WHITE, GREEN, BLUE, PURPLE, RED, YELLOW, BLACK, CIRCLE_RADIUS, TAN, COLOR_NAMES
from map_data.map_attributes import Post, City
from drawing.drawing_utils import draw_line, redraw_window, draw_end_game

selected_map = Map1()
WIDTH = selected_map.map_width+800
HEIGHT = selected_map.map_height
cities = selected_map.cities
routes = selected_map.routes

pygame.init()

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Hansa Sample Game')
win.fill(TAN)

for route in routes:
    draw_line(win, WHITE, route.cities[0].midpoint, route.cities[1].midpoint, 10, 2)

# Define players
players = [Player(GREEN, 1), Player(BLUE, 2), Player(PURPLE, 3), Player(RED, 4), Player(YELLOW, 5)]
player_boards = [PlayerBoard(WIDTH-800, i * 220, player) for i, player in enumerate(players)]
current_player = players[0]

def find_empty_post_in_adjacent_routes(city, route_to_exclude):
    for adjacent_route in city.routes:
        if adjacent_route != route_to_exclude:
            empty_post = adjacent_route.find_empty_post()
            if empty_post:
                return empty_post
    return None

def find_empty_post_in_next_level_routes(city, route_to_exclude):
    for adjacent_route in city.routes:
        if adjacent_route != route_to_exclude:
            for next_city in adjacent_route.cities:
                if next_city != city:
                    empty_post = find_empty_post_in_adjacent_routes(next_city, adjacent_route)
                    if empty_post:
                        return empty_post
    return None

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
    # Check for the game-ending condition after all points have been allocated
    for player in players:
        if player.score >= 3:
            end_game(player)

def handle_click(pos, button):
    route, post = find_post_by_position(pos)
    if post:
        if button == 1:  # left click
            if post.can_be_claimed_by(current_player, "square"):
                handle_left_click(post)
            elif post.is_owned() and post.owner != current_player:
                handle_displacement(post, route, "square")
        elif button == 3:  # right click
            if post.can_be_claimed_by(current_player, "circle"):
                handle_right_click(post)
            elif post.is_owned() and post.owner != current_player:
                handle_displacement(post, route, "circle")
        return

    for city in cities:  # iterate through cities
        if ((city.pos[0] < pos[0] < city.pos[0] + city.width) and
            (city.pos[1] < pos[1] < city.pos[1] + city.height) and
            button == 1 and current_player.actions > 0):
            
            # Print details for debugging
            print(f"Clicked on city {city.name}:")

            for route in city.routes:
                route_city_1, route_city_2 = route.cities
                print(f"  Route from {route_city_1.name} to {route_city_2.name}:")
                for post in route.posts:
                    owner_name = COLOR_NAMES[post.owner.color] if post.owner else "None"
                    print(f"    Post at position {post.pos}: owned by {owner_name}")

                if not route.is_controlled_by(current_player):
                    continue

                if city.controller or not city.has_required_piece_shape(current_player, route, city):
                    continue

                score_route(route)
                city.update_office_ownership(current_player, current_player.color)
                for post in route.posts:
                    if post.owner == current_player:
                        post.reset_post(post)

                current_player.actions -= 1
                check_and_switch_player()

                for player in players:
                    if player.score >= 3:
                        end_game(player)
                return

    # Check if any player board's Income Action button was clicked
    for board in player_boards:
        if hasattr(board, 'circle_buttons'):
            for idx, button_rect in enumerate(board.circle_buttons):
                if button_rect.collidepoint(pos):
                    board.income_action_based_on_circle_count(idx)
                    current_player.actions -= 1
                    check_and_switch_player()
                    return

def handle_displacement(post, route, displacing_piece_shape):
    # Determine the shape of the piece that will be displaced
    displaced_piece_shape = post.owner_piece_shape
    # Calculate the cost to displace
    cost = 2 if displaced_piece_shape == "square" else 3

    # Check if the current player has the required pieces
    if current_player.personal_supply_squares + current_player.personal_supply_circles < cost:
        return  # Not enough pieces, action cannot be performed

    displaced_player_color = post.owner.color

    # Depending on the click type, decide the shape and color to place on the post
    if displacing_piece_shape == "square":
        placed_shape = "square"
        post.circle_color = TAN
        post.square_color = current_player.color
    elif displacing_piece_shape == "circle":
        if current_player.personal_supply_circles == 0:
            return  # Invalid action if trying to place circle without having one
        placed_shape = "circle"
        post.circle_color = current_player.color
        post.square_color = TAN
    else:
        sys.exit()
    post.owner_piece_shape = placed_shape
    post.owner = current_player
    current_player.actions -= 1

    # Handle the cost payment with priority to squares
    squares_used = min(current_player.personal_supply_squares, cost)
    circles_used = cost - squares_used

    current_player.personal_supply_squares -= squares_used
    current_player.personal_supply_circles -= circles_used
    current_player.general_stock_squares += (cost - squares_used)  # Retur

    displaced = False
    for city in route.cities:
        empty_post = find_empty_post_in_adjacent_routes(city, route)
        if empty_post:
            set_displaced_post_color(empty_post, displaced_piece_shape, displaced_player_color, post.owner)
            displaced = True
            check_and_switch_player()
            return  # Exit the function after successful displacement

    if not displaced:
        for city in route.cities:
            empty_post = find_empty_post_in_next_level_routes(city, route)
            if empty_post:
                set_displaced_post_color(empty_post, displaced_piece_shape, displaced_player_color, post.owner)
                check_and_switch_player()
                return  # Exit the function after successful displacement

def set_displaced_post_color(empty_post, displaced_piece_shape, color, owner):
    if displaced_piece_shape == "circle":
        empty_post.circle_color = color
        empty_post.square_color = TAN
    else:  # it's a square
        empty_post.square_color = color
        empty_post.circle_color = TAN

    empty_post.owner_piece_shape = displaced_piece_shape
    empty_post.owner = owner  # Assigning the ownership of the displaced post.

def find_post_by_position(pos):
    for route in routes:
        for post in route.posts:
            if abs(post.pos[0] - pos[0]) < CIRCLE_RADIUS and abs(post.pos[1] - pos[1]) < CIRCLE_RADIUS:
                return route, post
    return None, None

def check_and_switch_player():
    if current_player.actions == 0:
        next_player()

def handle_left_click(post):
    if current_player.personal_supply_squares > 0:
        post.claim(current_player, "square")
        current_player.actions -= 1
        current_player.personal_supply_squares -= 1
        check_and_switch_player()
        
def handle_right_click(post):
    if current_player.personal_supply_circles > 0:
        post.claim(current_player, "circle")
        current_player.actions -= 1
        current_player.personal_supply_circles -= 1
        check_and_switch_player()

def next_player():
    global current_player
    current_player = players[(players.index(current_player)+1)%len(players)]
    current_player.actions = 2 if current_player.color == GREEN else 1

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONUP:
            handle_click(pygame.mouse.get_pos(), event.button)

    redraw_window(win, cities, routes, current_player, WIDTH, HEIGHT)

    # In the game loop:
    for player_board in player_boards:
        player_board.draw(win, current_player)

    pygame.display.flip()  # Update the screen
    pygame.time.delay(100)
