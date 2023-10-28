import pygame
import sys
from map_data.map1 import Map1
from player_info.player_attributes import Player, PlayerBoard
from map_data import constants
from drawing.drawing_utils import draw_shape, draw_text, draw_line

selected_map = Map1()
WIDTH = selected_map.map_width+800
HEIGHT = selected_map.map_height
cities = selected_map.cities
routes = selected_map.routes

pygame.init()

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Hansa Sample Game')
font = pygame.font.Font(None, 36)
win.fill(constants.TAN)
     
class Scoreboard:
    def __init__(self, players):
        self.players = players
        self.font = pygame.font.SysFont(None, 36)
        self.score_positions = [(WIDTH - 800, i*40 + 20) for i in range(len(players))]
        
    def draw(self, window):
        # Clear the area for the scoreboard
        max_height = len(self.players) * 40
        pygame.draw.rect(window, constants.TAN, (WIDTH - 200, 10, 190, max_height))

        for i, player in enumerate(self.players):
            color_name = constants.COLOR_NAMES[player.color]
            score_text = self.font.render(f"{color_name}: {player.score}", True, player.color)
            window.blit(score_text, self.score_positions[i])

for route in routes:
    draw_line(win, constants.WHITE, route.cities[0].midpoint, route.cities[1].midpoint, 10, 2)

# Define players
players = [Player(constants.GREEN, 1), Player(constants.BLUE, 2), Player(constants.PURPLE, 3), Player(constants.RED, 4), Player(constants.YELLOW, 5)]
player_boards = [PlayerBoard(WIDTH-800, i * 220, player.color, player) for i, player in enumerate(players)]
current_player = players[0]

def redraw_window():
    # Draw cities and their offices
    for city in cities:
        # Calculate the position of the rectangle
        rect_x = city.pos[0]
        rect_y = city.pos[1]

        # Use draw_shape function to draw the city rectangle with a border
        draw_shape(win, "rectangle", city.color, rect_x, rect_y, city.width, city.height)
        # Calculate text position to place it just below the city rectangle
        text_width = font.size(city.name)[0]
        text_x = city.pos[0] + (city.width - text_width) // 2
        text_y = city.pos[1] + city.height

        # Use draw_text function to render the city name below the rectangle
        draw_text(win, city.name, text_x, text_y, font, constants.BLACK)

        start_x = rect_x + constants.BUFFER  # Starting x-coordinate within the rectangle
        start_y = rect_y + city.height // 2 - constants.SQUARE_SIZE // 2  # Centered vertically in the rectangle

        for office in city.offices:
            if office.office_type == "square":
                draw_shape(win, "rectangle", office.color, start_x, start_y, constants.SQUARE_SIZE, constants.SQUARE_SIZE)
                start_x += constants.SQUARE_SIZE + constants.SPACING

            else:  
                circle_x = start_x + constants.CIRCLE_RADIUS
                circle_y = start_y + constants.SQUARE_SIZE // 2
                draw_shape(win, "circle", office.color, circle_x, circle_y, constants.CIRCLE_RADIUS)
                start_x += constants.CIRCLE_RADIUS * 2 + constants.SPACING

    for route in routes:
        for post in route.posts:
            post_x, post_y = post.pos

            # Draw the circle
            draw_shape(win, "circle", post.circle_color, post_x, post_y, width=constants.CIRCLE_RADIUS)

            # Draw the square if there is no owner or if the owner has placed a square piece
            if post.owner_piece_shape is None or post.owner_piece_shape == "square":
                draw_shape(win, "rectangle", post.square_color, post_x - constants.SQUARE_SIZE // 2, post_y - constants.SQUARE_SIZE // 2, width=constants.SQUARE_SIZE, height=constants.SQUARE_SIZE)

    text_str = f"Actions: {current_player.actions}"
    combined_text = font.render(text_str, True, constants.COLOR_NAMES[current_player.color])

    # Determine the position and size of the text area
    text_width = combined_text.get_width()
    text_height = combined_text.get_height()
    text_area = pygame.Rect(WIDTH // 2 - text_width // 2, HEIGHT - 50 - text_height // 2, text_width, text_height)

    # Clear the area with the background color
    pygame.draw.rect(win, constants.WHITE, text_area)

    # Draw the combined text onto the main window
    win.blit(combined_text, (WIDTH // 2 - text_width // 2, HEIGHT - 50 - text_height // 2))
    pygame.display.update(text_area)

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

        win.fill((255, 255, 255, 80))  # semi-transparent white background
        winner_text = font.render(f"Game Over! {constants.COLOR_NAMES[winning_player.color]} wins!", True, winning_player.color)
        rect = winner_text.get_rect()
        rect.center = win.get_rect().center
        win.blit(winner_text, rect)
        pygame.display.update()

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
    for route in routes:
        for post in route.posts:
            if abs(post.pos[0]-pos[0]) < constants.POST_RADIUS and abs(post.pos[1]-pos[1]) < constants.POST_RADIUS:
                # No one owns this post, it's free to be claimed.
                if post.owner is None:
                    # Place a circle or square based on left or right click.
                    if button == 1:  # left click for square
                        if (post.required_shape is None or post.required_shape == "square") and current_player.personal_supply_squares > 0:
                            post.square_color = current_player.color
                            post.owner = current_player
                            post.owner_piece_shape = "square"
                            current_player.actions -= 1
                            current_player.personal_supply_squares -= 1
                            if current_player.actions == 0:
                                next_player()
                            return

                    elif button == 3:  # right click for circle
                        if (post.required_shape is None or post.required_shape == "circle") and current_player.personal_supply_circles > 0:
                            post.circle_color = current_player.color
                            post.square_color = current_player.color
                            post.owner = current_player
                            post.owner_piece_shape = "circle"
                            current_player.actions -= 1
                            current_player.personal_supply_circles -= 1
                            if current_player.actions == 0:
                                next_player()
                            return

                # The post is owned.
                elif post.owner != current_player and button == 3:  # Right click on a post owned by someone else.
                    displaced_piece_shape = post.owner_piece_shape

                    # Calculate the cost to displace
                    cost = 2 if displaced_piece_shape == "square" else 3

                    if current_player.personal_supply_squares < cost:
                        return  # Not enough squares, action cannot be performed

                    displaced_player_color = post.owner.color

                    post.circle_color = constants.TAN
                    post.square_color = current_player.color
                    post.owner_piece_shape = "square"
                    post.owner = current_player
                    current_player.actions -= 1
                    current_player.general_stock_squares += (cost - 1)  # Return the extra squares used to general supply
                    current_player.personal_supply_squares -= cost

                    if current_player.actions == 0:
                        next_player()

                    displaced = False
                    for city in route.cities:
                        empty_post = find_empty_post_in_adjacent_routes(city, route)
                        if empty_post:
                            if displaced_piece_shape == "circle":
                                empty_post.circle_color = displaced_player_color
                                empty_post.square_color = constants.TAN
                            else:  # it's a square
                                empty_post.square_color = displaced_player_color
                                empty_post.circle_color = constants.TAN

                            empty_post.owner_piece_shape = displaced_piece_shape
                            empty_post.owner = post.owner  # Assigning the ownership of the displaced post.
                            displaced = True
                            break

                    if not displaced:
                        for city in route.cities:
                            empty_post = find_empty_post_in_next_level_routes(city, route)
                            if empty_post:
                                if displaced_piece_shape == "circle":
                                    empty_post.circle_color = displaced_player_color
                                    empty_post.square_color = constants.TAN
                                else:  # it's a square
                                    empty_post.square_color = displaced_player_color
                                    empty_post.circle_color = constants.TAN

                                empty_post.owner_piece_shape = displaced_piece_shape
                                empty_post.owner = post.owner  # Assigning the ownership of the displaced post.
                                break

                    return  # an action was performed, so return

    for city in cities:  # iterate through cities
        if (city.pos[0] < pos[0] < city.pos[0] + city.width) and (city.pos[1] < pos[1] < city.pos[1] + city.height) and button == 1 and current_player.actions > 0:
            for route in city.routes:
                if route.is_controlled_by(current_player) and city.controller is None:  # check if route is controlled and city is unclaimed
                    score_route(route)  # score the route before claiming the city
                    city.update_office_ownership(current_player, current_player.color)
                    for post in route.posts:
                        if post.color == current_player.color:  # remove player's posts from route
                            post.color = constants.BLACK
                    current_player.actions -= 1
                    if current_player.actions == 0:  # if no actions left for current player
                        next_player()  # switch to the next player
                    for player in players:
                        if player.score >= 3:
                            end_game(player)
                    return  # an action was performed, so return
    # Check if any player board's Income Action button was clicked
    for board in player_boards:
        if hasattr(board, 'circle_buttons'):
            for idx, button_rect in enumerate(board.circle_buttons):
                if button_rect.collidepoint(pos):
                    board.income_action_based_on_circle_count(idx)
                    current_player.actions -= 1
                    if current_player.actions == 0:  # if no actions left for current player
                        next_player()  # switch to the next player
                    return  # an action was performed, so return

def next_player():
    global current_player
    current_player = players[(players.index(current_player)+1)%len(players)]
    current_player.actions = 2 if current_player.color == constants.GREEN else 1

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONUP:
            handle_click(pygame.mouse.get_pos(), event.button)

    redraw_window()
    # scoreboard.draw(win)  # Draw the scoreboard on the window
    # In the game loop:
    for player_board in player_boards:
        player_board.draw(win, current_player)

    pygame.display.flip()  # Update the screen
    pygame.time.delay(100)
