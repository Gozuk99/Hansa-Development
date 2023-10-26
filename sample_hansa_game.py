import pygame
import sys
from map_data.map1 import Map1
from player_info.player_attributes import Player, PlayerBoard
from map_data import constants

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
    pygame.draw.line(win, constants.WHITE, route.cities[0].midpoint, route.cities[1].midpoint, 10)

# Define players
players = [Player(constants.GREEN), Player(constants.BLUE), Player(constants.PURPLE), Player(constants.PINK), Player(constants.WHITE)]
# scoreboard = Scoreboard(players)
# player_boards = [PlayerBoard(WIDTH-800, i * 220) for i in range(len(players))]
player_boards = [PlayerBoard(WIDTH-800, i * 220, player.color) for i, player in enumerate(players)]
current_player = players[0]

def redraw_window():
    # Draw cities and their offices
    for city in cities:
        # Calculate the position of the rectangle
        rect_x = city.pos[0]
        rect_y = city.pos[1]

        # Draw the city itself with the calculated RECT_WIDTH and RECT_HEIGHT
        pygame.draw.rect(win, city.color, (rect_x, rect_y, city.width, city.height))
        text = font.render(city.name, True, constants.BLACK)
        text_width = text.get_width()
        text_x = city.pos[0] + (city.width - text_width) // 2
        text_y = city.pos[1] + city.height  # Place text just below the city rectangle
        win.blit(text, (text_x, text_y))

        start_x = rect_x + constants.BUFFER  # Starting x-coordinate within the rectangle
        start_y = rect_y + city.height // 2 - constants.SQUARE_SIZE // 2  # Centered vertically in the rectangle

        for office in city.offices:
            if office.office_type == "square":
                # Calculate the x-coordinate of the square office
                square_x = start_x
                square_y = rect_y + city.height // 2 - constants.SQUARE_SIZE // 2
                pygame.draw.rect(win, office.color, (square_x, square_y, constants.SQUARE_SIZE, constants.SQUARE_SIZE))
                # Update the starting x-coordinate for the next office (square)
                start_x += constants.SQUARE_SIZE + constants.SPACING
            else:
                # Calculate the x-coordinate of the circle office
                circle_x = start_x + constants.CIRCLE_RADIUS
                circle_y = start_y + constants.SQUARE_SIZE // 2
                pygame.draw.circle(win, office.color, (circle_x, circle_y), constants.CIRCLE_RADIUS)
                # Update the starting x-coordinate for the next office (circle)
                start_x += constants.CIRCLE_RADIUS * 2 + constants.SPACING
            for route in routes:
                for post in route.posts:
                    # print(f"Drawing post at position {post.pos}")
                    pygame.draw.circle(win, post.color, post.pos, constants.POST_RADIUS)

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
                if button == 1 and post.color == constants.BLACK and current_player.actions > 0:  # left click
                    post.color = current_player.color
                    current_player.actions -= 1
                    if current_player.actions == 0:
                        next_player()
                    return  # an action was performed, so return

                elif button == 3 and post.color != constants.BLACK and post.color != current_player.color and current_player.actions > 0:  # right click
                    displaced_player_color = post.color
                    post.color = current_player.color
                    current_player.actions -= 1
                    if current_player.actions == 0:
                        next_player()
                    displaced = False
                    for city in route.cities:
                        empty_post = find_empty_post_in_adjacent_routes(city, route)
                        if empty_post:
                            empty_post.color = displaced_player_color
                            displaced = True
                            break
                    if not displaced:
                        for city in route.cities:
                            empty_post = find_empty_post_in_next_level_routes(city, route)
                            if empty_post:
                                empty_post.color = displaced_player_color
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
    for board in player_boards:
        board.draw(win)
    pygame.display.flip()  # Update the screen
    pygame.time.delay(100)
