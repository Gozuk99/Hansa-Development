import pygame
import sys

# Define Constants
WIDTH, HEIGHT = 1000, 1000
CITY_RADIUS = 30
POST_RADIUS = 15
CITY_NAME_OFFSET = 60

# Color variables
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (192, 192, 192)
PINK = (255, 192, 203)
TAN = (210, 180, 140)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)

COLOR_NAMES = {
    GREEN: "GREEN",
    BLUE: "BLUE",
    PURPLE: "PURPLE",
    GREY: "GREY",
    PINK: "PINK",
    BLACK: "BLACK",
    TAN: "TAN",
}

pygame.init()

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Hansa Sample Game')
font = pygame.font.Font(None, 36)

class Player:
    def __init__(self, color, actions):
        self.color = color
        self.actions = actions
        self.score = 0  # Initialize score to 0 for each player

class Post:
    def __init__(self, pos):
        self.pos = pos
        self.color = BLACK

class City:
    def __init__(self, name, pos, color):
        self.name = name
        self.pos = pos
        self.color = color
        self.routes = []
        self.controllers = []  # List of controlling players

    def add_route(self, route):
        self.routes.append(route)

    def change_color(self, new_color):
        self.color = new_color

    def add_controller(self, player):
        self.controllers.append(player)
        
    def get_controller(self):
        # The player in the rightmost post, or the player with the most posts is the one getting points
        controller_counts = {player: self.controllers.count(player) for player in self.controllers}
        return max(controller_counts, key=controller_counts.get)

    def draw(self, win):
        BORDER_THICKNESS = 4
        CITY_SIZE = 10
        DISTANCE_BETWEEN = 6  # Adjust this for the distance between the squares

        if self.name == "China":  # Special case for China
            # Draw the border
            pygame.draw.rect(win, BLACK, (self.pos[0]-CITY_SIZE//2-BORDER_THICKNESS, self.pos[1]-CITY_SIZE//2-BORDER_THICKNESS, 2*CITY_SIZE+DISTANCE_BETWEEN+2*BORDER_THICKNESS, CITY_SIZE+2*BORDER_THICKNESS))

            # Draw left square
            pygame.draw.rect(win, self.color, (self.pos[0]-CITY_SIZE//2, self.pos[1]-CITY_SIZE//2, CITY_SIZE, CITY_SIZE))

            # Draw right square
            pygame.draw.rect(win, self.color, (self.pos[0]+CITY_SIZE//2+DISTANCE_BETWEEN, self.pos[1]-CITY_SIZE//2, CITY_SIZE, CITY_SIZE))
        else:  # For all other cities, draw a circle with a border
            # Draw the border
            pygame.draw.circle(win, BLACK, self.pos, CITY_SIZE+BORDER_THICKNESS)

            # Draw the city
            pygame.draw.circle(win, self.color, self.pos, CITY_SIZE)

class Route:
    def __init__(self, cities, num_posts):
        self.cities = cities
        for city in cities:
            city.add_route(self)
        self.num_posts = num_posts
        self.posts = self.create_posts()

    def create_posts(self):
        city1, city2 = self.cities
        posts = []
        for i in range(1, self.num_posts+1):
            pos = ((city1.pos[0]+i/(self.num_posts+1)*(city2.pos[0]-city1.pos[0])),
                 (city1.pos[1]+i/(self.num_posts+1)*(city2.pos[1]-city1.pos[1])))
            posts.append(Post(pos))
        return posts

    def find_empty_post(self):
        for post in self.posts:
            if post.color == BLACK:
                return post
        return None
    
    def is_controlled_by(self, player):
        return all(post.color == player.color for post in self.posts)

    def is_complete(self):
        for post in self.posts:
            if post.color == BLACK:
                return False
        return True
        
class Scoreboard:
    def __init__(self, players):
        self.players = players
        self.font = pygame.font.SysFont(None, 36)
        self.score_positions = [(WIDTH - 150, i*40 + 20) for i in range(len(players))]

    def draw(self, window):
        for i, player in enumerate(self.players):
            color_name = COLOR_NAMES[player.color]
            score_text = self.font.render(f"{color_name}: {player.score}", True, player.color)
            window.blit(score_text, self.score_positions[i])

# Define cities
USA = City('USA', (WIDTH//5, HEIGHT//5), GREY)
UK = City('UK', (2*WIDTH//5, HEIGHT//5), GREY)
CHINA = City('CHINA', (3*WIDTH//5, HEIGHT//5), GREY)
BRAZIL = City('BRAZIL', (WIDTH//5, 4*HEIGHT//5), PINK)
EGYPT = City('EGYPT', (2*WIDTH//5, 4*HEIGHT//5), PINK)
AUSTRALIA = City('AUSTRALIA', (3*WIDTH//5, 4*HEIGHT//5), PINK)

# Define players
players = [Player(GREEN, 2), Player(BLUE, 1), Player(PURPLE, 1)]
scoreboard = Scoreboard(players)
current_player = players[0]

# Define routes
routes = [
    Route([USA, UK], 3),
    Route([UK, CHINA], 3),
    Route([USA, BRAZIL], 3),
    Route([CHINA, AUSTRALIA], 3),
    Route([BRAZIL, EGYPT], 3),
    Route([EGYPT, AUSTRALIA], 3),
]


def redraw_window():
    win.fill(TAN)
    # Draw scoreboard
    scoreboard.draw(win)
    # Draw routes
    for route in routes:
        pygame.draw.line(win, TAN, route.cities[0].pos, route.cities[1].pos, 6)
        for post in route.posts:
            pygame.draw.circle(win, post.color, post.pos, POST_RADIUS)
    # Draw cities
    for city in [USA, UK, CHINA, BRAZIL, EGYPT, AUSTRALIA]:
        pygame.draw.circle(win, city.color, city.pos, CITY_RADIUS)
        text = font.render(city.name, True, BLACK)
        win.blit(text, (city.pos[0] - text.get_width() // 2, city.pos[1] - CITY_RADIUS - CITY_NAME_OFFSET))
    # Draw current player and remaining actions
    text_color = (0, 0, 0)  # Black color
    player_color_text = font.render(COLOR_NAMES[current_player.color], True, current_player.color)
    actions_text = font.render(f"Actions Remaining: {current_player.actions}", True, text_color)

    # Combine the two text surfaces and blit them onto the window
    text = pygame.Surface((player_color_text.get_width() + actions_text.get_width(), player_color_text.get_height()))
    text.blit(player_color_text, (0, 0))
    text.blit(actions_text, (player_color_text.get_width(), 0))

    win.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 50))

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
        winner_text = font.render(f"Game Over! {COLOR_NAMES[winning_player.color]} wins!", True, winning_player.color)
        rect = winner_text.get_rect()
        rect.center = win.get_rect().center
        win.blit(winner_text, rect)
        pygame.display.update()

def score_route(route):
    # Allocate points
    for city in route.cities:
        if city.controllers:
            for player in players:
                if player == city.get_controller():
                    player.score += 1
                    break

    # Check for game ending condition after all points have been allocated
    for player in players:
        if player.score >= 3:
            end_game(player)

def handle_click(pos, button):
    for route in routes:
        for post in route.posts:
            if abs(post.pos[0]-pos[0]) < POST_RADIUS and abs(post.pos[1]-pos[1]) < POST_RADIUS:
                if button == 1 and post.color == BLACK and current_player.actions > 0:  # left click
                    post.color = current_player.color
                    current_player.actions -= 1
                    if current_player.actions == 0:
                        next_player()
                    return  # an action was performed, so return

                elif button == 3 and post.color != BLACK and post.color != current_player.color and current_player.actions > 0:  # right click
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
                    if route.is_complete():  # check if the route is complete after displacing a post
                        score_route(route)
                        for player in players:
                            if player.score >= 3:
                                end_game(player)
                    return  # an action was performed, so return

    for city in [USA, UK, CHINA, BRAZIL, EGYPT, AUSTRALIA]:  # iterate through cities
        if abs(city.pos[0]-pos[0]) < CITY_RADIUS and abs(city.pos[1]-pos[1]) < CITY_RADIUS and button == 1 and current_player.actions > 0:
            for route in city.routes:
                if route.is_controlled_by(current_player) and city.controller is None:  # check if route is controlled and city is unclaimed
                    score_route(route)  # score the route before claiming the city
                    city.add_controller(current_player)
                    city.color = current_player.color  # change color of the city
                    for post in route.posts:
                        if post.color == current_player.color:  # remove player's posts from route
                            post.color = BLACK
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
    current_player.actions = 2 if current_player.color == GREEN else 1

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONUP:
            handle_click(pygame.mouse.get_pos(), event.button)

    win.fill(BLACK)  # Clear the screen (fill it with BLACK)
    redraw_window()
    scoreboard.draw(win)  # Draw the scoreboard on the window
    pygame.display.flip()  # Update the screen
    pygame.time.delay(100)
