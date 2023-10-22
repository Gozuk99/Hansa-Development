import pygame
import sys

# Define Constants
WIDTH, HEIGHT = 1800, 1255
CITY_RADIUS = 30
POST_RADIUS = 15
CITY_NAME_OFFSET = 60
CIRCLE_RADIUS = 20
SQUARE_SIZE = 26

# Color variables
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (192, 192, 192)
PINK = (255, 192, 203)
TAN = (210, 180, 140)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
BLACKISH_BROWN = (50, 20, 20)

COLOR_NAMES = {
    GREEN: "GREEN",
    BLUE: "BLUE",
    PURPLE: "PURPLE",
    GREY: "GREY",
    PINK: "PINK",
    BLACK: "BLACK",
    TAN: "TAN",
}

cities = []

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

class Office:
    def __init__(self, office_type, color, awards_points):
        self.office_type = office_type  # "circle" or "square"
        self.color = color
        self.awards_points = awards_points
CITY_SIZE = 50

class City:
    def __init__(self, name, position, color):
        self.name = name
        self.pos = position
        self.color = color
        self.routes = []
        self.controllers = []  # List of controlling players
        self.offices = []  # List of offices within the city
        self.width = 0
        self.height = 0
        # self.rect = pygame.Rect(position[0] - (len(self.offices) * SQUARE_SIZE) // 2, position[1], CITY_SIZE + (len(self.offices) * SQUARE_SIZE), CITY_SIZE)

    def add_route(self, route):
        self.routes.append(route)

    def change_color(self, new_color):
        self.color = new_color

    def add_controller(self, player):
        self.controllers.append(player)
    
    def add_office(self, office):
        self.offices.append(office)

    def update_city_size(self, width, height):
        self.width = width
        self.height = height

    def get_controller(self):
        # Sort offices by position from left to right
        sorted_offices = sorted(self.offices, key=lambda office: office.position)

        # Determine the player controlling the rightmost office
        rightmost_office = sorted_offices[-1]
        rightmost_player = rightmost_office.player

        # Count the number of players controlling offices in the city
        player_counts = {office.player: sorted_offices.count(office) for office in sorted_offices}

        # Find the maximum number of offices controlled by a single player
        max_offices = max(player_counts.values())

        # Check if there's a tie for the number of offices controlled
        if list(player_counts.values()).count(max_offices) == 1:
            # If there's no tie, return the player controlling the rightmost office
            return rightmost_player
        else:
            # If there's a tie, return the rightmost player among those controlling the same number of offices
            tied_players = [player for player, offices in player_counts.items() if offices == max_offices]
            return max(tied_players, key=lambda player: sorted_offices[::-1].index(next(office for office in sorted_offices[::-1] if office.player == player)))

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

        # Calculate the midpoints of each city
        city1_midpoint = (city1.pos[0] + city1.width / 2, city1.pos[1] + city1.height / 2)
        city2_midpoint = (city2.pos[0] + city2.width / 2, city2.pos[1] + city2.height / 2)

        # Print city information for debugging
        print(f"City 1: {city1.name}, Width: {city1.width}, Height: {city1.height}, Midpoint: {city1_midpoint}")
        print(f"City 2: {city2.name}, Width: {city2.width}, Height: {city2.height}, Midpoint: {city2_midpoint}")

        for i in range(1, self.num_posts + 1):
            # Interpolate between the midpoints to determine post positions
            pos = (
                city1_midpoint[0] + i / (self.num_posts + 1) * (city2_midpoint[0] - city1_midpoint[0]),
                city1_midpoint[1] + i / (self.num_posts + 1) * (city2_midpoint[1] - city1_midpoint[1])
            )
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

DISTANCE_BETWEEN = 0
BUFFER = 10  # Buffer around the rectangle
SPACING = 10  # Increased spacing between the shapes

def set_city_width():
    for city in cities:
        num_offices = len(city.offices)
        
        # Calculate the number of offices for each type (circle and square)
        num_circle_offices = sum(1 for office in city.offices if office.office_type == "circle")
        num_square_offices = num_offices - num_circle_offices
        
        # Calculate the rectangle dimensions based on the actual number of offices
        RECT_WIDTH = (
            num_circle_offices * (CIRCLE_RADIUS * 2) +  # Total width of circle offices
            num_square_offices * SQUARE_SIZE +  # Total width of square offices
            2 * BUFFER +  # Buffer at the beginning and end
            SPACING*(num_offices - 1)
        )
        RECT_HEIGHT = max(CIRCLE_RADIUS * 2, SQUARE_SIZE) + BUFFER * 2
        city.update_city_size(RECT_WIDTH, RECT_HEIGHT)

# Define cities
GRONINGEN = City('Groningen', (115, 210), BLACKISH_BROWN)
GRONINGEN.add_office(Office("square", "grey", True))  # Square office that awards 1 point if claimed
GRONINGEN.add_office(Office("circle", "orange", False)) 
cities.append(GRONINGEN)

EMDEN = City('Emden', (476, 207), BLACKISH_BROWN)
EMDEN.add_office(Office("circle", "grey", False))
EMDEN.add_office(Office("square", "pink", False))
cities.append(EMDEN)

OSNABRUCK = City('Osnabruck', (434, 537), BLACKISH_BROWN)
OSNABRUCK.add_office(Office("square", "grey", False))
OSNABRUCK.add_office(Office("square", "orange", False))
OSNABRUCK.add_office(Office("square", "black", False))
cities.append(OSNABRUCK)

KAMPEN = City('Kampen', (107, 410), BLACKISH_BROWN)
KAMPEN.add_office(Office("square", "orange", False))
KAMPEN.add_office(Office("square", "black", False))
cities.append(KAMPEN)

ARNHEIM = City('Arnheim', (88, 655), BLACKISH_BROWN)
ARNHEIM.add_office(Office("square", "grey", False))
ARNHEIM.add_office(Office("circle", "grey", False))
ARNHEIM.add_office(Office("square", "orange", False))
ARNHEIM.add_office(Office("square", "pink", False))
cities.append(ARNHEIM)

DUISBURG = City('Duisburg', (179, 911), BLACKISH_BROWN)
DUISBURG.add_office(Office("square", "grey", False))
cities.append(DUISBURG)

DORTMUND = City('Dortmund', (432, 894), BLACKISH_BROWN)
DORTMUND.add_office(Office("circle", "grey", False))
DORTMUND.add_office(Office("square", "orange", False))
cities.append(DORTMUND)

MUNSTER = City('Munster', (474, 702), BLACKISH_BROWN)
MUNSTER.add_office(Office("circle", "grey", False))
MUNSTER.add_office(Office("square", "orange", False))
cities.append(MUNSTER)

COELLEN = City('Coellen', (105, 1062), BLACKISH_BROWN)
COELLEN.add_office(Office("square", "grey", True))
COELLEN.add_office(Office("square", "pink", False))
cities.append(COELLEN)

WARBURG = City('Warburg', (672, 1117), BLACKISH_BROWN)
WARBURG.add_office(Office("square", "orange", True))
WARBURG.add_office(Office("square", "pink", False))
cities.append(WARBURG)

PADERBORN = City('Paderborn', (786, 925), BLACKISH_BROWN)
PADERBORN.add_office(Office("square", "grey", False))
PADERBORN.add_office(Office("circle", "black", False))
cities.append(PADERBORN)

MINDEN = City('Minden', (757, 621), BLACKISH_BROWN)
MINDEN.add_office(Office("square", "grey", False))
MINDEN.add_office(Office("square", "orange", False))
MINDEN.add_office(Office("square", "pink", False))
MINDEN.add_office(Office("square", "black", False))
cities.append(MINDEN)

BREMEN = City('Bremen', (865, 283), BLACKISH_BROWN)
BREMEN.add_office(Office("square", "pink", False))
cities.append(BREMEN)

STADE = City('Stade', (934, 138), BLACKISH_BROWN)
STADE.add_office(Office("circle", "grey", True))
cities.append(STADE)

HANNOVER = City('Hannover', (1049, 499), BLACKISH_BROWN)
HANNOVER.add_office(Office("square", "grey", False))
HANNOVER.add_office(Office("square", "pink", False))
cities.append(HANNOVER)

HILDESHEIM = City('Hildesheim', (1089, 755), BLACKISH_BROWN)
HILDESHEIM.add_office(Office("square", "grey", False))
HILDESHEIM.add_office(Office("square", "black", False))
cities.append(HILDESHEIM)

GOTTINGEN = City('Gottingen', (1014, 1075), BLACKISH_BROWN)
GOTTINGEN.add_office(Office("square", "grey", False))
GOTTINGEN.add_office(Office("square", "orange", False))
cities.append(GOTTINGEN)

QUEDLINBURG = City('Quedlinburg', (1405, 983), BLACKISH_BROWN)
QUEDLINBURG.add_office(Office("circle", "orange", False))
QUEDLINBURG.add_office(Office("circle", "pink", False))
cities.append(QUEDLINBURG)

GOSLAR = City('Goslar', (1387, 839), BLACKISH_BROWN)
GOSLAR.add_office(Office("square", "grey", True))
cities.append(GOSLAR)

BRUNSWICK = City('Brunswick', (1253, 582), BLACKISH_BROWN)
BRUNSWICK.add_office(Office("square", "orange", True))
cities.append(BRUNSWICK)

LUNEBURG = City('Luneburg', (1359, 358), BLACKISH_BROWN)
LUNEBURG.add_office(Office("circle", "grey", True))
cities.append(LUNEBURG)

HAMBURG = City('Hamburg', (1274, 105), BLACKISH_BROWN)
HAMBURG.add_office(Office("square", "grey", False))
HAMBURG.add_office(Office("square", "orange", False))
HAMBURG.add_office(Office("square", "black", False))
cities.append(HAMBURG)

LUBECK = City('Lubeck', (1540, 122), BLACKISH_BROWN)
LUBECK.add_office(Office("square", "grey", True))
LUBECK.add_office(Office("square", "pink", False))
cities.append(LUBECK)

PERLEBERG = City('Perleberg', (1574, 308), BLACKISH_BROWN)
PERLEBERG.add_office(Office("square", "grey", False))
PERLEBERG.add_office(Office("square", "pink", False))
PERLEBERG.add_office(Office("circle", "black", False))
cities.append(PERLEBERG)

STENDAL = City('Stendal', (1538, 633), BLACKISH_BROWN)
STENDAL.add_office(Office("square", "grey", False))
STENDAL.add_office(Office("circle", "grey", False))
STENDAL.add_office(Office("square", "orange", False))
STENDAL.add_office(Office("square", "pink", False))
cities.append(STENDAL)

MAGDEBURG = City('Magdeburg', (1598, 929), BLACKISH_BROWN)
MAGDEBURG.add_office(Office("circle", "grey", False))
MAGDEBURG.add_office(Office("square", "orange", False))
cities.append(MAGDEBURG)

HALLE = City('Halle', (1544, 1063), BLACKISH_BROWN)
HALLE.add_office(Office("square", "grey", True))
HALLE.add_office(Office("square", "orange", False))
cities.append(HALLE)
set_city_width()

# CHINA = City('CHINA', (3*WIDTH//5, HEIGHT//5), BLACKISH_BROWN)
# BRAZIL = City('BRAZIL', (WIDTH//5, 4*HEIGHT//5), PINK)
# EGYPT = City('EGYPT', (2*WIDTH//5, 4*HEIGHT//5), PINK)
# AUSTRALIA = City('AUSTRALIA', (3*WIDTH//5, 4*HEIGHT//5), PINK)
# cities += [CHINA, BRAZIL, EGYPT, AUSTRALIA]

# Define players
players = [Player(GREEN, 2), Player(BLUE, 1), Player(PURPLE, 1)]
scoreboard = Scoreboard(players)
current_player = players[0]

# Define routes
routes = [
    Route([GRONINGEN, EMDEN], 3),
    Route([HALLE, MAGDEBURG], 3),
    # Route([GRONINGEN, BRAZIL], 3),
    # Route([CHINA, AUSTRALIA], 3),
    # Route([BRAZIL, EGYPT], 3),
    # Route([EGYPT, AUSTRALIA], 3),
]


def redraw_window():
    win.fill(TAN)
    # Draw scoreboard
    scoreboard.draw(win)

    # Draw cities and their offices
    for city in cities:
        # Calculate the position of the rectangle
        rect_x = city.pos[0]
        rect_y = city.pos[1]

        # Draw the city itself with the calculated RECT_WIDTH and RECT_HEIGHT
        pygame.draw.rect(win, city.color, (rect_x, rect_y, city.width, city.height))
        text = font.render(city.name, True, BLACK)
        text_width = text.get_width()
        text_x = city.pos[0] + (city.width - text_width) // 2
        text_y = city.pos[1] + city.height  # Place text just below the city rectangle
        win.blit(text, (text_x, text_y))

        start_x = rect_x + BUFFER  # Starting x-coordinate within the rectangle
        start_y = rect_y + city.height // 2 - SQUARE_SIZE // 2  # Centered vertically in the rectangle

        for i, office in enumerate(city.offices):
            if i > 0:
                if office.office_type == "square":
                    # Calculate the x-coordinate of the square office
                    square_x = start_x
                    square_y = rect_y + city.height // 2 - SQUARE_SIZE // 2
                    pygame.draw.rect(win, office.color, (square_x, square_y, SQUARE_SIZE, SQUARE_SIZE))
                    # Update the starting x-coordinate for the next office (square)
                    start_x += SQUARE_SIZE + SPACING
                else:
                    # Calculate the x-coordinate of the circle office
                    circle_x = start_x + CIRCLE_RADIUS
                    circle_y = start_y + SQUARE_SIZE // 2
                    pygame.draw.circle(win, office.color, (circle_x, circle_y), CIRCLE_RADIUS)
                    # Update the starting x-coordinate for the next office (circle)
                    start_x += CIRCLE_RADIUS * 2 + SPACING
            else:
                if office.office_type == "square":
                    # Calculate the x-coordinate of the square office
                    square_x = start_x
                    square_y = rect_y + city.height // 2 - SQUARE_SIZE // 2
                    pygame.draw.rect(win, office.color, (square_x, square_y, SQUARE_SIZE, SQUARE_SIZE))
                    # Update the starting x-coordinate for the next office (square)
                    start_x += SQUARE_SIZE + SPACING
                else:
                    # Calculate the x-coordinate of the circle office
                    circle_x = start_x + CIRCLE_RADIUS
                    circle_y = start_y + SQUARE_SIZE // 2
                    pygame.draw.circle(win, office.color, (circle_x, circle_y), CIRCLE_RADIUS)
                    # Update the starting x-coordinate for the next office (circle)
                    start_x += CIRCLE_RADIUS * 2 + SPACING
    # Draw routes
    for route in routes:
        # pygame.draw.line(win, WHITE, route.cities[0].pos, route.cities[1].pos, 6)
        for post in route.posts:
            pygame.draw.circle(win, post.color, post.pos, POST_RADIUS)

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

    for city in [GRONINGEN, EMDEN, CHINA, BRAZIL, EGYPT, AUSTRALIA]:  # iterate through cities
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
