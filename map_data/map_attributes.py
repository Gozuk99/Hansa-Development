# map.py
import pygame
from map_data.constants import BLACK, CIRCLE_RADIUS, SQUARE_SIZE, BUFFER, SPACING, TAN, COLOR_NAMES, YELLOW, BLACK, WHITE
from drawing.drawing_utils import draw_shape, draw_text

class Map:
    def __init__(self):
        self.placeholder = 0

class City:
    def __init__(self, name, position, color):
        self.name = name
        self.pos = position
        self.color = color
        self.routes = []
        self.controller = None  # Player controlling the city
        self.offices = []  # List of offices within the city
        self.width = 0
        self.height = 0
        self.midpoint = (0, 0)  # Initialize midpoint with (0, 0)
        self.upgrade_city_type = None

    def add_route(self, route):
        self.routes.append(route)

    def change_color(self, new_color):
        self.color = new_color

    def assign_upgrade_type(self, upgrade_type):
        self.upgrade_city_type =upgrade_type

    def add_office(self, office):
        self.offices.append(office)

    def update_office_ownership(self, player, color):
        for office in self.offices:
            if office.controller is None:
                office.controller = player
                office.color = color
                break

    def update_city_size_based_on_offices(self):
        num_offices = len(self.offices)
        num_circle_offices = sum(1 for office in self.offices if office.shape == "circle")
        num_square_offices = num_offices - num_circle_offices
        
        # Calculate the rectangle dimensions based on the actual number of offices
        rect_width = (
            num_circle_offices * (CIRCLE_RADIUS * 2) +  # Total width of circle offices
            num_square_offices * SQUARE_SIZE +  # Total width of square offices
            2 * BUFFER +  # Buffer at the beginning and end
            SPACING*(num_offices - 1)
        )
        rect_height = max(CIRCLE_RADIUS * 2, SQUARE_SIZE) + BUFFER * 2
        
        self.width = rect_width
        self.height = rect_height
        self.midpoint = (self.pos[0] + rect_width / 2, self.pos[1] + rect_height / 2)

    def get_controller(self):
        if not self.offices:
            return None  # No offices in the city

        # Determine the player controlling the rightmost office
        rightmost_office = self.offices[-1]
        rightmost_player = rightmost_office.controller

        # Count the number of players controlling offices in the city
        player_counts = {}
        for office in self.offices:
            if office.controller is not None:
                if office.controller in player_counts:
                    player_counts[office.controller] += 1
                else:
                    player_counts[office.controller] = 1

        if not player_counts:
            return None  # No offices controlled by any player in the city

        # Find the maximum number of offices controlled by a single player
        max_offices = max(player_counts.values())

        # Check if there's a tie for the number of offices controlled
        if list(player_counts.values()).count(max_offices) == 1:
            # If there's no tie, return the player controlling the rightmost office
            self.controller = rightmost_player
            return rightmost_player
        else:
            # If there's a tie, return the rightmost player among those controlling the same number of offices
            tied_players = [player for player, offices in player_counts.items() if offices == max_offices]
            self.controller = max(tied_players, key=lambda player: self.offices[::-1].index(next(office for office in self.offices[::-1] if office.controller == player)))
            return max(tied_players, key=lambda player: self.offices[::-1].index(next(office for office in self.offices[::-1] if office.controller == player)))
    
    def has_required_piece_shape(self, player, route, city):
        """Returns True if the player has the required piece shape on the route to claim an office in the city."""
        required_shape = city.get_next_open_office_shape()

        return any(post.owner_piece_shape == required_shape and post.owner == player for post in route.posts)
    
    def get_next_open_office_shape(self):
        """Return the shape of the next open office in the city. If all offices are claimed, return None."""
        for office in self.offices:
            if office.is_open():
                return office.shape
        return None

class Upgrade:
    def __init__(self, city_name, upgrade_type, x_pos, y_pos, width, height):
        self.city_name = city_name
        self.upgrade_type = upgrade_type
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width
        self.height = height

    def draw_upgrades_on_map(self, window):
        font = pygame.font.SysFont(None, 28)
        draw_shape(window, "rectangle", YELLOW, self.x_pos, self.y_pos, width=self.width, height=self.height)

        x_font_centered = self.x_pos + (self.width / 2)
        y_font_centered = self.y_pos + (self.height / 2)
        draw_text(window, self.upgrade_type, x_font_centered, y_font_centered, font, color=BLACK, centered=True)

class Office:
    def __init__(self, shape, color, awards_points):
        self.shape = shape  # "circle" or "square"
        self.color = color
        self.awards_points = awards_points
        self.controller = None  # Initialize controller as None
    def is_open(self):
        """Return True if the office is unclaimed."""
        return self.controller is None

class Route:
    def __init__(self, cities, num_posts, has_bonus_marker=False):
        self.cities = cities
        for city in cities:
            city.add_route(self)
        self.num_posts = num_posts
        self.has_bonus_marker = has_bonus_marker
        self.posts = self.create_posts()

    def create_posts(self, buffer=0.1):
        city1, city2 = self.cities
        posts = []

        # Print cities' midpoints to ensure they are correct
        # print(f"City1 Midpoint: {city1.midpoint}")
        # print(f"City2 Midpoint: {city2.midpoint}")

        for i in range(1, self.num_posts + 1):
            # Adjust the buffer to control the post distribution
            t = i / (self.num_posts + 1)
            t = buffer + (1 - 2 * buffer) * t

            # Print the 't' value to ensure it's computed correctly
            # print(f"Interpolation Factor t for post {i}: {t}")

            pos = (
                city1.midpoint[0] + t * (city2.midpoint[0] - city1.midpoint[0]),
                city1.midpoint[1] + t * (city2.midpoint[1] - city1.midpoint[1])
            )
            
            # Print the computed position for each post
            # print(f"Computed Position for Post {i}: {pos}")

            posts.append(Post(pos))

        return posts

    def find_empty_post(self):
        for post in self.posts:
            if post.owner == None:
                return post
        return None
    
    def is_controlled_by(self, player):
        return all(post.owner == player for post in self.posts)

    def is_complete(self):
        for post in self.posts:
            if post.color == BLACK:
                return False
        return True
    
class Post:
    def __init__(self, position, owner=None, circle_color=BLACK, square_color=BLACK, required_shape=None):
        self.pos = position
        self.owner = owner  # This represents the player who owns the post.
        self.owner_piece_shape = None  # This represents the player who owns the post.
        self.circle_color = TAN
        self.square_color = TAN
        self.required_shape = required_shape  # can be "circle", "square", or None if no specific requirement

    def reset_post(self):
        self.circle_color = TAN
        self.square_color = TAN
        self.owner = None
        self.owner_piece_shape = None
    
    def valid_post_to_displace_to(self):
        self.circle_color = WHITE
        self.square_color = WHITE

    def is_owned(self):
        return self.owner is not None

    def can_be_claimed_by(self, player, shape):
        return self.owner is None and (self.required_shape is None or self.required_shape == shape)

    def claim(self, player, shape):
        if shape == "circle":
            self.circle_color = player.color
            self.square_color = TAN
        else:
            self.square_color = player.color
            self.circle_color = TAN
        self.owner = player
        self.owner_piece_shape = shape
    
    def DEBUG_print_post_details(self):
        print(f"Post Details!!!")
        print(f"Required Shape {self.required_shape}")
        print(f"Post {self.pos}")
        print(f"Owner Piece Shape {self.owner_piece_shape}")
        print(f"Owner {COLOR_NAMES[self.owner.color]}")
        print(f"Circle Color {COLOR_NAMES[self.circle_color]}")
        print(f"Square Color {COLOR_NAMES[self.square_color]}")
        