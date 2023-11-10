# map.py
import pygame
import random
from map_data.constants import BLACK, CIRCLE_RADIUS, SQUARE_SIZE, BUFFER, SPACING, TAN, COLOR_NAMES, YELLOW, BLACK, WHITE, ORANGE, PINK, PRIVILEGE_COLORS
from drawing.drawing_utils import draw_shape, draw_text, draw_line

class Map:
    def __init__(self):
        # This should never change
        self.initial_bonus_types = ['Move3', 'SwapOffice', 'PlaceAdjacent']
        self.bonus_marker_pool = []
        self.place_new_bonus_marker = False
        # Prepare the starting bonus markers
        self.assign_bm_pool_default()

    def assign_starting_bonus_markers(self):
         # Ensure we shuffle the initial bonus types to randomize the assignment
        random.shuffle(self.initial_bonus_types)

        for route in self.routes:
            # Check if the route has a bonus marker
            if route.has_bonus_marker:
                if self.initial_bonus_types:  # Check if there are still bonus types available
                    bm_type = self.initial_bonus_types.pop()
                    # print(f"Assigning bonus marker: {bm_type} to route between {route.cities[0].name} and {route.cities[1].name}")
                    route.assign_bonus_marker(bm_type)
                else:
                    print(f"Ran out of initial bonus types to assign for route between {route.cities[0].name} and {route.cities[1].name}")

    def assign_bm_pool_default(self):
        # Default bonus markers
        default_bonus_markers = {
            'PlaceAdjacent': 3,
            'SwapOffice': 2,
            'Move3': 1,
            'UpgradeAbility': 2,
            '3Actions': 2,
            '4Actions': 2
        }
        # Add the default bonus markers to the pool
        for bm_type, count in default_bonus_markers.items():
            self.bonus_marker_pool.extend([bm_type] * count)
        random.shuffle(self.bonus_marker_pool)
    
    def assign_bm_pool_random(self):
        # All possible bonus markers including expansions
        all_bonus_markers = {
            'PlaceAdjacent': 3,
            'SwapOffice': 2,
            'Move3': 1,
            'UpgradeAbility': 2,
            '3Actions': 2,
            '4Actions': 2,
            'Exchange Bonus Marker': 2,
            'Tribute for Establish a Trading Post': 2,
            'Block Trade Route': 2
        }

        # Create a list of all bonus markers based on their maximum count
        all_bonus_markers_list = [bm_type for bm_type, max_count in all_bonus_markers.items() for _ in range(max_count)]

        # Shuffle the list of all possible bonus markers
        random.shuffle(all_bonus_markers_list)

        # Take exactly 12 bonus markers to form the bonus marker pool
        self.bonus_marker_pool = all_bonus_markers_list[:12]
    
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
            print(f"No offices in {self.name}, therefore no controller.")
            return None  # No offices in the city

        # Determine if the rightmost office has a controller
        rightmost_office = self.offices[-1]
        rightmost_player = rightmost_office.controller
        if rightmost_player:
            print(f"Rightmost office in {self.name} is controlled by Player {COLOR_NAMES[rightmost_player.color]}")
        else:
            print(f"Rightmost office in {self.name} has no controller.")

        # Count the number of players controlling offices in the city
        player_counts = {}
        for office in self.offices:
            if office.controller:
                if office.controller in player_counts:
                    player_counts[office.controller] += 1
                else:
                    player_counts[office.controller] = 1

        if not player_counts:
            print(f"No players control any offices in {self.name}.")
            return None  # No offices controlled by any player in the city

        for player, count in player_counts.items():
            print(f"Player {COLOR_NAMES[player.color]} controls {count} office(s) in {self.name}.")

        # Find the maximum number of offices controlled by a single player
        max_offices = max(player_counts.values())

        # Identify the player(s) with the maximum number of offices
        players_with_max_offices = [player for player, count in player_counts.items() if count == max_offices]

        # If only one player has the max, they're the controller, regardless of the rightmost office
        if len(players_with_max_offices) == 1:
            controlling_player = players_with_max_offices[0]
            self.controller = controlling_player
            print(f"Player {COLOR_NAMES[controlling_player.color]} controls the most offices in {self.name}.")
            return controlling_player

        # If there's a tie, find the rightmost player among those tied
        rightmost_tied_player = max(
            players_with_max_offices,
            key=lambda player: self.offices[::-1].index(next(office for office in self.offices[::-1] if office.controller == player))
        )
        self.controller = rightmost_tied_player
        print(f"There is a tie. Rightmost player among tied players in {self.name} is Player {COLOR_NAMES[rightmost_tied_player.color]}.")
        return rightmost_tied_player
    
    def has_empty_office(self):
        for office in self.offices:
            if office.controller == None:
                return True
        return False

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
    def get_next_open_office_color(self):
        """Return the color of the next open office in the city. If all offices are claimed, return None."""
        for office in self.offices:
            if office.is_open():
                return office.color
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
        draw_shape(window, "rectangle", YELLOW, self.x_pos, self.y_pos, width=self.width, height=self.height)

        # If upgrade type is 'SpecialPrestigePoints', handle it differently
        if self.upgrade_type == "SpecialPrestigePoints":
            self.draw_special_prestige_points(window)
            return

        x_font_centered = self.x_pos + (self.width / 2)
        y_font_centered = self.y_pos + (self.height / 2)
        font = pygame.font.SysFont(None, 28)
        draw_text(window, self.upgrade_type, x_font_centered, y_font_centered, font, color=BLACK, centered=True)

class SpecialPrestigePoints:
    def __init__(self, city_name, x_pos, y_pos, width, height):
        self.city_name = city_name
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width
        self.height = height

        self.circle_data = [
            {"color": WHITE, "value": 7, "owner": None},
            {"color": ORANGE, "value": 8, "owner": None},
            {"color": PINK, "value": 9, "owner": None},
            {"color": BLACK, "value": 11, "owner": None}
        ]
    def claim_highest_prestige(self, player):
        # Log player details and privileges
        print(f"Player's Color: {player.color}, Player's Privilege: {player.privilege}")

        # Player's privilege color
        player_privilege_color = player.privilege  # This should directly return one of the values in PRIVILEGE_COLORS
            
        # Check if the player's privilege allows claiming the circle
        try:
            player_privilege_index = PRIVILEGE_COLORS.index(player_privilege_color)
        except ValueError:
            print(f"Player's privilege color {player_privilege_color} not found in PRIVILEGE_COLORS.")
            return

        # Check each circle in decreasing order of value
        for circle in sorted(self.circle_data, key=lambda x: x["value"], reverse=True):
            # Log circle details
            print(f"Checking Circle: {circle}")

            # Map RGB to its string name
            circle_color_name = COLOR_NAMES.get(circle["color"])
            if circle_color_name not in PRIVILEGE_COLORS:
                print(f"Circle color {circle_color_name} not in PRIVILEGE_COLORS. Skipping...")
                continue  # Skip circles with colors not in PRIVILEGE_COLORS

            # Circle's privilege index
            circle_privilege_index = PRIVILEGE_COLORS.index(circle_color_name)
                
            if circle["owner"] is None and player_privilege_index >= circle_privilege_index:
                circle["owner"] = player
                # Change the circle color to the player's RGB to indicate ownership
                circle["color"] = player.color
                print(f"Circle claimed by player. Circle color changed to {player.color}.")
                break
        
    def draw_special_prestige_points(self, window):
        draw_shape(window, "rectangle", YELLOW, self.x_pos, self.y_pos, width=self.width, height=self.height)

        # Define the total width of all circles and spaces combined
        total_width = (CIRCLE_RADIUS * 2) * 4 + (SPACING * 3)

        # Define starting position for the circles
        start_x = self.x_pos + (self.width - total_width) / 2 + CIRCLE_RADIUS  # Adjust the starting position
        start_y = self.y_pos + self.height / 2  # This centers the circle vertically in the rectangle

        for circle in self.circle_data:
            # Draw circle with the circle's color (either a privilege color or a player's color)
            pygame.draw.circle(window, circle["color"], (int(start_x), int(start_y)), CIRCLE_RADIUS)

            # Render text
            font = pygame.font.SysFont(None, 36)  # Use default font, size 36
            text_surface = font.render(str(circle["value"]), True, WHITE if circle["color"] == BLACK else BLACK)
            text_rect = text_surface.get_rect(center=(start_x, start_y))
            window.blit(text_surface, text_rect)

            # Adjust start_x for next circle
            start_x += CIRCLE_RADIUS * 2 + SPACING

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
    def __init__(self, cities, num_posts, has_bonus_marker=False, bonus_marker_type=None):
        self.cities = cities
        for city in cities:
            city.add_route(self)
        self.num_posts = num_posts
        self.has_bonus_marker = has_bonus_marker
        # self.bonus_marker = self.assign_bonus_marker(bonus_marker_type) if has_bonus_marker else None
        self.bonus_marker = None  # Don't assign it yet
        self.posts = self.create_posts()

    def create_posts(self, buffer=0.1):
        city1, city2 = self.cities
        posts = []

        for i in range(1, self.num_posts + 1):
            # Adjust the buffer to control the post distribution
            t = i / (self.num_posts + 1)
            t = buffer + (1 - 2 * buffer) * t

            pos = (
                city1.midpoint[0] + t * (city2.midpoint[0] - city1.midpoint[0]),
                city1.midpoint[1] + t * (city2.midpoint[1] - city1.midpoint[1])
            )
            
            posts.append(Post(pos))

        return posts

    def find_empty_post(self):
        for post in self.posts:
            if post.owner == None:
                return post
        return None
    
    def has_tradesmen(self):
        for post in self.posts:
            if post.owner is not None:
                # print(f"Route check: Tradesman found at post {self.posts.index(post)+1}/{self.num_posts} on the route between {self.cities[0].name} and {self.cities[1].name}.")
                return True
        print(f"Route check: No tradesmen found on the route between {self.cities[0].name} and {self.cities[1].name}.")
        return False

    def has_empty_office_in_cities(self):
        city1, city2 = self.cities
        if city1.has_empty_office() or city2.has_empty_office():
            print(f"Route check: At least one empty office found in cities {city1.name} or {city2.name}.")
            return True
        print(f"Route check: No empty offices found in cities {city1.name} and {city2.name}.")
        return False
    
    def is_controlled_by(self, player):
        return all(post.owner == player for post in self.posts)

    def is_complete(self):
        for post in self.posts:
            if post.color == BLACK:
                return False
        return True
    
    def assign_bonus_marker(self, bm_type):
        if not self.bonus_marker:  # Only assign if there's no bonus marker already
            # print(f"Route between {self.cities[0].name} and {self.cities[1].name} is being assigned a bonus marker of type {bm_type}")
            self.bonus_marker = BonusMarker(bm_type)
        else:
            print(f"Route between {self.cities[0].name} and {self.cities[1].name} already has a bonus marker assigned")

class BonusMarker:
    def __init__(self, type):
        self.type = type

    def draw_board_bonus_markers(self, screen, position):
        # Draw the bonus marker as a simple shape (e.g., a circle)
        pygame.draw.circle(screen, BLACK, position, 30)
        # Draw the text for the bonus marker type
        font = pygame.font.SysFont(None, 24)
        text = font.render(self.type, True, WHITE)  # Render the text with the bonus marker's type
        text_rect = text.get_rect(center=position)  # Get a rect object to center the text inside the circle
        screen.blit(text, text_rect)  # Draw the text to the screen at the specified position

class Post:
    def __init__(self, position, owner=None, required_shape=None):
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

    def can_be_claimed_by(self, shape):
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
        