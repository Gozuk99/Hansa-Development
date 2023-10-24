# map.py
import pygame
from map_data.constants import BLACK, CIRCLE_RADIUS, SQUARE_SIZE, BUFFER, SPACING

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

    def add_route(self, route):
        self.routes.append(route)

    def change_color(self, new_color):
        self.color = new_color
    
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
        num_circle_offices = sum(1 for office in self.offices if office.office_type == "circle")
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
class Office:
    def __init__(self, office_type, color, awards_points):
        self.office_type = office_type  # "circle" or "square"
        self.color = color
        self.awards_points = awards_points
        self.controller = None  # Initialize controller as None
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
    
class Post:
    def __init__(self, pos):
        self.pos = pos
        self.color = BLACK