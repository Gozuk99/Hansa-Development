# player_attributes.py

import pygame
import sys

from map_data.map1 import Map1
from map_data.constants import COLOR_NAMES, WHITE, GREEN, BLUE, PURPLE, RED, YELLOW, SQUARE_SIZE, CIRCLE_RADIUS, WHITE, ORANGE, PINK, BLACK, CITY_KEYS_MAX_VALUES, ACTIONS_MAX_VALUES, PRIVILEGE_COLORS, BOOK_OF_KNOWLEDGE_MAX_VALUES, BANK_MAX_VALUES
from drawing.drawing_utils import draw_shape, draw_text
from player_info.player_attributes import Player, DisplacedPlayer

class Game:
    def __init__(self, map_num, num_players):
        self.selected_map = self.assign_map(map_num)
        self.num_players = num_players

        self.players = self.create_players(num_players)
        self.current_player_index = 0
        self.current_player = self.players[self.current_player_index]
        self.replace_bonus_marker = 0

        self.displaced_player = DisplacedPlayer()
        self.waiting_for_displaced_player = False
        self.original_route_of_displacement = None
        self.waiting_for_bm_swap_office = False
        self.waiting_for_bm_upgrade_choice = False
        self.waiting_for_bm_move3_choice = False
        self.all_empty_posts = []

        self.east_west_completed_count = 0
        self.players_who_completed_east_west = set()  # Track players who have completed the connection


    def create_players(self, num_players):
        # Create player objects based on the number of players
        colors = [GREEN, BLUE, PURPLE, RED, YELLOW]  # Assume these are defined somewhere
        # colors = [GREEN, BLUE, PURPLE]  # Assume these are defined somewhere
        players = [Player(color, i+1) for i, color in enumerate(colors[:num_players])]
        for player in players:
            player.actions_remaining = player.actions  # Initialize actions_remaining for each player
        return players

    def assign_map(self, map_num):
        # Logic to assign a map based on map_num
        if map_num == 1:
            return Map1()
        # Add additional maps as needed
        # ...
    
    def claim_bonus_marker(self):
        # This method would be called when a player claims a new bonus marker
        self.bonus_markers_to_place += 1
        print(f"Player has claimed a bonus marker. {self.bonus_markers_to_place} bonus marker(s) to place.")
    
    def place_bonus_marker(self):
        # This method would be called when a bonus marker is successfully placed on the map
        if self.bonus_markers_to_place > 0:
            self.bonus_markers_to_place -= 1
            print(f"Bonus marker placed. {self.bonus_markers_to_place} bonus marker(s) left to place.")
        else:
            print("No bonus markers to place.")

    def switch_player_if_needed(self):
        # print(f"Attempting to switch player. Replace Bonus Marker: {self.replace_bonus_marker}, Actions Remaining: {self.current_player.actions_remaining}")
        if self.replace_bonus_marker == 0 and self.current_player.actions_remaining == 0:
            print(f"Conditions met. Switching from Player {self.current_player_index+1} - {COLOR_NAMES[self.current_player.color]}.")
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            self.current_player = self.players[self.current_player_index]
            self.current_player.actions_remaining = self.current_player.actions
            print(f"Switched to Player {self.current_player_index+1} - {COLOR_NAMES[self.current_player.color]}.")
        elif self.replace_bonus_marker > 0 and self.current_player.actions_remaining == 0:
            print(f"{COLOR_NAMES[self.current_player.color]} - Place a Bonus Marker to Finish your Turn.")
    
    def reset_valid_posts(self):
        for post in self.all_empty_posts:
            post.reset_post()
        self.all_empty_posts.clear()

    def check_for_east_west_connection(self):
        if self.current_player in self.players_who_completed_east_west:
            print(f"{COLOR_NAMES[self.current_player.color]} has already completed the East-West Connection.")
            return
        
        if not self.check_if_player_has_matching_offices_in_east_west('Stendal', 'Arnheim'):
            return

        if self.has_east_west_connection('Stendal', 'Arnheim'):
            # Points for the 1st, 2nd, and 3rd completions
            east_west_points = [7, 4, 2]

            # Check if the connection has been completed less than 3 times
            if self.east_west_completed_count < len(east_west_points):
                awarded_points = east_west_points[self.east_west_completed_count]
                self.current_player.score += awarded_points
                self.east_west_completed_count += 1
                self.players_who_completed_east_west.add(self.current_player)

                print(f"{COLOR_NAMES[self.current_player.color]} has created an East-West Connection and is awarded {awarded_points} points! Total score is now {self.current_player.score}.")
            else:
                print(f"East West Connection has been completed 3+ times. No points awarded to {COLOR_NAMES[self.current_player.color]}.")
    
    def check_if_player_has_matching_offices_in_east_west(self, start_city_name, end_city_name):
        start_city = next((city for city in self.selected_map.cities if city.name == start_city_name), None)
        end_city = next((city for city in self.selected_map.cities if city.name == end_city_name), None)

        if not start_city or not end_city:
            return False

        start_city_owners = {office.controller for office in start_city.offices if office.controller}
        end_city_owners = {office.controller for office in end_city.offices if office.controller}

        # Check for intersection between the sets of owners
        common_owners = start_city_owners.intersection(end_city_owners)
        return bool(common_owners)
    
    def has_east_west_connection(self, start_city_name, end_city_name, visited=None):
        # This is a recursive depth-first search (DFS) algorithm.
        if visited is None:
            visited = set()

        start_city = next((city for city in self.selected_map.cities if city.name == start_city_name), None)
        end_city = next((city for city in self.selected_map.cities if city.name == end_city_name), None)

        # Check if both cities exist in the game.
        if start_city is None or end_city is None:
            return False

        # If we've reached the end city, return True
        if start_city == end_city:
            return True

        # Mark the start city as visited
        visited.add(start_city)

        # Go through each route connected to the start city
        for route in start_city.routes:
            # Check all cities connected to this route
            for connected_city in route.cities:
                # Skip if we've already visited this city or if the connected city doesn't have the player's office
                if connected_city in visited or not connected_city.has_office_controlled_by(self.current_player):
                    continue

                # Recursively check if the connected city leads to the end city
                if self.has_east_west_connection(connected_city.name, end_city_name, visited):
                    return True

        # If none of the routes lead to the end city, return False
        return False
    
    def check_for_game_end(self):
        # Check if the bonus marker pool is empty or any player has reached the score threshold
        if not self.selected_map.bonus_marker_pool or any(player.score >= 3 for player in self.players):
            # Find the player with the highest score
            highest_scoring_players = [player for player in self.players if player.score == max(player.score for player in self.players)]
            # If there's a tie, you might need additional logic to determine the winner
            return highest_scoring_players
        # Game continues if no end condition is met
        return None