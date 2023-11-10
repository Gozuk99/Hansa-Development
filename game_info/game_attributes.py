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
        self.waiting_for_bm_upgrade_choice = False
        self.waiting_for_bm_move3_choice = False
        self.all_empty_posts = []

    def create_players(self, num_players):
        # Create player objects based on the number of players
        # colors = [GREEN, BLUE, PURPLE, RED, YELLOW]  # Assume these are defined somewhere
        colors = [GREEN, BLUE, PURPLE]  # Assume these are defined somewhere
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