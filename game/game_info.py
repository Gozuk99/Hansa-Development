# game_attributes.py
import random
from map_data.map1 import Map1
from map_data.map2 import Map2
from map_data.map3 import Map3
from map_data.constants import COLOR_NAMES, WHITE, GREEN, BLUE, PURPLE, RED, YELLOW
from player_info.player_attributes import Player, DisplacedPlayer, PlayerBoard, UPGRADE_MAX_VALUES

class Game:
    def __init__(self, map_num, num_players):
        self.map_num = map_num
        self.selected_map = self.assign_map(map_num, num_players)
        self.num_players = num_players

        self.players = self.create_players(num_players)
        self.current_player_index = 0
        self.current_player = self.players[self.current_player_index]
        self.active_player = self.current_player_index
        self.replace_bonus_marker = 0

        self.displaced_player = DisplacedPlayer()
        self.waiting_for_displaced_player = False
        self.waiting_for_bm_swap_office = False
        self.waiting_for_bm_upgrade_ability = False
        self.waiting_for_bm_move_any_2 = False
        self.waiting_for_bm_move3 = False

        self.waiting_for_bm_green_city = False
        self.waiting_for_place2_in_scotland_or_wales = False
        self.original_route_of_displacement = None
        self.all_empty_posts = []
        self.tile_pool = []
        self.initialize_tile_pool()
        print(f"Tile Pool: {self.tile_pool}")
        self.tile_rects = []

        self.current_full_cities_count = 0

        self.east_west_completed_count = 0
        self.players_who_completed_east_west = set()  # Track players who have completed the connection

        self.cardiff_priv = None
        self.carlisle_priv = None
        self.london_priv = None

        self.DisplaceAnywhereOwner = None
        self.OneActionOwner = None
        self.OneIncomeIfOthersIncomeOwner = None
        self.OneDisplacedPieceOwner = None
        self.FourPtsPerOwnedCityOwner = None
        self.SevenPtsPerCompletedAbilityOwner = None

    def create_players(self, num_players):
        colors = [GREEN, BLUE, PURPLE, RED, YELLOW]
        players = []
        
        for i, color in enumerate(colors[:num_players]):
            new_player = Player(color, i+1)
            new_player.board = PlayerBoard(self.selected_map.map_width, i * 220, new_player)  # Create and assign the board directly here
            new_player.actions_remaining = new_player.actions  # Initialize actions_remaining for the player

            if self.map_num == 1:
                self.selected_map.assign_mission_cards(new_player)  # Assign a mission card to the player

            players.append(new_player)
        
        return players
    
    def initialize_tile_pool(self):
        # 6 default tiles
        tiles = ["DisplaceAnywhere", "+1Action", "+1IncomeIfOthersIncome", "+1DisplacedPiece", "+4PtsPerOwnedCity", "+7PtsPerCompletedAbility"]

        num_tiles = self.num_players

        for i in range(num_tiles):
            tile = tiles.pop(random.randint(0, len(tiles)-1))
            self.tile_pool.append(tile)

    def assign_map(self, map_num, num_players):
        # Logic to assign a map based on map_num
        if map_num == 1:
            return Map1(num_players)
        elif map_num == 2:
            return Map2()
        elif map_num == 3:
            return Map3(num_players)
    
    def switch_player_if_needed(self):
        # print(f"Attempting to switch player. Replace Bonus Marker: {self.replace_bonus_marker}, Actions Remaining: {self.current_player.actions_remaining}")
        if self.replace_bonus_marker == 0 and self.current_player.actions_remaining == 0:
            print(f"Conditions met. Switching from Player {self.current_player_index+1} - {COLOR_NAMES[self.current_player.color]}.")
            self.current_player.ending_turn = False
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            self.current_player = self.players[self.current_player_index]
            self.current_player.actions_remaining = self.current_player.actions
            if self.OneActionOwner == self.current_player:
                self.current_player.actions_remaining += 1

            self.active_player = self.current_player_index

            if self.cardiff_priv or self.carlisle_priv or self.london_priv:
                self.current_player.refresh_map3_priv_actions(self)

            print(f"Switched to Player {self.current_player_index+1} - {COLOR_NAMES[self.current_player.color]}.")
        elif self.replace_bonus_marker > 0 and self.current_player.actions_remaining == 0:
            print(f"{COLOR_NAMES[self.current_player.color]} - Place a Bonus Marker to Finish your Turn.")
    
    def reset_valid_posts(self):
        for post in self.all_empty_posts:
            post.reset_post()
        self.all_empty_posts.clear()

    def check_brown_blue_priv(self, route):
        if route.region is not None:
            # Check for Wales region
            if route.region == "Wales":
                if not (self.cardiff_priv == self.current_player or self.london_priv == self.current_player):
                    print("Cannot claim post in BROWN - Incorrect Privilege")
                    return False
                if self.current_player.brown_priv_count == 0:
                    print("Used all privilege already in Brown!")
                    return False
                else:
                    self.current_player.brown_priv_count -= 1

            # Check for Scotland region
            elif route.region == "Scotland":
                if not (self.carlisle_priv == self.current_player or self.london_priv == self.current_player):
                    print("Cannot claim post in BLUE - Incorrect Privilege")
                    return False
                if self.current_player.blue_priv_count == 0:
                    print("Used all privilege already in Blue!")
                    return False
                else:
                    self.current_player.blue_priv_count -= 1
        return True

    def check_for_east_west_connection(self):
        if self.current_player in self.players_who_completed_east_west:
            print(f"{COLOR_NAMES[self.current_player.color]} has already completed the East-West Connection.")
            return
        
        if not self.check_if_player_has_matching_offices_in_east_west(self.selected_map.east_west_cities[0], self.selected_map.east_west_cities[1]):
            return

        if self.has_east_west_connection(self.selected_map.east_west_cities[0], self.selected_map.east_west_cities[1]):
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
        
    def get_bonus_marker_points(self, total_bms):
        if total_bms == 1:
            return 1
        elif 2 <= total_bms <= 3:
            return 3
        elif 4 <= total_bms <= 5:
            return 6
        elif 6 <= total_bms <= 7:
            return 10
        elif 8 <= total_bms <= 9:
            return 15
        elif total_bms >= 10:
            return 21
        else:
            return 0
        
        #1 initial points
        #2 fully developed abilities
        #3 prestige points for total bonus markers collected
        #  1-1, 2or3-3, 4or5-6, 6or7-10, 8or9-15, 10+ - 21
        #4 specialprestigepoints 7/8/9/11
        #5 prestige points for cities, 2 per control
        #6 largest network x key       
    
    def dfs_network_size(self, player, city, visited_cities):
        if city in visited_cities:
            return 0  # This city is already part of the current network

        visited_cities.add(city)
        offices_in_city = sum(1 for office in city.offices if office.controller == player)
        network_size = offices_in_city  # Count the number of offices instead of just the city

        for route in city.routes:
            for connected_city in route.cities:
                if connected_city != city and connected_city.has_office_owned_by(player) and connected_city not in visited_cities:
                    network_size += self.dfs_network_size(player, connected_city, visited_cities)

        return network_size

    def calculate_largest_network(self, player):
        largest_network = 0
        all_visited_cities = set()

        for city in self.selected_map.cities:
            if city.has_office_owned_by(player) and city not in all_visited_cities:
                visited_cities = set()
                network_size = self.dfs_network_size(player, city, visited_cities)
                largest_network = max(largest_network, network_size)
                all_visited_cities.update(visited_cities)

        return largest_network

    def finalize_end_of_game_points(self):
        for player in self.players:
            initial_points = player.score
            ability_points = 0
            bonus_marker_points = 0
            special_prestige_points = 0
            city_control_points = 0
            largest_network_points = 0

            # 2. Fully developed abilities
            for ability in ['keys', 'book', 'actions', 'bank']: # exclude keys intentionally
                if getattr(player, ability) == UPGRADE_MAX_VALUES[ability]:
                    ability_points += 4  # Assuming 4 points for each fully developed ability
                    if self.SevenPtsPerCompletedAbilityOwner == player:
                        ability_points += 3 # for a total of 7 points per fully developed ability

            # 3. Prestige points for total bonus markers collected
            total_bms = len(player.bonus_markers) + len(player.used_bonus_markers)
            bonus_marker_points += self.get_bonus_marker_points(total_bms)

            # 5. Add Special Prestige Points
            for upgrade_city in self.selected_map.upgrade_cities:
                if upgrade_city.upgrade_type == "SpecialPrestigePoints":
                    special_prestige_points += upgrade_city.get_special_prestige_points_for_player(player)

            # 6. Add points for control of cities
            for city in self.selected_map.cities:
                if city.get_controller() == player:
                    city_control_points += 2  # 2 points per city controlled
                    if self.FourPtsPerOwnedCityOwner == player:
                        city_control_points += 2  # for a total of 4 points per city controlled

            # 7. Points for the largest network
            largest_network_points += self.calculate_largest_network(player) * player.keys

            # Sum up the final score
            player.final_score = (initial_points + ability_points + bonus_marker_points + special_prestige_points + city_control_points + largest_network_points)

            # Print or store the score breakdown for display
            score_breakdown = {
                'Initial Points': initial_points,
                'Ability Points': ability_points,
                'Bonus Marker Points': bonus_marker_points,
                'Special Prestige Points': special_prestige_points,
                'City Control Points': city_control_points,
                'Largest Network Points': largest_network_points
            }

            # Ensure final score is not less than the initial score
            player.final_score = max(player.final_score, initial_points)

            # You can print the score breakdown here if needed
            print(f"Score breakdown for {COLOR_NAMES[player.color]}: {score_breakdown}")

    def check_for_game_end(self):

        self.current_full_cities_count = sum(1 for city in self.selected_map.cities if city.city_is_full())
        
        # Check if the bonus marker pool is empty or any player has reached the score threshold
        end_conditions_met = (not self.selected_map.bonus_marker_pool or 
                              any(player.score >= 20 for player in self.players) or
                              self.current_full_cities_count >= self.selected_map.max_full_cities)

        if end_conditions_met:
            # Finalize points before determining the winner
            self.finalize_end_of_game_points()

            # Find the player(s) with the highest score
            highest_score = max(player.final_score for player in self.players)
            highest_scoring_players = [player for player in self.players if player.final_score == highest_score]

            # If there's a tie, you might need additional logic to determine the winner
            return highest_scoring_players

        # Game continues if no end condition is met
        return None

    # # Example usage in your game loop:
    # winning_players = game.check_for_game_end()
    # if winning_players:
    #     # Handle end of game, which might involve displaying a message with draw_end_game
    #     draw_end_game(win, winning_players)
    #     # Maybe break out of the game loop or wait for user input to restart or exit
