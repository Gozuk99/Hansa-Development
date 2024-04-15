import torch
import json
import time
import sys
from game.game_info import Game
from map_data.constants import GREEN, BLUE, PURPLE, RED, YELLOW, BLACKISH_BROWN, DARK_RED, DARK_GREEN, DARK_BLUE, GREY, MAX_CITIES, MAX_ROUTES, COLOR_NAMES, WHITE, ORANGE, PINK, BLACK 
from map_data.map_attributes import BonusMarker

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

class BoardData:
    def __init__(self):
        self.game_tensor_size = 0
        self.city_tensor_size = 0
        self.route_tensor_size = 0
        self.player_tensor_size = 0
        self.all_game_state_size = 0
    
    def get_game_state(self, game):
        # game_tensor Size: torch.Size([49])    // 70
        # city_tensor Size: torch.Size([1740])  // 2700
        # route_tensor Size: torch.Size([880])  // 1400
        # player_tensor Size: torch.Size([185]) // 275

        # Game:
        start_time = time.time()
        game_tensor = self.fill_game_tensor(game)
        end_time = time.time()
        # print(f"game_tensor Size: {game_tensor.size()[0]}")
        self.game_tensor_size = game_tensor.size()[0]
        execution_time1 = end_time - start_time

        # Map info:
        start_time = time.time()
        city_tensor = self.fill_city_tensor(game)
        end_time = time.time()
        # print(f"city_tensor Size: {city_tensor.size()[0]}")
        self.city_tensor_size = city_tensor.size()[0]
        execution_time2 = end_time - start_time
        
        # Route Info
        start_time = time.time()
        route_tensor = self.fill_route_tensor(game)
        end_time = time.time()
        # print(f"route_tensor Size: {route_tensor.size()[0]}")
        self.route_tensor_size = route_tensor.size()[0]
        execution_time3 = end_time - start_time
        
        #Player Info
        start_time = time.time()
        player_tensor = self.fill_player_info_tensor(game)
        end_time = time.time()
        # print(f"player_tensor Size: {player_tensor.size()[0]}")
        self.player_tensor_size = player_tensor.size()[0]
        execution_time4 = end_time - start_time
        
        # print(f"game_tensor Execution Time: {execution_time1} seconds, Size: {game_tensor.size()}")
        # print(f"city_tensor Execution Time: {execution_time2} seconds, Size: {city_tensor.size()}")
        # print(f"route_tensor Execution Time: {execution_time3} seconds, Size: {route_tensor.size()}")
        # print(f"player_tensor Execution Time: {execution_time4} seconds, Size: {player_tensor.size()}")

        flattened_game_state = torch.cat([game_tensor, city_tensor, route_tensor, player_tensor], dim=0)

        # print(f"All Game State Size: {flattened_game_state.size()[0]}")
        self.all_game_state_size = flattened_game_state.size()[0]

        return flattened_game_state

    def fill_game_tensor(self, game):
        # Initial game info
        initial_game_info = torch.tensor([game.map_num, game.num_players, 
                                          game.active_player, game.replace_bonus_marker,
                                          game.current_player_index, game.selected_map.max_full_cities, 
                                          game.current_full_cities_count, game.east_west_completed_count,
                                          game.waiting_for_bm_swap_office, game.waiting_for_bm_upgrade_ability,
                                          game.waiting_for_bm_move_any_2, game.waiting_for_bm_move3,
                                          game.waiting_for_bm_exchange_bm, game.waiting_for_bm_tribute_trading_post, game.waiting_for_bm_block_trade_route,
                                          game.waiting_for_bm_green_city, game.waiting_for_place2_in_scotland_or_wales], device=device, dtype=torch.uint8)

        # Privileges info
        cardiff_priv, carlisle_priv, london_priv = self.assign_blue_brown_priv_mapping(game)
        privileges_info = torch.tensor([cardiff_priv, carlisle_priv, london_priv], device=device, dtype=torch.uint8)
        
        # Special Prestige Points info
        special_prestige_points_info = self.assign_special_prestige_points_mapping(game)
        bonus_marker_info = self.assign_bonus_marker_pool_mapping(game)
        tile_pool_info = self.assign_tile_pool_mapping(game)
        tile_owner_info = self.assign_tile_owner_mapping(game)
        tile_to_buy_info = self.assign_tile_buying_info(game)

        # Concatenate all game info into one tensor
        game_info = torch.cat((initial_game_info, privileges_info, special_prestige_points_info, bonus_marker_info, tile_pool_info, tile_owner_info, tile_to_buy_info), dim=0).unsqueeze(0)  # Add an extra dimension for batch size
        # flatten the tensor
        game_info = game_info.flatten()

        # Pad zeroes to the end until the size of tensor is of size 70
        # This is to allow any missing data to be added without fear of breaking the model
        game_info = torch.cat((game_info, torch.zeros(70 - game_info.size()[0], device=device, dtype=torch.uint8)), dim=0)

        # print(f"game_tensor Size: {game_info.size()}")
        return game_info

    def assign_blue_brown_priv_mapping(self, game):
        colors = [GREEN, BLUE, PURPLE, RED, YELLOW]

        cardiff_priv = colors.index(game.cardiff_priv) + 1 if game.cardiff_priv else 0
        carlisle_priv = colors.index(game.carlisle_priv) + 1 if game.carlisle_priv else 0
        london_priv = colors.index(game.london_priv) + 1 if game.london_priv else 0

        return cardiff_priv, carlisle_priv, london_priv

    def assign_special_prestige_points_mapping(self, game):
        circle_mappings = []
        for circle in game.selected_map.specialprestigepoints.circle_data:
            if circle["owner"]:
                # Find the index of the owner player in the game players list
                owner_index = game.players.index(circle["owner"]) + 1  # +1 to handle zero-based indexing
                circle_mappings.append(owner_index)
            else:
                circle_mappings.append(0)  # 0 represents unowned

        # Convert to tensor and ensure it's on the correct device
        special_prestige_points_info = torch.tensor(circle_mappings, device=device, dtype=torch.uint8)
        return special_prestige_points_info

    def assign_bonus_marker_pool_mapping(self, game):
        #12 max BMs in game.selected_map.bonus_marker_pool
        bm_pool_mappings = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bm_mapping = {
            'PlaceAdjacent': 1,
            'SwapOffice': 2,
            'Move3': 3,
            'UpgradeAbility': 4,
            '3Actions': 5,
            '4Actions': 6,
            'ExchangeBonusMarker': 7,
            'Tribute4EstablishingTP': 8,
            'BlockTradeRoute': 9
        }

        for i, bm in enumerate(game.selected_map.bonus_marker_pool):
            bm_pool_mappings[i] = bm_mapping.get(bm, 0)

        # print(f"bm_pool_mappings {bm_pool_mappings}")

        # Convert to tensor and ensure it's on the correct device
        bm_pool_mappings_info = torch.tensor(bm_pool_mappings, device=device, dtype=torch.uint8)
        # print (f"bm_pool_mappings_info {bm_pool_mappings_info}")
        return bm_pool_mappings_info

    def assign_tile_pool_mapping(self, game):
        tile_pool_mappings = [0, 0, 0, 0, 0]
        tile_mapping = {
            'DisplaceAnywhere': 1,
            '+1Action': 2,
            '+1IncomeIfOthersIncome': 3,
            '+1DisplacedPiece': 4,
            '+4PtsPerOwnedCity': 5,
            '+7PtsPerCompletedAbility': 6
        }

        for i, tile in enumerate(game.tile_pool):
            tile_pool_mappings[i] = tile_mapping.get(tile, 0)
        tile_pool_mappings_info = torch.tensor(tile_pool_mappings, device=device, dtype=torch.uint8)
        # print(f"tile_pool_mappings_info {tile_pool_mappings_info}")
        return tile_pool_mappings_info

    def assign_tile_owner_mapping(self, game):
        tile_owner_mappings = [0, 0, 0, 0, 0, 0]

        for player in game.players:
            if game.DisplaceAnywhereOwner == player:
                tile_owner_mappings[0] = player.order + 1  # +1 to handle zero-based indexing
            if game.OneActionOwner == player:
                tile_owner_mappings[1] = player.order + 1
            if game.OneIncomeIfOthersIncomeOwner == player:
                tile_owner_mappings[2] = player.order + 1
            if game.OneDisplacedPieceOwner == player:
                tile_owner_mappings[3] = player.order + 1
            if game.FourPtsPerOwnedCityOwner == player:
                tile_owner_mappings[4] = player.order + 1
            if game.SevenPtsPerCompletedAbilityOwner == player:
                tile_owner_mappings[5] = player.order + 1

        tile_owner_mappings_info = torch.tensor(tile_owner_mappings, device=device, dtype=torch.uint8)
        return tile_owner_mappings_info
        
    def assign_tile_buying_info(self, game):
        tile_mapping = {
            'DisplaceAnywhere': 1,
            '+1Action': 2,
            '+1IncomeIfOthersIncome': 3,
            '+1DisplacedPiece': 4,
            '+4PtsPerOwnedCity': 5,
            '+7PtsPerCompletedAbility': 6
        }
        bm_mapping = {
            'PlaceAdjacent': 1,
            'SwapOffice': 2,
            'Move3': 3,
            'UpgradeAbility': 4,
            '3Actions': 5,
            '4Actions': 6,
            'ExchangeBonusMarker': 7,
            'Tribute4EstablishingTP': 8,
            'BlockTradeRoute': 9
        }

        tile_to_buy = tile_mapping.get(game.tile_to_buy, 0)
        waiting_for_buy_tile_with_bm = 1 if game.waiting_for_buy_tile_with_bm == True else 0
        first_bm_to_spend_on_tile = bm_mapping.get(game.first_bm_to_spend_on_tile, 0)

        tile_buying_info = torch.tensor([tile_to_buy, waiting_for_buy_tile_with_bm, first_bm_to_spend_on_tile], device=device, dtype=torch.uint8)
        return tile_buying_info        
    
    def fill_city_tensor(self, game):
        city_num_attributes = 10  # 10 attributes for city - currently tracking 8, 2 are placeholders
        office_num_attributes = 8  # 8 attributes for each office - currently tracking 5, 3 are placeholders
        self.max_offices = 10  # Maximum number of offices per city
        all_city_info = torch.zeros(MAX_CITIES, city_num_attributes + office_num_attributes * self.max_offices, device=device, dtype=torch.uint8)

        for i, city in enumerate(game.selected_map.cities):
            city_num, city_color = self.assign_city_name_and_color_mapping(game, city)
            city1_upgrade, city2_upgrade = self.assign_city_upgrade_type_mapping(city)
            city_tributes = self.assign_city_tribute_mapping(city)

            city_data = [city_num, city_color, city1_upgrade, city2_upgrade, city_tributes[0], city_tributes[1], city_tributes[2], city_tributes[3]]
            city_data += [0] * (city_num_attributes - len(city_data))  # Pad with zeros

            office_data = [self.assign_office_mapping(office) for office in city.offices]
            office_data += [(0, 0, 0, 0, 0)] * (self.max_offices - len(office_data))  # Pad with zeros if there are fewer than 10 offices

            # Flatten office data and pad with zeros
            office_data_flat = []
            for office in office_data:
                office_flat = list(office)
                office_flat += [0] * (office_num_attributes - len(office_flat))  # Pad with zeros
                office_data_flat += office_flat

            city_tensor = torch.tensor(city_data + office_data_flat, dtype=torch.uint8, device=device)
            all_city_info[i] = city_tensor

        all_city_info = all_city_info.flatten()
        # print(f"all_city_info Size: {all_city_info.size()}")

        return all_city_info

    def assign_city_name_and_color_mapping(self, game, city):
        map1_cities_mapping = {
                'Groningen': 1,
                'Emden': 2,
                'Osnabruck': 3,
                'Kampen': 4,
                'Arnheim': 5,
                'Duisburg': 6,
                'Dortmund': 7,
                'Munster': 8,
                'Coellen': 9,
                'Warburg': 10,
                'Paderborn': 11,
                'Minden': 12,
                'Bremen': 13,
                'Stade': 14,
                'Hannover': 15,
                'Hildesheim': 16,
                'Gottingen': 17,
                'Quedlinburg': 18,
                'Goslar': 19,
                'Brunswick': 20,
                'Luneburg': 21,
                'Hamburg': 22,
                'Lubeck': 23,
                'Perleberg': 24,
                'Stendal': 25,
                'Magdeburg': 26,
                'Halle': 27
            }
        map2_cities_mapping = {
                'Lubeck': 1,
                'Mismar': 2,
                'Stralsund': 3,
                'Malmo': 4,
                'Visby': 5,
                'Danzig': 6,
                'Konigsberg': 7,
                'Belgard': 8,
                'Anklam': 9,
                'Waren': 10,
                'Perleberg': 11,
                'Havelberg': 12,
                'Stettin': 13,
                'Kulm': 14,
                'Elbing': 15,
                'Braunsberg': 16,
                'Allenstein': 17,
                'Frankfurt': 18,
                'BerlinColln': 19,
                'Brandenburg': 20,
                'Tangermunde': 21,
                'Magdeburg': 22,
                'Halle': 23,
                'Mittenberg': 24,
                'Dresden': 25,
                'Breslau': 26,
                'Thorn': 27,
                'Krackau': 28
            }
        map3_cities_mapping = {
                'Glasgom': 1,
                'Edinbaurgh': 2,
                'Dunbar': 3,
                'Falkirk': 4,
                'Carlisle': 5,
                'Newcastle': 6,
                'IsleOfMan': 7,
                'Conway': 8,
                'Chester': 9,
                'Montgomery': 10,
                'Pembroke': 11,
                'Cardiff': 12,
                'Richmond': 13,
                'Durham': 14,
                'Lancaster': 15,
                'York': 16,
                'Hereford': 17,
                'Coventry': 18,
                'Nottingham': 19,
                'Norwich': 20,
                'Cambridge': 21,
                'Ipswich': 22,
                'Oxford': 23,
                'London': 24,
                'Canterbury': 25,
                'Calais': 26,
                'Southhampton': 27,
                'Salisbury': 28,
                'Plymouth': 29,
                'Bristol': 30
            }
        city_colors = [GREY, BLACKISH_BROWN, DARK_RED, DARK_GREEN, DARK_BLUE, (65, 103, 114)]

        if game.map_num == 1:
            city_num = map1_cities_mapping.get(city.name, 0)
        elif game.map_num == 2:
            city_num = map2_cities_mapping.get(city.name, 0)
        elif game.map_num == 3:
            city_num = map3_cities_mapping.get(city.name, 0)
        city_color = city_colors.index(city.color) + 1
        return city_num, city_color

    def assign_city_upgrade_type_mapping(self, city):
        city1_upgrade = 0
        city2_upgrade = 0
        upgrade_type_mapping = {
                'Keys': 1,
                'Privilege': 2,
                'Book': 3,
                'Actions': 4,
                'Bank': 5,
                'SpecialPrestigePoints': 6
            }
            # Assign values based on the upgrades present in the city
        if len(city.upgrade_city_type) > 0:
            city1_upgrade = upgrade_type_mapping.get(city.upgrade_city_type[0], 0)
        if len(city.upgrade_city_type) > 1:
            city2_upgrade = upgrade_type_mapping.get(city.upgrade_city_type[1], 0)

        return city1_upgrade, city2_upgrade

    def assign_city_tribute_mapping(self, city):
        city_tributes = [0] * 4
        for i, player in enumerate(city.tributed_players):
            if not player:
                city_tributes[i] = 0
            else:
                city_tributes[i] = player.order + 1  # +1 to handle zero-based indexing
        return city_tributes 

    def assign_office_mapping(self, office):
        office_shape_mapping = {
                'square': 1,
                'circle': 2
            }
        office_color_mapping = {
                tuple(WHITE): 1,
                tuple(ORANGE): 2,
                tuple(PINK): 3,
                tuple(BLACK): 4,
                tuple(GREEN): 5,
                tuple(BLUE): 6,
                tuple(PURPLE): 7,
                tuple(RED): 8,
                tuple(YELLOW): 9
            }

        office_shape = office_shape_mapping.get(office.shape, 0)
        office_color = office_color_mapping.get(tuple(office.color), 0)
        office_controller = office.controller.order + 1 if office.controller else 0
        office_points = office.awards_points
        office_place_adjacent = 1 if office.place_adjacent_office else 0
        return office_shape, office_color, office_controller, office_points, office_place_adjacent

    def fill_route_tensor(self, game):
        self.route_num_attributes = 10  # 10 attributes for route
        self.post_num_attributes = 5  # 5 attributes for each post
        self.max_posts = 5  # Maximum number of posts per route
        all_route_info = torch.zeros(MAX_ROUTES, self.route_num_attributes + self.post_num_attributes * self.max_posts, device=device, dtype=torch.uint8)

        for i, route in enumerate(game.selected_map.routes):
            # Get numerical values for route attributes
            city1_num, _ = self.assign_city_name_and_color_mapping(game, route.cities[0])
            city2_num, _ = self.assign_city_name_and_color_mapping(game, route.cities[1])
            route_region = self.assign_region_mapping(route.region)
            route_has_bm, route_bm_type = self.assign_bonus_marker_mapping(route.has_bonus_marker, route.bonus_marker)
            route_perm_bm = self.assign_permanent_bm_mapping(route.has_permanent_bm_type)

            route_info = [city1_num, city2_num, route.num_posts, route_region, route_has_bm, route_bm_type, route_perm_bm]
            route_info += [0] * (self.route_num_attributes - len(route_info))  # Pad with zeros

            # Get post information
            post_info = []
            for post in route.posts:
                owner_shape = self.assign_post_shape_mapping(post.owner_piece_shape)
                post_owner = self.assign_player_mapping(game, post.owner)
                post_blocked = 1 if post.blocked_bm else 0 
                post_data = [owner_shape, post_owner, post_blocked]
                post_data += [0] * (self.post_num_attributes - len(post_data))  # Pad with zeros
                post_info.extend(post_data)

            # Pad post_info if fewer than 5 posts
            post_info += [0] * ((self.max_posts - len(route.posts)) * self.post_num_attributes)

            # Combine data into a single tensor
            all_route_info[i] = torch.tensor(route_info + post_info, dtype=torch.uint8, device=device)

        all_route_info = all_route_info.flatten()
        # print(f"all_route_info Size: {all_route_info.size()}")
        return all_route_info

    def assign_region_mapping(self, region):
        region_mapping = {
            'Scotland': 1,
            'Wales': 2,
            None: 0  # No specific region
        }
        return region_mapping.get(region, 0)

    def assign_bonus_marker_mapping(self, has_bonus_marker, bonus_marker):
        bm_type = 0
        bm_mapping = {
            'PlaceAdjacent': 1,
            'SwapOffice': 2,
            'Move3': 3,
            'UpgradeAbility': 4,
            '3Actions': 5,
            '4Actions': 6,
            'ExchangeBonusMarker': 7,
            'Tribute4EstablishingTP': 8,
            'BlockTradeRoute': 9
        }
        if has_bonus_marker:
            bm_type = bm_mapping.get(bonus_marker.type, 0)

        return has_bonus_marker, bm_type  

    def assign_permanent_bm_mapping(self, permanent_bm_type):
        # Example mapping, modify based on your game's permanent bonus markers
        permanent_bm_mapping = {
            'MoveAny2': 1,
            '+1Priv': 2,
            'ClaimGreenCity': 3,
            'Place2TradesmenFromRoute': 4,
            'Place2ScotlandOrWales': 5,
            None: 0
        }
        return permanent_bm_mapping.get(permanent_bm_type, 0)

    def assign_post_shape_mapping(self, shape):
        shape_mapping = {
            'circle': 1,
            'square': 2,
            None: 0
        }
        return shape_mapping.get(shape, 0)

    def assign_player_mapping(self, game, player):
        # Map player objects to a unique identifier (e.g., player index)
        if player is None:
            return 0
        else:
            return game.players.index(player) + 1  # Assuming game.players is a list of players

    def fill_player_info_tensor(self, game):
        max_players = 5 
        num_player_attributes = 55

        # Precompute priv and bank mappings for all players
        priv_mappings = [self.assign_priv_mapping(player) for player in game.players]
        bm_mappings = [self.assign_bm_mapping(player) for player in game.players]
        player_unused_bm, player_used_bm = zip(*bm_mappings)  # Unpack the tuples into two lists

        # Initialize the tensor on the appropriate device
        player_info = torch.zeros(max_players, num_player_attributes, device=device, dtype=torch.uint8)

        # Fill in the data for the actual players
        for i, player in enumerate(game.players):
            player_data = [player.order, player.score, player.final_score, player.pieces_to_pickup, player.pieces_to_place,
                        player.keys_index, player.keys, priv_mappings[i], player.book, player.actions_index, player.actions, player.actions_remaining,
                        player.bank, player.general_stock_squares, player.general_stock_circles, player.personal_supply_squares,
                        player.personal_supply_circles, int(player.ending_turn), player.brown_priv_count, player.blue_priv_count]

            mission_card_city1, mission_card_city2, mission_card_city3 = self.assign_mission_card_mapping(game, player)
            player_data.extend([mission_card_city1, mission_card_city2, mission_card_city3])

            # Add bonus marker info to player_data
            player_data.extend(list(player_unused_bm[i]) + list(player_used_bm[i]))

            # Pad player_data with zeros until it reaches the desired length of 55, incase we forget some attributes to keep track of without breaking the model
            player_data.extend([0] * (55 - len(player_data)))

            # Combine all info into a single tensor
            player_info[i] = torch.tensor(player_data, dtype=torch.uint8, device=device)

        
        player_info = player_info.flatten()
        # print(f"player_info Size: {player_info.size()}")
        return player_info

    def assign_priv_mapping(self, player):
        priv_mapping = {
                'WHITE': 1,
                'ORANGE': 2,
                'PINK': 3,
                'BLACK': 4
            }
        return priv_mapping.get(player.privilege, 0)

    def assign_bm_mapping(self, player):
        bm_mapping = {
            'PlaceAdjacent': 1,
            'SwapOffice': 2,
            'Move3': 3,
            'UpgradeAbility': 4,
            '3Actions': 5,
            '4Actions': 6,
            'ExchangeBonusMarker': 7,
            'Tribute4EstablishingTP': 8,
            'BlockTradeRoute': 9
        }

        player_unused_bm = [bm_mapping.get(bm.type, 0) for bm in player.bonus_markers]
        player_used_bm = [bm_mapping.get(bm.type, 0) for bm in player.used_bonus_markers]

        # Padding if necessary
        player_unused_bm += [0] * (12 - len(player_unused_bm))
        player_used_bm += [0] * (12 - len(player_used_bm))

        return player_unused_bm, player_used_bm

    def assign_mission_card_mapping(self, game, player):
        mission_card_cities = [0, 0, 0]

        if game.map_num == 1:
            for i, city_name in enumerate(player.mission_card):
                # Find the city object with the matching name
                city = next((c for c in game.selected_map.cities if c.name == city_name), None)
                if city is not None:
                    city_num, _ = self.assign_city_name_and_color_mapping(game, city)  # Ignore the color
                    mission_card_cities[i] = city_num

        return tuple(mission_card_cities)
    
    def load_game_state_JSON(self, filename):
        with open(filename, 'r') as f:
            game_state_JSON = json.load(f)

        game = Game(game_state_JSON['game_info']['map_num'], game_state_JSON['game_info']['num_players'])

        self.load_game_info_JSON(game, game_state_JSON['game_info'])
        self.load_city_info_JSON(game, game_state_JSON['city_info'])
        self.load_route_info_JSON(game, game_state_JSON['route_info'])
        self.load_player_info_JSON(game, game_state_JSON['player_info'])

        return game
    
    def load_game_info_JSON(self, game, game_info_JSON):
        game.active_player = game_info_JSON['active_player']
        game.replace_bonus_marker = game_info_JSON['replace_bonus_marker']
        game.current_player_index = game_info_JSON['current_player_index']
        game.current_player = game.players[game.current_player_index]
        game.current_full_cities_count = game_info_JSON['current_full_cities_count']
        game.east_west_completed_count = game_info_JSON['east_west_completed_count']
        game.waiting_for_bm_swap_office = game_info_JSON['waiting_for_bm_swap_office']
        game.waiting_for_bm_upgrade_ability = game_info_JSON['waiting_for_bm_upgrade_ability']
        game.waiting_for_bm_move_any_2 = game_info_JSON['waiting_for_bm_move_any_2']
        game.waiting_for_bm_move3 = game_info_JSON['waiting_for_bm_move3']
        print(f"waiting_for_bm_move3: {game.waiting_for_bm_move3}")
        game.waiting_for_bm_exchange_bm = game_info_JSON['waiting_for_bm_exchange_bm']
        game.waiting_for_bm_tribute_trading_post = game_info_JSON['waiting_for_bm_tribute_trading_post']
        game.waiting_for_bm_block_trade_route = game_info_JSON['waiting_for_bm_block_trade_route']

        game.cardiff_prov = game.player[game_info_JSON['cardiff_priv']] if game_info_JSON['cardiff_priv'] else None
        game.carlisle_prov = game.player[game_info_JSON['carlisle_priv']] if game_info_JSON['carlisle_priv'] else None
        game.london_prov = game.player[game_info_JSON['london_priv']] if game_info_JSON['london_priv'] else None
        
        colors = ['white', 'orange', 'pink', 'black']
        for i, color in enumerate(colors):
            owner_order = game_info_JSON[f'special_prestige_points_{color}']
            if owner_order is not None:
                game.selected_map.specialprestigepoints.circle_data[i]['owner'] = game.players[owner_order - 1]
            else:
                game.selected_map.specialprestigepoints.circle_data[i]['owner'] = None

        for i in range(12):
            bonus_marker_name = game_info_JSON[f'bonus_marker{i+1}']
            if bonus_marker_name is not None:
                # Assuming get_bonus_marker_by_name is a function that returns a bonus marker given its name
                game.selected_map.bonus_marker_pool[i] = bonus_marker_name

        # Unmap tile_pool_info_JSON
        for i in range(len(game.players)):
            tile_pool = game_info_JSON[f'available_tile{i+1}']
            if tile_pool is not None:
                game.tile_pool[i] = tile_pool

        # Unmap tile_owner_info_JSON
        if game_info_JSON['tile_owner_DisplaceAnywhere'] is not None:
            game.DisplaceAnywhereOwner = game.players[game_info_JSON['tile_owner_DisplaceAnywhere'] - 1]

        if game_info_JSON['tile_owner_+1Action'] is not None:
            game.OneActionOwner = game.players[game_info_JSON['tile_owner_+1Action'] - 1]

        if game_info_JSON['tile_owner_+1IncomeIfOthersIncome'] is not None:
            game.OneIncomeIfOthersIncomeOwner = game.players[game_info_JSON['tile_owner_+1IncomeIfOthersIncome'] - 1]

        if game_info_JSON['tile_owner_+1DisplacedPiece'] is not None:
            game.OneDisplacedPieceOwner = game.players[game_info_JSON['tile_owner_+1DisplacedPiece'] - 1]

        if game_info_JSON['tile_owner_+4PtsPerOwnedCity'] is not None:
            game.FourPtsPerOwnedCityOwner = game.players[game_info_JSON['tile_owner_+4PtsPerOwnedCity'] - 1]

        if game_info_JSON['tile_owner_+7PtsPerCompletedAbility'] is not None:
            game.SevenPtsPerCompletedAbilityOwner = game.players[game_info_JSON['tile_owner_+7PtsPerCompletedAbility'] - 1]

        # Unmap tile_to_buy_info_JSON
        game.tile_to_buy = game_info_JSON['tile_to_buy']
        game.waiting_for_buy_tile_with_bm = game_info_JSON['waiting_for_buy_tile_with_bm']
        game.first_bm_to_spend_on_tile = game_info_JSON['first_bm_to_spend_on_tile']
    
    def load_city_info_JSON(self, game, city_info_JSON):
        city_colors = {
            'grey': GREY,
            'blackish_brown': BLACKISH_BROWN,
            'dark_red': DARK_RED,
            'dark_green': DARK_GREEN,
            'dark_blue': DARK_BLUE,
            'blue_brown_city': (65, 103, 114),
        }
        player_colors = {
            'green': GREEN,
            'blue': BLUE,
            'purple': PURPLE,
            'red': RED,
            'yellow': YELLOW
        }

        for i, city in enumerate(game.selected_map.cities):
            if city.name == city_info_JSON[i]['city_name'] and city.color == city_colors[city_info_JSON[i]['city_color']]:
                # Initialize the tributed players
                city.tributed_players = [None, None, None, None]

                # Load the tributed players
                for j in range(4):
                    player_index = city_info_JSON[i][f'city_tribute{j}']
                    if player_index is not None:
                        city.tributed_players[j] = game.players[player_index - 1]

                for j, office_data in enumerate(city_info_JSON[i]['offices']):
                    # Get the office from the city
                    office = city.offices[j]

                    # Map the color
                    office.color = office_data['office_color']

                    # Map the controller
                    if office_data['office_controller'] is not None:
                        for player in game.players:
                            if player.color == office_data['office_controller']:
                                office.controller = player

                    # Map the place_adjacent attribute
                    office.place_adjacent_office = office_data['office_place_adjacent']

    def load_route_info_JSON(self, game, route_info_JSON):
        for route_data in route_info_JSON:
            for route in game.selected_map.routes:
                if set(route.cities[i].name for i in range(2)) == set(route_data['route_cities']):
                    route.has_bonus_marker = route_data['route_has_bm']
                    if route_data['route_bm_type'] is not None:
                        route.bonus_marker = BonusMarker(route_data['route_bm_type'])
                    else:
                        route.bonus_marker = None
                    route.has_permanent_bm_type = route_data['route_perm_bm']

                    for i, post_data in enumerate(route_data['posts']):
                        post = route.posts[i]
                        if post_data['post_owner'] is not None:
                            for player in game.players:
                                if player.color == tuple(post_data['post_owner']):
                                    post.claim(player, post_data['post_owner_shape'])
                                    break
                        else:
                            post.owner = None
                        post.blocked_bm = post_data['post_blocked']

    def load_player_info_JSON(self, game, player_info_JSON):
        # Reverse dictionary to map color names to RGB values
        color_names = {
            'green': GREEN,
            'blue': BLUE,
            'purple': PURPLE,
            'red': RED,
            'yellow': YELLOW
        }
        for player_data in player_info_JSON:
            # Convert color name to RGB value
            player_data_color = color_names[player_data['player_color']]

            for player in game.players:
                if player.color == player_data_color and player.order == player_data['player_order']:
                    player.score = player_data['player_score']
                    player.keys = player_data['player_keys']
                    player.privilege = player_data['player_privilege']
                    player.book = player_data['player_book']
                    player.actions_index = player_data['player_actions_index']
                    player.actions = player_data['player_actions']
                    player.actions_remaining = player_data['player_actions_remaining']
                    player.bank = player_data['player_bank']
                    player.general_stock_squares = player_data['player_general_stock_squares']
                    player.general_stock_circles = player_data['player_general_stock_circles']
                    player.personal_supply_squares = player_data['player_personal_supply_squares']
                    player.personal_supply_circles = player_data['player_personal_supply_circles']
                    player.brown_priv_count = player_data['player_brown_priv_count']
                    player.blue_priv_count = player_data['player_blue_priv_count']

                    for bm in player_data.get('player_unused_bonus_markers', []):
                        player.bonus_markers.append(BonusMarker(type=bm, owner=player))

                    for bm in player_data.get('player_used_bonus_markers', []):
                        player.used_bonus_markers.append(BonusMarker(type=bm, owner=player))
    
    def save_game_state_JSON(self, game):
        game_state_JSON = {}
        game_info_JSON = self.fill_game_info_JSON(game)
        city_info_JSON = self.fill_city_info_JSON(game)
        route_info_JSON = self.fill_route_info_JSON(game)
        player_info_JSON = self.fill_player_info_JSON(game)

        # # Combine all JSON data into a single dictionary
        game_state_JSON['game_info'] = game_info_JSON
        game_state_JSON['city_info'] = city_info_JSON
        game_state_JSON['route_info'] = route_info_JSON
        game_state_JSON['player_info'] = player_info_JSON

        # save game_state_JSON to a file
        with open('game_state_JSON.json', 'w') as f:
            json.dump(game_state_JSON, f, indent=4)

    def fill_game_info_JSON(self, game):
        game_info_JSON = {
            'map_num': game.map_num,
            'num_players': game.num_players,
            'active_player': game.active_player,
            'replace_bonus_marker': game.replace_bonus_marker,
            'current_player_index': game.current_player_index,
            'max_full_cities': game.selected_map.max_full_cities,
            'current_full_cities_count': game.current_full_cities_count,
            'east_west_completed_count': game.east_west_completed_count,
            'waiting_for_bm_swap_office': game.waiting_for_bm_swap_office,
            'waiting_for_bm_upgrade_ability': game.waiting_for_bm_upgrade_ability,
            'waiting_for_bm_move_any_2': game.waiting_for_bm_move_any_2,
            'waiting_for_bm_move3': game.waiting_for_bm_move3,
            'waiting_for_bm_exchange_bm': game.waiting_for_bm_exchange_bm,
            'waiting_for_bm_tribute_trading_post': game.waiting_for_bm_tribute_trading_post,
            'waiting_for_bm_block_trade_route': game.waiting_for_bm_block_trade_route,
            'waiting_for_bm_green_city': game.waiting_for_bm_green_city,
            'waiting_for_place2_in_scotland_or_wales': game.waiting_for_place2_in_scotland_or_wales,
            'cardiff_priv': game.cardiff_priv.order if game.cardiff_priv else None,
            'carlisle_priv': game.carlisle_priv.order if game.carlisle_priv else None,
            'london_priv': game.london_priv.order if game.london_priv else None
        }

        # Special Prestige Points info
        special_prestige_points_info_JSON = {
            'special_prestige_points_white': None,
            'special_prestige_points_orange': None,
            'special_prestige_points_pink': None,
            'special_prestige_points_black': None
        }

        # Special Prestige Points info
        if game.selected_map.specialprestigepoints.circle_data:
            colors = ['white', 'orange', 'pink', 'black']

            for i, color in enumerate(colors):
                circle_data = game.selected_map.specialprestigepoints.circle_data[i]
                if 'owner' in circle_data and circle_data['owner']:
                    special_prestige_points_info_JSON[f'special_prestige_points_{color}'] = circle_data['owner'].order
        
        bonus_marker_pool_info_JSON = {}
        for i in range(len(game.selected_map.bonus_marker_pool)):
            bonus_marker_pool_info_JSON[f'bonus_marker{i+1}'] = game.selected_map.bonus_marker_pool[i] if game.selected_map.bonus_marker_pool[i] else None

        tile_pool_info_JSON = {}
        for i in range(len(game.players)):
            tile_pool_info_JSON[f'available_tile{i+1}'] = game.tile_pool[i] if game.tile_pool[i] else None

        tile_owner_info_JSON = {
            'tile_owner_DisplaceAnywhere': game.DisplaceAnywhereOwner.order if game.DisplaceAnywhereOwner else None,
            'tile_owner_+1Action': game.OneActionOwner.order if game.OneActionOwner else None,
            'tile_owner_+1IncomeIfOthersIncome': game.OneIncomeIfOthersIncomeOwner.order if game.OneIncomeIfOthersIncomeOwner else None,
            'tile_owner_+1DisplacedPiece': game.OneDisplacedPieceOwner.order if game.OneDisplacedPieceOwner else None,
            'tile_owner_+4PtsPerOwnedCity': game.FourPtsPerOwnedCityOwner.order if game.FourPtsPerOwnedCityOwner else None,
            'tile_owner_+7PtsPerCompletedAbility': game.SevenPtsPerCompletedAbilityOwner.order if game.SevenPtsPerCompletedAbilityOwner else None
        }
        tile_to_buy_info_JSON = {
            'tile_to_buy': game.tile_to_buy,
            'waiting_for_buy_tile_with_bm': game.waiting_for_buy_tile_with_bm,
            'first_bm_to_spend_on_tile': game.first_bm_to_spend_on_tile
        }

        # Combine all game info into one JSON dictionary
        game_info_JSON.update(special_prestige_points_info_JSON)
        game_info_JSON.update(bonus_marker_pool_info_JSON)
        game_info_JSON.update(tile_pool_info_JSON)
        game_info_JSON.update(tile_owner_info_JSON)
        game_info_JSON.update(tile_to_buy_info_JSON)

        # print(json.dumps(game_info_JSON, indent=4))
        # print("Size of the game_info_JSON dictionary:", len(game_info_JSON))

        return game_info_JSON

    def fill_city_info_JSON(self, game):
        city_info_JSON = []
        city_colors = {
            GREY: 'grey',
            BLACKISH_BROWN: 'blackish_brown',
            DARK_RED: 'dark_red',
            DARK_GREEN: 'dark_green',
            DARK_BLUE: 'dark_blue',
            (65, 103, 114): 'blue_brown_city',
        }
        player_colors = {
            GREEN: 'green',
            BLUE: 'blue',
            PURPLE: 'purple',
            RED: 'red',
            YELLOW: 'yellow'
        }
        for city in game.selected_map.cities:
            city_data_JSON = {
                'city_name': city.name,
                'city_color': city_colors[city.color],
                'city1_upgrade': city.upgrade_city_type[0] if city.upgrade_city_type else None,
                'city2_upgrade': city.upgrade_city_type[1] if len(city.upgrade_city_type) > 1 else None,
                'city_tribute0': city.tributed_players[0] if city.tributed_players else None,
                'city_tribute1': city.tributed_players[1] if len(city.tributed_players) > 1 else None,
                'city_tribute2': city.tributed_players[2] if len(city.tributed_players) > 2 else None,
                'city_tribute3': city.tributed_players[3] if len(city.tributed_players) > 3 else None
            }

            office_data_JSON = []
            for office in city.offices:
                office_data = {
                    'office_shape': office.shape,
                    'office_color': office.color if office.color else None,
                    'office_controller': player_colors[office.controller.color] if office.controller else None,
                    'office_points': office.awards_points,
                    'office_place_adjacent': office.place_adjacent_office
                }
                office_data_JSON.append(office_data)

            city_data_JSON['offices'] = office_data_JSON
            city_info_JSON.append(city_data_JSON)

        # print(json.dumps(city_info_JSON, indent=4))
        # total_size = sum(len(city_dict) for city_dict in city_info_JSON)
        # print("Total size of all dictionaries in city_info_JSON:", total_size)

        return city_info_JSON
    
    def fill_route_info_JSON(self, game):
        route_info_JSON = []
        for route in game.selected_map.routes:
            route_data_JSON = {
                'route_cities': [city.name for city in route.cities],
                'route_regions': route.region,
                'route_has_bm': route.has_bonus_marker,
                'route_bm_type': route.bonus_marker.type if route.bonus_marker else None,
                'route_perm_bm': route.has_permanent_bm_type,
            }

            post_data_JSON = []
            for post in route.posts:
                post_data = {
                    'post_owner': post.owner.color if post.owner else None,
                    'post_owner_shape': post.owner_piece_shape,
                    'post_required_shape': post.required_shape,
                    'post_region' : post.region,
                    'post_blocked': post.blocked_bm
                }
                post_data_JSON.append(post_data)

            route_data_JSON['posts'] = post_data_JSON
            route_info_JSON.append(route_data_JSON)

        # print(json.dumps(route_info_JSON, indent=4))
        # total_size = sum(len(route_dict) for route_dict in route_info_JSON)
        # print("Total size of all dictionaries in route_info_JSON:", total_size)

        return route_info_JSON
    
    def fill_player_info_JSON(self, game):
        player_info_JSON = []

        player_colors = {
            GREEN: 'green',
            BLUE: 'blue',
            PURPLE: 'purple',
            RED: 'red',
            YELLOW: 'yellow'
        }

        for player in game.players:
            player_data_JSON = {
                'player_color': player_colors[player.color],
                'player_order': player.order,
                'player_score': player.score,
                'player_final_score': player.final_score,
                'player_keys_index': player.keys_index,
                'player_keys': player.keys,
                'player_privilege': player.privilege,
                'player_book': player.book,
                'player_actions_index': player.actions_index,
                'player_actions': player.actions,
                'player_actions_remaining': player.actions_remaining,
                'player_bank': player.bank,
                'player_general_stock_squares': player.general_stock_squares,
                'player_general_stock_circles': player.general_stock_circles,
                'player_personal_supply_squares': player.personal_supply_squares,
                'player_personal_supply_circles': player.personal_supply_circles,
                'player_brown_priv_count': player.brown_priv_count,
                'player_blue_priv_count': player.blue_priv_count,
                'player_mission_card_cities' : [city for city in player.mission_card] if player.mission_card else None,
                'player_ending_turn': player.ending_turn
            }
            for bm in player.bonus_markers:
                player_data_JSON[f'player_bm_{bm.type}'] = bm.type if bm else None
            for used_bm in player.used_bonus_markers:
                player_data_JSON[f'player_used_bm_{used_bm.type}'] = used_bm.type if used_bm else None
            
            player_info_JSON.append(player_data_JSON)
        print(json.dumps(player_info_JSON, indent=4))
        total_size = sum(len(player_dict) for player_dict in player_info_JSON)
        print("Total size of all dictionaries in player_info_JSON:", total_size)

        return player_info_JSON