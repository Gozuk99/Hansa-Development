"""Fixed-size neural-network observation encoding for Hansa game positions."""

import torch
from map_data.constants import (
    GREEN,
    BLUE,
    PURPLE,
    RED,
    YELLOW,
    BLACKISH_BROWN,
    DARK_RED,
    DARK_GREEN,
    DARK_BLUE,
    GREY,
    MAX_CITIES,
    MAX_ROUTES,
    COLOR_NAMES,
    WHITE,
    ORANGE,
    PINK,
    BLACK,
)
from map_data.map_attributes import BonusMarker

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ObservationEncoder:
    GAME_INFO_SIZE = 70
    CITY_NUM_ATTRIBUTES = 10
    OFFICE_NUM_ATTRIBUTES = 8
    MAX_OFFICES = 10
    ROUTE_NUM_ATTRIBUTES = 10
    POST_NUM_ATTRIBUTES = 5
    MAX_POSTS = 5
    MAX_PLAYERS = 5
    PLAYER_ATTRIBUTES = 55

    BONUS_MARKER_TYPE_TO_ID = {
        "PlaceAdjacent": 1,
        "SwapOffice": 2,
        "Move3": 3,
        "UpgradeAbility": 4,
        "3Actions": 5,
        "4Actions": 6,
        "ExchangeBonusMarker": 7,
        "Tribute4EstablishingTP": 8,
        "BlockTradeRoute": 9,
    }

    TILE_TYPE_TO_ID = {
        "DisplaceAnywhere": 1,
        "+1Action": 2,
        "+1IncomeIfOthersIncome": 3,
        "+1DisplacedPiece": 4,
        "+4PtsPerOwnedCity": 5,
        "+7PtsPerCompletedAbility": 6,
    }

    PERMANENT_BM_TYPE_TO_ID = {
        "MoveAny2": 1,
        "+1Priv": 2,
        "ClaimGreenCity": 3,
        "Place2TradesmenFromRoute": 4,
        "Place2ScotlandOrWales": 5,
        None: 0,
    }

    REGION_MAPPING = {"Scotland": 1, "Wales": 2, None: 0}

    POST_SHAPE_MAPPING = {"circle": 1, "square": 2, None: 0}

    OFFICE_SHAPE_MAPPING = {"square": 1, "circle": 2}

    OFFICE_COLOR_MAPPING = {
        tuple(WHITE): 1,
        tuple(ORANGE): 2,
        tuple(PINK): 3,
        tuple(BLACK): 4,
        tuple(GREEN): 5,
        tuple(BLUE): 6,
        tuple(PURPLE): 7,
        tuple(RED): 8,
        tuple(YELLOW): 9,
    }

    PRIVILEGE_COLORS = [GREEN, BLUE, PURPLE, RED, YELLOW]

    UPGRADE_TYPE_TO_ID = {
        "Keys": 1,
        "Privilege": 2,
        "Book": 3,
        "Actions": 4,
        "Bank": 5,
        "SpecialPrestigePoints": 6,
    }

    CITY_COLOR_VALUES = [GREY, BLACKISH_BROWN, DARK_RED, DARK_GREEN, DARK_BLUE, (65, 103, 114)]

    MAP_CITY_NAME_MAPPINGS = {
        1: {
            "Groningen": 1,
            "Emden": 2,
            "Osnabruck": 3,
            "Kampen": 4,
            "Arnheim": 5,
            "Duisburg": 6,
            "Dortmund": 7,
            "Munster": 8,
            "Coellen": 9,
            "Warburg": 10,
            "Paderborn": 11,
            "Minden": 12,
            "Bremen": 13,
            "Stade": 14,
            "Hannover": 15,
            "Hildesheim": 16,
            "Gottingen": 17,
            "Quedlinburg": 18,
            "Goslar": 19,
            "Brunswick": 20,
            "Luneburg": 21,
            "Hamburg": 22,
            "Lubeck": 23,
            "Perleberg": 24,
            "Stendal": 25,
            "Magdeburg": 26,
            "Halle": 27,
        },
        2: {
            "Lubeck": 1,
            "Mismar": 2,
            "Stralsund": 3,
            "Malmo": 4,
            "Visby": 5,
            "Danzig": 6,
            "Konigsberg": 7,
            "Belgard": 8,
            "Anklam": 9,
            "Waren": 10,
            "Perleberg": 11,
            "Havelberg": 12,
            "Stettin": 13,
            "Kulm": 14,
            "Elbing": 15,
            "Braunsberg": 16,
            "Allenstein": 17,
            "Frankfurt": 18,
            "BerlinColln": 19,
            "Brandenburg": 20,
            "Tangermunde": 21,
            "Magdeburg": 22,
            "Halle": 23,
            "Mittenberg": 24,
            "Dresden": 25,
            "Breslau": 26,
            "Thorn": 27,
            "Krackau": 28,
        },
        3: {
            "Glasgom": 1,
            "Edinbaurgh": 2,
            "Dunbar": 3,
            "Falkirk": 4,
            "Carlisle": 5,
            "Newcastle": 6,
            "IsleOfMan": 7,
            "Conway": 8,
            "Chester": 9,
            "Montgomery": 10,
            "Pembroke": 11,
            "Cardiff": 12,
            "Richmond": 13,
            "Durham": 14,
            "Lancaster": 15,
            "York": 16,
            "Hereford": 17,
            "Coventry": 18,
            "Nottingham": 19,
            "Norwich": 20,
            "Cambridge": 21,
            "Ipswich": 22,
            "Oxford": 23,
            "London": 24,
            "Canterbury": 25,
            "Calais": 26,
            "Southhampton": 27,
            "Salisbury": 28,
            "Plymouth": 29,
            "Bristol": 30,
        },
    }

    def __init__(self):
        self.game_tensor_size = 0
        self.city_tensor_size = 0
        self.route_tensor_size = 0
        self.player_tensor_size = 0
        self.all_game_state_size = 0

    def _tensor(self, values, length=None):
        tensor = torch.tensor(values, device=device, dtype=torch.uint8)
        if length is not None and tensor.numel() < length:
            tensor = torch.cat(
                (tensor, torch.zeros(length - tensor.numel(), device=device, dtype=torch.uint8)),
                dim=0,
            )
        return tensor

    def _mapped_list(self, values, mapping, length=None):
        mapped = [mapping.get(value, 0) for value in values]
        if length is not None:
            mapped.extend([0] * max(0, length - len(mapped)))
        return mapped

    def get_game_state(self, game):
        game_tensor = self.fill_game_tensor(game)
        city_tensor = self.fill_city_tensor(game)
        route_tensor = self.fill_route_tensor(game)
        player_tensor = self.fill_player_info_tensor(game)

        self.game_tensor_size = game_tensor.numel()
        self.city_tensor_size = city_tensor.numel()
        self.route_tensor_size = route_tensor.numel()
        self.player_tensor_size = player_tensor.numel()

        flattened_game_state = torch.cat(
            [game_tensor, city_tensor, route_tensor, player_tensor], dim=0
        )
        self.all_game_state_size = flattened_game_state.numel()
        return flattened_game_state

    def fill_game_tensor(self, game):
        # Initial game info
        initial_game_info = torch.tensor(
            [
                game.map_num,
                game.num_players,
                game.active_player,
                game.replace_bonus_marker,
                game.current_player_index,
                game.selected_map.max_full_cities,
                game.current_full_cities_count,
                game.east_west_completed_count,
                game.waiting_for_bm_swap_office,
                game.waiting_for_bm_upgrade_ability,
                game.waiting_for_bm_move_any_2,
                game.waiting_for_bm_move3,
                game.waiting_for_bm_exchange_bm,
                game.waiting_for_bm_tribute_trading_post,
                game.waiting_for_bm_block_trade_route,
                game.waiting_for_bm_green_city,
                game.waiting_for_place2_in_scotland_or_wales,
            ],
            device=device,
            dtype=torch.uint8,
        )

        # Privileges info
        cardiff_priv, carlisle_priv, london_priv = self.assign_blue_brown_priv_mapping(game)
        privileges_info = torch.tensor(
            [cardiff_priv, carlisle_priv, london_priv], device=device, dtype=torch.uint8
        )

        # Special Prestige Points info
        special_prestige_points_info = self.assign_special_prestige_points_mapping(game)
        bonus_marker_info = self.assign_bonus_marker_pool_mapping(game)
        tile_pool_info = self.assign_tile_pool_mapping(game)
        tile_owner_info = self.assign_tile_owner_mapping(game)
        tile_to_buy_info = self.assign_tile_buying_info(game)

        # Concatenate all game info into one tensor
        game_info = torch.cat(
            (
                initial_game_info,
                privileges_info,
                special_prestige_points_info,
                bonus_marker_info,
                tile_pool_info,
                tile_owner_info,
                tile_to_buy_info,
            ),
            dim=0,
        ).unsqueeze(0)  # Add an extra dimension for batch size
        # flatten the tensor
        game_info = game_info.flatten()

        # Pad zeroes to the end until the size of tensor is of size 70
        # This is to allow any missing data to be added without fear of breaking the model
        game_info = torch.cat(
            (game_info, torch.zeros(70 - game_info.size()[0], device=device, dtype=torch.uint8)),
            dim=0,
        )

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
                owner_index = game.players.index(circle["owner"]) + 1
                circle_mappings.append(owner_index)
            else:
                circle_mappings.append(0)

        return torch.tensor(circle_mappings, device=device, dtype=torch.uint8)

    def assign_bonus_marker_pool_mapping(self, game):
        bm_pool_mappings = [0] * 12
        for i, bm in enumerate(game.selected_map.bonus_marker_pool):
            bm_pool_mappings[i] = self.BONUS_MARKER_TYPE_TO_ID.get(bm, 0)

        return torch.tensor(bm_pool_mappings, device=device, dtype=torch.uint8)

    def assign_tile_pool_mapping(self, game):
        tile_pool_mappings = [0] * 5
        for i, tile in enumerate(game.tile_pool):
            tile_pool_mappings[i] = self.TILE_TYPE_TO_ID.get(tile, 0)

        return torch.tensor(tile_pool_mappings, device=device, dtype=torch.uint8)

    def assign_tile_owner_mapping(self, game):
        tile_owner_mappings = [0] * 6
        owners = [
            game.DisplaceAnywhereOwner,
            game.OneActionOwner,
            game.OneIncomeIfOthersIncomeOwner,
            game.OneDisplacedPieceOwner,
            game.FourPtsPerOwnedCityOwner,
            game.SevenPtsPerCompletedAbilityOwner,
        ]

        for i, owner in enumerate(owners):
            if owner is not None:
                tile_owner_mappings[i] = owner.order + 1

        return torch.tensor(tile_owner_mappings, device=device, dtype=torch.uint8)

    def assign_tile_buying_info(self, game):
        tile_to_buy = self.TILE_TYPE_TO_ID.get(game.tile_to_buy, 0)
        waiting_for_buy_tile_with_bm = 1 if game.waiting_for_buy_tile_with_bm else 0
        first_bm_to_spend_on_tile = self.BONUS_MARKER_TYPE_TO_ID.get(
            game.first_bm_to_spend_on_tile, 0
        )

        return torch.tensor(
            [tile_to_buy, waiting_for_buy_tile_with_bm, first_bm_to_spend_on_tile],
            device=device,
            dtype=torch.uint8,
        )

    def fill_city_tensor(self, game):
        city_num_attributes = (
            10  # 10 attributes for city - currently tracking 8, 2 are placeholders
        )
        office_num_attributes = (
            8  # 8 attributes for each office - currently tracking 5, 3 are placeholders
        )
        self.max_offices = 10  # Maximum number of offices per city
        all_city_info = torch.zeros(
            MAX_CITIES,
            city_num_attributes + office_num_attributes * self.max_offices,
            device=device,
            dtype=torch.uint8,
        )

        for i, city in enumerate(game.selected_map.cities):
            city_num, city_color = self.assign_city_name_and_color_mapping(game, city)
            city1_upgrade, city2_upgrade = self.assign_city_upgrade_type_mapping(city)
            city_tributes = self.assign_city_tribute_mapping(city)

            city_data = [
                city_num,
                city_color,
                city1_upgrade,
                city2_upgrade,
                city_tributes[0],
                city_tributes[1],
                city_tributes[2],
                city_tributes[3],
            ]
            city_data += [0] * (city_num_attributes - len(city_data))  # Pad with zeros

            office_data = [self.assign_office_mapping(office) for office in city.offices]
            office_data += [(0, 0, 0, 0, 0)] * (
                self.max_offices - len(office_data)
            )  # Pad with zeros if there are fewer than 10 offices

            # Flatten office data and pad with zeros
            office_data_flat = []
            for office in office_data:
                office_flat = list(office)
                office_flat += [0] * (office_num_attributes - len(office_flat))  # Pad with zeros
                office_data_flat += office_flat

            city_tensor = torch.tensor(
                city_data + office_data_flat, dtype=torch.uint8, device=device
            )
            all_city_info[i] = city_tensor

        all_city_info = all_city_info.flatten()
        # print(f"all_city_info Size: {all_city_info.size()}")

        return all_city_info

    def assign_city_name_and_color_mapping(self, game, city):
        city_num = self.MAP_CITY_NAME_MAPPINGS[game.map_num].get(city.name, 0)
        city_color = self.CITY_COLOR_VALUES.index(city.color) + 1
        return city_num, city_color

    def assign_city_upgrade_type_mapping(self, city):
        city1_upgrade = 0
        city2_upgrade = 0
        if len(city.upgrade_city_type) > 0:
            city1_upgrade = self.UPGRADE_TYPE_TO_ID.get(city.upgrade_city_type[0], 0)
        if len(city.upgrade_city_type) > 1:
            city2_upgrade = self.UPGRADE_TYPE_TO_ID.get(city.upgrade_city_type[1], 0)

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
        office_shape = self.OFFICE_SHAPE_MAPPING.get(office.shape, 0)
        office_color = self.OFFICE_COLOR_MAPPING.get(tuple(office.color), 0)
        office_controller = office.controller.order + 1 if office.controller else 0
        office_points = office.awards_points
        office_place_adjacent = 1 if office.place_adjacent_office else 0
        return office_shape, office_color, office_controller, office_points, office_place_adjacent

    def fill_route_tensor(self, game):
        self.route_num_attributes = 10  # 10 attributes for route
        self.post_num_attributes = 5  # 5 attributes for each post
        self.max_posts = 5  # Maximum number of posts per route
        all_route_info = torch.zeros(
            MAX_ROUTES,
            self.route_num_attributes + self.post_num_attributes * self.max_posts,
            device=device,
            dtype=torch.uint8,
        )

        for i, route in enumerate(game.selected_map.routes):
            # Get numerical values for route attributes
            city1_num, _ = self.assign_city_name_and_color_mapping(game, route.cities[0])
            city2_num, _ = self.assign_city_name_and_color_mapping(game, route.cities[1])
            route_region = self.assign_region_mapping(route.region)
            route_has_bm, route_bm_type = self.assign_bonus_marker_mapping(
                route.has_bonus_marker, route.bonus_marker
            )
            route_perm_bm = self.assign_permanent_bm_mapping(route.has_permanent_bm_type)

            tribute_owners = [
                self.assign_player_mapping(game, owner) for owner in route.tribute_owners[:2]
            ]
            tribute_owners += [0] * (2 - len(tribute_owners))
            route_info = [
                city1_num,
                city2_num,
                route.num_posts,
                route_region,
                route_has_bm,
                route_bm_type,
                route_perm_bm,
                tribute_owners[0],
                tribute_owners[1],
                len(route.block_marker_owners),
            ]
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
            all_route_info[i] = torch.tensor(
                route_info + post_info, dtype=torch.uint8, device=device
            )

        all_route_info = all_route_info.flatten()
        # print(f"all_route_info Size: {all_route_info.size()}")
        return all_route_info

    def assign_region_mapping(self, region):
        return self.REGION_MAPPING.get(region, 0)

    def assign_bonus_marker_mapping(self, has_bonus_marker, bonus_marker):
        if has_bonus_marker:
            return has_bonus_marker, self.BONUS_MARKER_TYPE_TO_ID.get(bonus_marker.type, 0)
        return has_bonus_marker, 0

    def assign_permanent_bm_mapping(self, permanent_bm_type):
        return self.PERMANENT_BM_TYPE_TO_ID.get(permanent_bm_type, 0)

    def assign_post_shape_mapping(self, shape):
        return self.POST_SHAPE_MAPPING.get(shape, 0)

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
        player_info = torch.zeros(
            max_players, num_player_attributes, device=device, dtype=torch.uint8
        )

        # Fill in the data for the actual players
        for i, player in enumerate(game.players):
            player_data = [
                player.order,
                player.score,
                player.final_score,
                player.pieces_to_pickup,
                player.pieces_to_place,
                player.keys_index,
                player.keys,
                priv_mappings[i],
                player.book,
                player.actions_index,
                player.actions,
                player.actions_remaining,
                player.bank,
                player.general_stock_squares,
                player.general_stock_circles,
                player.personal_supply_squares,
                player.personal_supply_circles,
                int(player.ending_turn),
                player.brown_priv_count,
                player.blue_priv_count,
            ]

            mission_card_city1, mission_card_city2, mission_card_city3 = (
                self.assign_mission_card_mapping(game, player)
            )
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
        priv_mapping = {"WHITE": 1, "ORANGE": 2, "PINK": 3, "BLACK": 4}
        return priv_mapping.get(player.privilege, 0)

    def assign_bm_mapping(self, player):
        player_unused_bm = [
            self.BONUS_MARKER_TYPE_TO_ID.get(bm.type, 0) for bm in player.bonus_markers
        ]
        player_used_bm = [
            self.BONUS_MARKER_TYPE_TO_ID.get(bm.type, 0) for bm in player.used_bonus_markers
        ]

        player_unused_bm += [0] * (12 - len(player_unused_bm))
        player_used_bm += [0] * (12 - len(player_used_bm))

        return player_unused_bm, player_used_bm

    def assign_mission_card_mapping(self, game, player):
        mission_card_cities = [0, 0, 0]

        if game.use_mission_cards and player is game.current_player and player.mission_card:
            for i, city_name in enumerate(player.mission_card):
                # Find the city object with the matching name
                city = next((c for c in game.selected_map.cities if c.name == city_name), None)
                if city is not None:
                    city_num, _ = self.assign_city_name_and_color_mapping(
                        game, city
                    )  # Ignore the color
                    mission_card_cities[i] = city_num

        return tuple(mission_card_cities)
