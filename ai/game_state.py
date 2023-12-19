import torch
import time
import sys
from game.game_info import Game
from map_data.constants import GREEN, BLUE, PURPLE, RED, YELLOW, BLACKISH_BROWN, DARK_RED, DARK_GREEN, DARK_BLUE, GREY, MAX_CITIES, MAX_ROUTES, COLOR_NAMES

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

def get_game_state(game):
    # game_tensor Size: torch.Size([19])
    # city_tensor Size: torch.Size([1470])
    # route_tensor Size: torch.Size([680])
    # player_tensor Size: torch.Size([185])

    # Game:
    start_time = time.time()
    game_tensor = fill_game_tensor(game)
    game_tensor = game_tensor.flatten()
    end_time = time.time()
    print(f"game_tensor Size: {game_tensor.size()}")
    execution_time1 = end_time - start_time

    # Map info:
    start_time = time.time()
    city_tensor = fill_city_tensor(game)
    city_tensor = city_tensor.flatten()
    end_time = time.time()
    print(f"city_tensor Size: {city_tensor.size()}")
    execution_time2 = end_time - start_time
    
    # Route Info
    start_time = time.time()
    route_tensor = fill_route_tensor(game)
    route_tensor = route_tensor.flatten()
    end_time = time.time()
    print(f"route_tensor Size: {route_tensor.size()}")
    execution_time3 = end_time - start_time
    
    #Player Info
    start_time = time.time()
    player_tensor = fill_player_info_tensor(game)
    player_tensor = player_tensor.flatten()
    end_time = time.time()
    print(f"player_tensor Size: {player_tensor.size()}")
    execution_time4 = end_time - start_time
    
    # print(f"game_tensor Execution Time: {execution_time1} seconds, Size: {game_tensor.size()}")
    # print(f"city_tensor Execution Time: {execution_time2} seconds, Size: {city_tensor.size()}")
    # print(f"route_tensor Execution Time: {execution_time3} seconds, Size: {route_tensor.size()}")
    # print(f"player_tensor Execution Time: {execution_time4} seconds, Size: {player_tensor.size()}")

    flattened_game_state = torch.cat([game_tensor, city_tensor, route_tensor, player_tensor], dim=0)

    # print(f"All Game State Size: {flattened_game_state.size()}")
    return flattened_game_state

def fill_game_tensor(game):
    # Initial game info
    initial_game_info = torch.tensor([game.map_num, game.num_players, game.active_player,
                                      game.current_player_index + 1, game.current_player.actions_remaining,
                                      game.selected_map.max_full_cities, game.current_full_cities_count,
                                      game.east_west_completed_count,
                                      game.waiting_for_bm_swap_office, game.waiting_for_bm_upgrade_ability,
                                      game.waiting_for_bm_move_any_2, game.waiting_for_bm_move3], device=device, dtype=torch.uint8)

    # Privileges info
    cardiff_priv, carlisle_priv, london_priv = assign_blue_brown_priv_mapping(game)
    privileges_info = torch.tensor([cardiff_priv, carlisle_priv, london_priv], device=device, dtype=torch.uint8)
    
    # Special Prestige Points info
    special_prestige_points_info = assign_special_prestige_points_mapping(game)

    # Concatenate all game info into one tensor
    game_info = torch.cat((initial_game_info, privileges_info, special_prestige_points_info), dim=0).unsqueeze(0)  # Add an extra dimension for batch size
    # print(f"game_info {game_info}")
    return game_info

def assign_blue_brown_priv_mapping(game):
    colors = [GREEN, BLUE, PURPLE, RED, YELLOW]

    cardiff_priv = colors.index(game.cardiff_priv) + 1 if game.cardiff_priv else 0
    carlisle_priv = colors.index(game.carlisle_priv) + 1 if game.carlisle_priv else 0
    london_priv = colors.index(game.london_priv) + 1 if game.london_priv else 0

    return cardiff_priv, carlisle_priv, london_priv

def unmap_blue_brown_priv_mapping(cardiff_priv, carlisle_priv, london_priv):
    colors = [GREEN, BLUE, PURPLE, RED, YELLOW]

    cardiff_priv = colors[cardiff_priv - 1] if cardiff_priv else None
    carlisle_priv = colors[carlisle_priv - 1] if carlisle_priv else None
    london_priv = colors[london_priv - 1] if london_priv else None

    return cardiff_priv, carlisle_priv, london_priv

def assign_special_prestige_points_mapping(game):
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

def unmap_special_prestige_points_mapping(special_prestige_points_info, game):
    if len(special_prestige_points_info) != len(game.selected_map.specialprestigepoints.circle_data):
        print(f"Error: Mismatch in the size of special prestige points info and circle data.")
        print(f"special_prestige_points_info {len(special_prestige_points_info)}, circle_data {len(game.selected_map.specialprestigepoints.circle_data)}")
        return  # or handle this situation appropriately
    
    for i, circle in enumerate(game.selected_map.specialprestigepoints.circle_data):
        if special_prestige_points_info[i] != 0:
            # Find the player object based on the index
            owner_player = game.players[special_prestige_points_info[i] - 1]  # -1 to handle zero-based indexing
            circle["owner"] = owner_player
            circle["color"] = owner_player.color
        else:
            circle["owner"] = None

def fill_city_tensor(game):
    num_attributes = 49  # 4 attributes for city + 10 offices * 4 attributes each + 5 adjacent city numbers
    all_city_info = torch.zeros(MAX_CITIES, num_attributes, device=device, dtype=torch.uint8)

    for i, city in enumerate(game.selected_map.cities):
        city_num, city_color = assign_city_name_and_color_mapping(game, city)
        city1_upgrade, city2_upgrade = assign_city_upgrade_type_mapping(city)

        city_data = [city_num, city_color, city1_upgrade, city2_upgrade]
        office_data = [assign_office_mapping(office) for office in city.offices]
        # Pad with zeros if there are fewer than 10 offices
        office_data += [(0, 0, 0, 0)] * (10 - len(office_data))
        
        # Flatten office data
        office_data_flat = [attribute for office in office_data for attribute in office]

        # Get adjacent city numbers
        adjacent_city_numbers = [assign_city_name_and_color_mapping(game, adjacent_city)[0] for adjacent_route in city.routes for adjacent_city in adjacent_route.cities if adjacent_city != city]
        # Pad with zeros if there are fewer than 5 routes
        adjacent_city_numbers += [0] * (5 - len(adjacent_city_numbers))

        # Combine data into a single tensor
        city_tensor = torch.tensor(city_data + office_data_flat + adjacent_city_numbers, dtype=torch.uint8, device=device)
        all_city_info[i] = city_tensor

    return all_city_info

def unmap_city_tensor(city_tensor, game, city):
    # Split the tensor into its parts
    city_data_list = list(map(int, city_tensor))
    city_data = city_data_list[:4]
    office_data = city_data_list[4:44]
    adjacent_city_numbers = city_data_list[44:49]

    # Unpack the office data into a list of tuples
    office_data = [tuple(office_data[i:i + 4]) for i in range(0, len(office_data), 4)]

    # Convert city data to their original values
    city_name = unmap_city_name_and_color_mapping(city_data[0], game)

    # Convert office data to their original values
    office_data = [unmap_office_mapping(office, game) for office in office_data]

    if city.name == city_name:
        # Ensure the number of offices in the tensor matches the number of offices in the city
        for office, new_office_data in zip(city.offices, office_data):
            office_shape, office_color, office_controller, office_points = new_office_data
            # office.shape = office_shape
            # office.color = office_color
            if office_controller:
                office.controller = office_controller  
                office.color = office.controller.color
            else:
                office.controller = None
            # office.awards_points = office_points
    else:
        sys.exit(f"City name: {city.name} does not match city name in tensor: {city_name}")

def assign_city_name_and_color_mapping(game, city):
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

def unmap_city_name_and_color_mapping(city_num, game):
    map1_cities_mapping = {
            1: 'Groningen',
            2: 'Emden',
            3: 'Osnabruck',
            4: 'Kampen',
            5: 'Arnheim',
            6: 'Duisburg',
            7: 'Dortmund',
            8: 'Munster',
            9: 'Coellen',
            10: 'Warburg',
            11: 'Paderborn',
            12: 'Minden',
            13: 'Bremen',
            14: 'Stade',
            15: 'Hannover',
            16: 'Hildesheim',
            17: 'Gottingen',
            18: 'Quedlinburg',
            19: 'Goslar',
            20: 'Brunswick',
            21: 'Luneburg',
            22: 'Hamburg',
            23: 'Lubeck',
            24: 'Perleberg',
            25: 'Stendal',
            26: 'Magdeburg',
            27: 'Halle'
        }
    map2_cities_mapping = {
            1: 'Lubeck',
            2: 'Mismar',
            3: 'Stralsund',
            4: 'Malmo',
            5: 'Visby',
            6: 'Danzig',
            7: 'Konigsberg',
            8: 'Belgard',
            9: 'Anklam',
            10: 'Waren',
            11: 'Perleberg',
            12: 'Havelberg',
            13: 'Stettin',
            14: 'Kulm',
            15: 'Elbing',
            16: 'Braunsberg',
            17: 'Allenstein',
            18: 'Frankfurt',
            19: 'BerlinColln',
            20: 'Brandenburg',
            21: 'Tangermunde',
            22: 'Magdeburg',
            23: 'Halle',
            24: 'Mittenberg',
            25: 'Dresden',
            26: 'Breslau',
            27: 'Thorn',
            28: 'Krackau'
    }
    map3_cities_mapping = {
            1: 'Glasgom',
            2: 'Edinbaurgh',
            3: 'Dunbar',
            4: 'Falkirk',
            5: 'Carlisle',
            6: 'Newcastle',
            7: 'IsleOfMan',
            8: 'Conway',
            9: 'Chester',
            10: 'Montgomery',
            11: 'Pembroke',
            12: 'Cardiff',
            13: 'Richmond',
            14: 'Durham',
            15: 'Lancaster',
            16: 'York',
            17: 'Hereford',
            18: 'Coventry',
            19: 'Nottingham',
            20: 'Norwich',
            21: 'Cambridge',
            22: 'Ipswich',
            23: 'Oxford',
            24: 'London',
            25: 'Canterbury',
            26: 'Calais',
            27: 'Southhampton',
            28: 'Salisbury',
            29: 'Plymouth',
            30: 'Bristol'
        }

    if game.map_num == 1:
        city_name = map1_cities_mapping.get(city_num, 0)
    elif game.map_num == 2:
        city_name = map2_cities_mapping.get(city_num, 0)
    elif game.map_num == 3:
        city_name = map3_cities_mapping.get(city_num, 0)

    return city_name

def assign_city_upgrade_type_mapping(city):
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

def assign_office_mapping(office):
    colors = [GREEN, BLUE, PURPLE, RED, YELLOW]
    office_shape_mapping = {
            'square': 1,
            'circle': 2
        }
    office_color_mapping = {
            'WHITE': 1,
            'ORANGE': 2,
            'PINK': 3,
            'BLACK': 4
        }
    office_shape = office_shape_mapping.get(office.shape, 0)
    office_color = office_color_mapping.get(office.color, 0)
    office_controller = colors.index(office.controller.color) + 1 if office.controller else 0
    office_points = office.awards_points
    return office_shape, office_color, office_controller, office_points

def unmap_office_mapping(office, game):
    if len(office) != 4:
        print(f"Error: Office data does not have 4 attributes.")
        return None, None, None, 0

    office_shape_mapping = {
        0: None,
        1: 'square',
        2: 'circle'
    }
    office_color_mapping = {
        0: None,
        1: 'WHITE',
        2: 'ORANGE',
        3: 'PINK',
        4: 'BLACK'
    }
    
    office_shape = office_shape_mapping.get(office[0], None)
    office_color = office_color_mapping.get(office[1], None)

    # Convert the player index to the corresponding player object
    office_controller_index = office[2]
    office_controller = game.players[office_controller_index - 1] if office_controller_index else None

    office_points = office[3]
    return office_shape, office_color, office_controller, office_points

def fill_route_tensor(game):
    num_attributes = 7 + 5 * 2  # 6 route attributes + 2 post attributes * 5 posts
    all_route_info = torch.zeros(MAX_ROUTES, num_attributes, device=device, dtype=torch.uint8)

    for i, route in enumerate(game.selected_map.routes):
        # Get numerical values for route attributes
        city1_num, _ = assign_city_name_and_color_mapping(game, route.cities[0])
        city2_num, _ = assign_city_name_and_color_mapping(game, route.cities[1])
        route_region = assign_region_mapping(route.region)
        route_has_bm, route_bm_type = assign_bonus_marker_mapping(route.has_bonus_marker, route.bonus_marker)
        route_perm_bm = assign_permanent_bm_mapping(route.has_permanent_bm_type)

        # print(f"city1_num {city1_num}, city2_num {city2_num}, route_has_bm {route_has_bm}, route_bm_type {route_bm_type}")
        route_info = [city1_num, city2_num, route.num_posts, route_region, route_has_bm, route_bm_type, route_perm_bm]

        # Get post information
        post_info = []
        for post in route.posts:
            owner_shape = assign_post_shape_mapping(post.owner_piece_shape)
            post_owner = assign_player_mapping(game, post.owner) 
            post_info.extend([owner_shape, post_owner])

        # Pad post_info if fewer than 5 posts
        post_info += [0] * ((5 - len(route.posts)) * 2)

        # Combine data into a single tensor
        all_route_info[i] = torch.tensor(route_info + post_info, dtype=torch.uint8, device=device)

    return all_route_info

def unmap_route_tensor(route_tensor, route, game):
    # Ensure route_tensor is a list of integers
    route_info_list = list(map(int, route_tensor))

    # Split the tensor into route and post information
    route_info = route_info_list[:7]
    post_info = route_info_list[7:]

    # Convert route data to original values
    city1_name = unmap_city_name_and_color_mapping(route_info[0], game)
    city2_name = unmap_city_name_and_color_mapping(route_info[1], game)
    num_posts = route_info[2]  # Number of posts on the route
    route_region = unmap_region_mapping(route_info[3])
    route_has_bm = route_info[4]
    route_bm_type = unmap_bonus_marker_mapping(route_info[5])
    route_perm_bm = unmap_permanent_bm_mapping(route_info[6])

    print(f"city1_name {city1_name}, city2_name {city2_name}, num_posts {num_posts}, route_has_bm {route_has_bm}, route_bm_type {route_bm_type}")

    # Convert post data to original values
    post_info = [tuple(post_info[i:i + 2]) for i in range(0, len(post_info), 2)]
    owner_shapes = [unmap_post_shape_mapping(post[0]) for post in post_info]
    post_owners = [unmap_player_mapping(post[1], game) for post in post_info]

    # Verify and update route attributes
    if route.cities[0].name == city1_name and route.cities[1].name == city2_name and \
       route.region == route_region and len(route.posts) == num_posts:
        if route_bm_type:
            route.assign_map_new_bonus_marker(route_bm_type)
        else:
            route.has_bonus_marker = False
            route.bonus_marker = None
        route.has_permanent_bm_type = route_perm_bm
        for i, post in enumerate(route.posts):
            if post_owners[i] is not None:
                post.claim(post_owners[i], owner_shapes[i])
    else:
        print(f"Error: Route data mismatch for route between {city1_name} and {city2_name}")

def assign_region_mapping(region):
    region_mapping = {
        'Scotland': 1,
        'Wales': 2,
        None: 0  # No specific region
    }
    return region_mapping.get(region, 0)

def unmap_region_mapping(region):
    region_mapping = {
        1: 'Scotland',
        2: 'Wales',
        0: None  # No specific region
    }
    return region_mapping.get(region, None)

def assign_bonus_marker_mapping(has_bonus_marker, bonus_marker):
    bm_type = 0
    bm_mapping = {
        'PlaceAdjacent': 1,
        'SwapOffice': 2,
        'Move3': 3,
        'UpgradeAbility': 4,
        '3Actions': 5,
        '4Actions': 6,
        'Exchange Bonus Marker': 7,
        'Tribute for Establish a Trading Post': 8,
        'Block Trade Route': 9
    }
    if has_bonus_marker:
        bm_type = bm_mapping.get(bonus_marker.type, 0)

    return has_bonus_marker, bm_type

def unmap_bonus_marker_mapping(bonus_marker_type):
    bm_mapping = {
        1: 'PlaceAdjacent',
        2: 'SwapOffice',
        3: 'Move3',
        4: 'UpgradeAbility',
        5: '3Actions',
        6: '4Actions',
        7: 'Exchange Bonus Marker',
        8: 'Tribute for Establish a Trading Post',
        9: 'Block Trade Route',
        0: None
    }
    bm_type = bm_mapping.get(bonus_marker_type, None)

    return bm_type    

def assign_permanent_bm_mapping(permanent_bm_type):
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

def unmap_permanent_bm_mapping(permanent_bm_type):
    # Example mapping, modify based on your game's permanent bonus markers
    permanent_bm_mapping = {
        1: 'MoveAny2',
        2: '+1Priv',
        3: 'ClaimGreenCity',
        4: 'Place2TradesmenFromRoute',
        5: 'Place2ScotlandOrWales',
        0: None
    }
    return permanent_bm_mapping.get(permanent_bm_type, None)

def assign_post_shape_mapping(shape):
    shape_mapping = {
        'circle': 1,
        'square': 2,
        None: 0
    }
    return shape_mapping.get(shape, 0)

def unmap_post_shape_mapping(shape):
    shape_mapping = {
        1: 'circle',
        2: 'square',
        0: None
    }
    return shape_mapping.get(shape, None)

def assign_player_mapping(game, player):
    # Map player objects to a unique identifier (e.g., player index)
    if player is None:
        return 0
    else:
        return game.players.index(player) + 1  # Assuming game.players is a list of players

def unmap_player_mapping(player_index, game):
    # Map unique identifier (e.g., player index) to player object
    if player_index == 0:
        return None
    else:
        return game.players[player_index - 1]  # Assuming game.players is a list of players

def fill_player_info_tensor(game):
    max_players = 5 
    num_attributes = 37

    # Precompute priv and bank mappings for all players
    priv_mappings = [assign_priv_mapping(player) for player in game.players]
    bank_mappings = [assign_bank_mapping(player) for player in game.players]
    bm_mappings = [assign_bm_mapping(player) for player in game.players]
    player_unused_bm, player_used_bm = zip(*bm_mappings)  # Unpack the tuples into two lists

    # Initialize the tensor on the appropriate device
    player_info = torch.zeros(max_players, num_attributes, device=device, dtype=torch.uint8)

    # Fill in the data for the actual players
    for i, player in enumerate(game.players):
        player_data = [player.score, player.keys, priv_mappings[i], player.book, player.actions_index, player.actions, bank_mappings[i],
                       player.general_stock_squares, player.general_stock_circles, 
                       player.personal_supply_squares, player.personal_supply_circles, 
                       player.brown_priv_count, player.blue_priv_count]

        # Combine all info into a single tensor
        player_info[i] = torch.tensor(player_data + list(player_unused_bm[i]) + list(player_used_bm[i]), dtype=torch.uint8, device=device)

    return player_info

def unmap_player_tensor(player_tensor, player):
    # Convert player_tensor to a list of integers
    player_info_list = list(map(int, player_tensor))

    # Split the tensor into player data and bonus marker data
    player_data = player_info_list[:14]
    player_unused_bm = player_info_list[14:26]
    player_used_bm = player_info_list[26:]

    # Convert player data to their original values
    player.score = player_data[0]
    player.keys = player_data[1]
    player.privilege = unmap_priv_mapping(player_data[2])
    player.book = player_data[3]
    player.actions_index = player_data[4]
    player.actions = player_data[5]
    player.bank = unmap_bank_mapping(player_data[6])
    player.general_stock_squares = player_data[7]
    player.general_stock_circles = player_data[8]
    player.personal_supply_squares = player_data[9]
    player.personal_supply_circles = player_data[10]
    player.brown_priv_count = player_data[11]
    player.blue_priv_count = player_data[12]

    # Convert bonus markers to their original values
    player_unused_bm = [unmap_bm_mapping(bm) for bm in player_unused_bm]
    player_used_bm = [unmap_bm_mapping(bm) for bm in player_used_bm]

    # Update player's bonus markers
    player.bonus_markers = [bm for bm in player_unused_bm if bm is not None]
    player.used_bonus_markers = [bm for bm in player_used_bm if bm is not None]

def assign_priv_mapping(player):
    priv_mapping = {
            'WHITE': 1,
            'ORANGE': 2,
            'PINK': 3,
            'BLACK': 4
        }
    return priv_mapping.get(player.privilege, 0)

def unmap_priv_mapping(priv):
    priv_mapping = {
            1: 'WHITE',
            2: 'ORANGE',
            3: 'PINK',
            4: 'BLACK'
        }
    return priv_mapping.get(priv, None)

def assign_bank_mapping(player):
    bank_mapping = {
        3: 1,
        5: 2,
        7: 3,
        50: 4
    }
    return bank_mapping.get(player.bank, 0)

def unmap_bank_mapping(bank):
    bank_mapping = {
            1: 3,
            2: 5,
            3: 7,
            4: 50
        }
    return bank_mapping.get(bank, None)

def assign_bm_mapping(player):
    bm_mapping = {
            'PlaceAdjacent': 1,
            'SwapOffice': 2,
            'Move3': 3,
            'UpgradeAbility': 4,
            '3Actions': 5,
            '4Actions': 6,
            'Exchange Bonus Marker': 7,
            'Tribute for Establish a Trading Post': 8,
            'Block Trade Route': 9
        }

    # Convert bonus markers to their numerical representations
    player_unused_bm = [bm_mapping.get(bm.type, 0) for bm in player.bonus_markers]
    player_used_bm = [bm_mapping.get(bm.type, 0) for bm in player.used_bonus_markers]

    # Padding if necessary
    player_unused_bm += [0] * (12 - len(player_unused_bm))
    player_used_bm += [0] * (12 - len(player_used_bm))

    return player_unused_bm, player_used_bm

def unmap_bm_mapping(bm):
    bm_mapping = {
            1: 'PlaceAdjacent',
            2: 'SwapOffice',
            3: 'Move3',
            4: 'UpgradeAbility',
            5: '3Actions',
            6: '4Actions',
            7: 'Exchange Bonus Marker',
            8: 'Tribute for Establish a Trading Post',
            9: 'Block Trade Route'
        }
    return bm_mapping.get(bm, None)

def save_game_state_to_file(game):
    game_state = get_game_state(game)
    with open('game_states_for_training.txt', 'a') as f:
        # Split the game state tensor into its parts
        game_tensor = game_state[:19]
        city_tensor = game_state[19:19+1470]
        route_tensor = game_state[19+1470:19+1470+680]
        player_tensor = game_state[19+1470+680:]

        # Convert each tensor to a list and join the elements into a string
        game_tensor_str = ', '.join(map(str, game_tensor.flatten().tolist()))
        city_tensor_str = ', '.join(map(str, city_tensor.flatten().tolist()))
        route_tensor_str = ', '.join(map(str, route_tensor.flatten().tolist()))
        player_tensor_str = ', '.join(map(str, player_tensor.flatten().tolist()))

        # game_tensor Size: torch.Size([19])
        # city_tensor Size: torch.Size([1470])
        # route_tensor Size: torch.Size([680])
        # player_tensor Size: torch.Size([185])

        # Print each string to the file
        print(f"Game Tensor: {game_tensor_str}", file=f)
        print(f"City Tensor: {city_tensor_str}", file=f)
        print(f"Route Tensor: {route_tensor_str}", file=f)
        print(f"Player Tensor: {player_tensor_str}", file=f)

def load_game_from_file(filename):
    game = get_game_info(filename)
    get_player_info(filename, game)
    get_city_info(filename, game)
    get_route_info(filename, game)

    return game

def get_game_info(filename):
    open_file = open(filename, "r")
    lines = open_file.readlines()
    for line in lines:
        if line.startswith("Game Tensor:"):
            game_str = line.split("Game Tensor: ")[1]
            game_list = game_str.split(", ")
            int_game_list = list(map(int, game_list))

            map_num = int_game_list[0]
            print(f"Map Num: {map_num}")
            num_players = int_game_list[1]
            print(f"Num Players: {num_players}")
            active_player = int_game_list[2]
            current_player_index = int_game_list[3]
            actions_remaining = int_game_list[4]
            # max_full_cities = int_game_list[5]
            current_full_cities_count = int_game_list[6]
            east_west_completed_count = int_game_list[7]
            waiting_for_bm_swap_office = int_game_list[8]
            waiting_for_bm_upgrade_ability = int_game_list[9]
            waiting_for_bm_move_any_2 = int_game_list[10]
            waiting_for_bm_move3 = int_game_list[11]
            cardiff_priv = int_game_list[12]
            carlisle_priv = int_game_list[13]
            london_priv = int_game_list[14]
            special_prestige_points = int_game_list[15:19]

            break
    
    game = Game(map_num, num_players)
    game.active_player = active_player
    game.current_player_index = current_player_index - 1
    game.current_player = game.players[current_player_index - 1]

    game.current_player.actions_remaining = actions_remaining
    game.current_full_cities_count = current_full_cities_count
    game.east_west_completed_count = east_west_completed_count
    game.waiting_for_bm_swap_office = waiting_for_bm_swap_office
    game.waiting_for_bm_upgrade_ability = waiting_for_bm_upgrade_ability
    game.waiting_for_bm_move_any_2 = waiting_for_bm_move_any_2
    game.waiting_for_bm_move3 = waiting_for_bm_move3
    game.cardiff_priv, game.carlisle_priv, game.london_priv = unmap_blue_brown_priv_mapping(cardiff_priv, carlisle_priv, london_priv)
    unmap_special_prestige_points_mapping(special_prestige_points, game)
    
    return game

def get_player_info(filename, game):
    colors = [GREEN, BLUE, PURPLE, RED, YELLOW]

    with open(filename, "r") as open_file:
        lines = open_file.readlines()

    for line in lines:
        if line.startswith("Player Tensor:"):
            player_str = line.split("Player Tensor: ")[1].strip()
            player_data_list = player_str.split(", ")
            num_attributes_per_player = 37  # number of attributes for each player

            for i, player in enumerate(game.players):
                start_index = i * num_attributes_per_player
                end_index = start_index + num_attributes_per_player
                player_tensor = player_data_list[start_index:end_index]
                player.color = colors[i]
                unmap_player_tensor(player_tensor, player)

def get_city_info(filename, game):
    with open(filename, "r") as open_file:
        lines = open_file.readlines()

    for line in lines:
        if line.startswith("City Tensor:"):
            city_str = line.split("City Tensor: ")[1].strip()
            city_data_list = city_str.split(", ")
            num_attributes_per_city = 49  # number of attributes for each city

            for i, city in enumerate(game.selected_map.cities):
                start_index = i * num_attributes_per_city
                end_index = start_index + num_attributes_per_city
                city_tensor = city_data_list[start_index:end_index]
                unmap_city_tensor(city_tensor, game, city)

def get_route_info(filename, game):
    with open(filename, "r") as open_file:
        lines = open_file.readlines()

    for line in lines:
        if line.startswith("Route Tensor:"):
            route_str = line.split("Route Tensor: ")[1].strip()
            route_data_list = route_str.split(", ")
            num_attributes_per_route = 17  # number of attributes for each route

            for i, route in enumerate(game.selected_map.routes):
                start_index = i * num_attributes_per_route
                end_index = start_index + num_attributes_per_route
                route_tensor = route_data_list[start_index:end_index]
                unmap_route_tensor(route_tensor, route, game)
