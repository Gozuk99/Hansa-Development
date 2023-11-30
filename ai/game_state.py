import torch
import time
from map_data.constants import GREEN, BLUE, PURPLE, RED, YELLOW, BLACKISH_BROWN, DARK_RED, DARK_GREEN, DARK_BLUE, GREY

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

def get_game_state(game):
    # Type of Action:
    # normal, displaced, use_bm, replace_bm, perm_bonus_marker
    # 1-5, 1/0, 1/0, 1/0

    # Game:
    start_time = time.time()
    game_tensor = fill_game_tensor(game)
    end_time = time.time()
    execution_time1 = end_time - start_time

    # Map info:
    start_time = time.time()
    city_tensor = fill_city_tensor(game)
    end_time = time.time()
    execution_time2 = end_time - start_time
    
    # Route Info
    start_time = time.time()
    route_tensor = fill_route_tensor(game)
    end_time = time.time()
    execution_time3 = end_time - start_time
    
    #Player Info
    start_time = time.time()
    player_tensor = fill_player_info_tensor(game)
    end_time = time.time()
    execution_time4 = end_time - start_time
    
    print(f"game_tensor Execution Time: {execution_time1} seconds, Size: {game_tensor.size()}")
    print(f"city_tensor Execution Time: {execution_time2} seconds, Size: {city_tensor.size()}")
    print(f"route_tensor Execution Time: {execution_time3} seconds, Size: {route_tensor.size()}")
    print(f"player_tensor Execution Time: {execution_time4} seconds, Size: {player_tensor.size()}")

    flattened_game_state = torch.cat([game_tensor.flatten(), city_tensor.flatten(), route_tensor.flatten(), player_tensor.flatten()], dim=0)

    print(f"All Game State Size: {flattened_game_state.size()}")
    return flattened_game_state

def fill_game_tensor(game):
    # Initial game info
    initial_game_info = torch.tensor([game.map_num, game.num_players, 
                                      game.current_player_index + 1, game.current_player.actions_remaining,
                                      game.selected_map.max_full_cities, game.current_full_cities_count,
                                      game.east_west_completed_count], device=device, dtype=torch.uint8)

    # Privileges info
    cardiff_priv, carlisle_priv, london_priv = assign_blue_brown_priv_mapping(game)
    privileges_info = torch.tensor([cardiff_priv, carlisle_priv, london_priv], device=device, dtype=torch.uint8)
    
    # Special Prestige Points info
    special_prestige_points_info = assign_special_prestige_points_mapping(game)

    # Concatenate all game info into one tensor
    game_info = torch.cat((initial_game_info, privileges_info, special_prestige_points_info), dim=0).unsqueeze(0)  # Add an extra dimension for batch size
    print(f"game_info {game_info}")
    return game_info

def assign_blue_brown_priv_mapping(game):
    colors = [GREEN, BLUE, PURPLE, RED, YELLOW]

    cardiff_priv = colors.index(game.cardiff_priv) + 1 if game.cardiff_priv else 0
    carlisle_priv = colors.index(game.carlisle_priv) + 1 if game.carlisle_priv else 0
    london_priv = colors.index(game.london_priv) + 1 if game.london_priv else 0

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

def fill_city_tensor(game):
    max_cities = 30
    num_attributes = 49  # 4 attributes for city + 10 offices * 4 attributes each + 5 adjacent city numbers
    all_city_info = torch.zeros(max_cities, num_attributes, device=device, dtype=torch.uint8)

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
            'Munster': 8,
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
    office_controller = colors.index(office.controller) + 1 if office.controller else 0
    office_points = office.awards_points
    return office_shape, office_color, office_controller, office_points

def fill_route_tensor(game):
    max_routes = 40
    num_attributes = 6 + 5 * 2  # 6 route attributes + 2 post attributes * 5 posts
    all_route_info = torch.zeros(max_routes, num_attributes, device=device, dtype=torch.uint8)

    for i, route in enumerate(game.selected_map.routes):
        # Get numerical values for route attributes
        city1_num, _ = assign_city_name_and_color_mapping(game, route.cities[0])
        city2_num, _ = assign_city_name_and_color_mapping(game, route.cities[1])
        route_region = assign_region_mapping(route.region)
        route_bm = assign_bonus_marker_mapping(route.has_bonus_marker)
        route_perm_bm = assign_permanent_bm_mapping(route.has_permanent_bm_type)

        route_info = [city1_num, city2_num, route.num_posts, route_region, route_bm, route_perm_bm]

        # Get post information
        post_info = []
        for post in route.posts:
            post_shape = assign_post_shape_mapping(post.required_shape)
            post_owner = assign_player_mapping(game, post.owner)  # You need to create this function
            post_info.extend([post_shape, post_owner])

        # Pad post_info if fewer than 5 posts
        post_info += [0] * ((5 - len(route.posts)) * 2)

        # Combine data into a single tensor
        all_route_info[i] = torch.tensor(route_info + post_info, dtype=torch.uint8, device=device)

    return all_route_info

def assign_region_mapping(region):
    region_mapping = {
        'Scotland': 1,
        'Wales': 2,
        None: 0  # No specific region
    }
    return region_mapping.get(region, 0)

def assign_bonus_marker_mapping(has_bonus_marker):
    return 1 if has_bonus_marker else 0

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

def assign_post_shape_mapping(shape):
    shape_mapping = {
        'circle': 1,
        'square': 2,
        None: 0
    }
    return shape_mapping.get(shape, 0)

def assign_player_mapping(game, player):
    # Map player objects to a unique identifier (e.g., player index)
    if player is None:
        return 0
    else:
        return game.players.index(player) + 1  # Assuming game.players is a list of players

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
        # print (f"player_info[{i}] {player_info[i]}")
    
    return player_info

def assign_priv_mapping(player):
    priv_mapping = {
            'WHITE': 1,
            'ORANGE': 2,
            'PINK': 3,
            'BLACK': 4
        }
    return priv_mapping.get(player.privilege, 0)

def assign_bank_mapping(player):
    bank_mapping = {
            '3': 1,
            '5': 2,
            '7': 3,
            'E': 4
        }
    return bank_mapping.get(player.bank, 0)

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

def get_available_actions(game):
    player = game.current_player
    actions = []

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

    for route in game.selected_map.routes:
        if route.is_controlled_by(player):
            if can_player_claim_office_in_city(game, player, route, route.cities[0]):
                print(f"Can claim {route.cities[0].name} with route")
            if can_player_claim_office_in_city(game, player, route, route.cities[1]):
                print(f"Can claim {route.cities[1].name} with route")
            print(f"Can claim route between {route.cities[0].name} and {route.cities[1].name} for points")

        for post in route.posts:
            if check_brown_blue_priv(game, route):
                if player.personal_supply_squares > 0:
                    if post.can_be_claimed_by("square"):
                        # print(f"Can claim post between {route.cities[0].name} and {route.cities[1].name} with square.")
                        actions.append(post)
                if player.personal_supply_circles > 0:
                    if post.can_be_claimed_by("circle"):
                        # print(f"Can claim post between {route.cities[0].name} and {route.cities[1].name} with circle.")
                        actions.append(post)
    if player.bonus_markers:
        for bm in player.bonus_markers:
            actions.append(bm)

def can_player_claim_office_in_city(game, player, route, city):
    next_open_office_color = city.get_next_open_office_color()

    if player.player_can_claim_office(next_open_office_color) and city.color != DARK_GREEN:
        if city.has_required_piece_shape(player, route):
            placed_piece_shape = city.get_next_open_office_shape()
            print(f"Can place a {placed_piece_shape.upper()} into an office of {city.name}")
    elif 'PlaceAdjacent' in (bm.type for bm in player.bonus_markers):
        print(f"Can place a square into a NEW office of {city.name}.")

    if "SpecialPrestigePoints" in city.upgrade_city_type and route.contains_a_circle():
        if game.selected_map.specialprestigepoints.claim_highest_prestige(player):
            print(f"Can claim route for Special Prestige at {city.name}")
    elif any(upgrade_type in ["Keys", "Privilege", "Book", "Actions", "Bank"] for upgrade_type in city.upgrade_city_type):
        for upgrade in game.selected_map.upgrade_cities:
            if upgrade.city_name == city.name:
                print(f"Can claim route to UPGRADE {city.upgrade_city_type} in {city.name}.")

def check_brown_blue_priv(game, route):
    if route.region is not None:
        # Check for Wales region
        if route.region == "Wales":
            if not (game.cardiff_priv == game.current_player or game.london_priv == game.current_player):
                # print("Cannot claim post in BROWN - Incorrect Privilege")
                return False
            if game.current_player.brown_priv_count == 0:
                # print("Used all privilege already in Brown!")
                return False

        # Check for Scotland region
        elif route.region == "Scotland":
            if not (game.carlisle_priv == game.current_player or game.london_priv == game.current_player):
                # print("Cannot claim post in BLUE - Incorrect Privilege")
                return False
            if game.current_player.blue_priv_count == 0:
                # print("Used all privilege already in Blue!")
                return False
    return True