import torch
import time
from map_data.constants import GREEN, BLUE, PURPLE, RED, YELLOW, BLACKISH_BROWN, DARK_RED, DARK_GREEN, DARK_BLUE, GREY, COLOR_NAMES

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#create action space
#1 INCOME - 5 valid options: 0-4 circles, and squares leftover - mask out options based on circles
#2 CLAIM_POST - all posts claim with circle, or square - mask out options with incorrect region(blue or brown priv), incorrect personal_supply, occupied, incorrect shape
#3 DISPLACE - all posts owned by opponents - mask out if invalid personal supply,
#4 MOVE - query all posts -> add least desirable owned piece to array, loop for player.book -> when met, loop for player.book -> ONLY VALID ACTIONS are CLAIM, same rules.
#5 CLAIM_ROUTE - query all routes owned by player - add claim city office, upgrade X, claim route for points
#6 BM - ?
#7 PERM BM - Use immediately
#8 REPLACE BM - Choose a route?
#9 PICK UP - all posts owned by me, mask out not owned, add to array.
#10 END TURN - use BM or End Turn

# output = HansaNN(game_state_tensor)
# valid_actions = mask_out_invalid_actions(output)

def masking_out_invalid_actions(game, game_state):
    
    claim_post_tensor = mask_post_action(game) #size 242 (claiming with a square or circle))
    claim_route_tensor = mask_claim_route(game) #size 380 (claim for points, office, or upgrade)
    income_tensor = mask_income_actions(game) #size 5 (0-4 circles + leftover squares)
    bonus_marker_tensor = mask_bm(game) #size 8 (8 total BM types to use)
    permanent_bonus_marker_tensor = mask_perm_bm(game) #MAYBE remove?
    replace_bm_tensor = mask_replace_bm(game) #size 40
    end_turn_tensor = mask_end_turn(game) #size 1 (allowed to end turn if no bonus markers to replace)

    # Concatenate all tensors into one big tensor representing all possible actions
    all_actions_tensor = torch.cat([income_tensor, claim_post_tensor, 
                                    claim_route_tensor, bonus_marker_tensor, permanent_bonus_marker_tensor,
                                    replace_bm_tensor, end_turn_tensor], dim=0)
    
    return all_actions_tensor

def mask_post_action(game):
    current_player = game.current_player

    post_tensor = torch.zeros(121 * 2, device=device, dtype=torch.uint8)  # 121 max posts * 2 for squares and circles
    can_displace = False

    if current_player.personal_supply_squares + current_player.personal_supply_circles > 1:
        can_displace = True

    post_idx = 0
    for route in game.selected_map.routes:
        for post in route.posts:  # Make sure to iterate over posts in each route
            # Common checks for valid region transition and post not being owned
            is_post_owned = post.is_owned()
            is_post_empty = not is_post_owned

            # CLAIM AS DISPLACED PLAYER - if post is empty
            if game.waiting_for_displaced_player:
                displaced_player = game.displaced_player
                if post in game.all_empty_posts:
                    must_use_displaced_piece = (not displaced_player.played_displaced_shape) and (displaced_player.total_pieces_to_place == 1)
                    can_place_displaced_piece = displaced_player.holding_pieces > 0 and (not post.required_shape or post.required_shape == displaced_player.displaced_shape)

                    if must_use_displaced_piece:
                        if post.required_shape == displaced_player.displaced_shape or post.required_shape is None:
                            post_tensor[post_idx] = 1  # Offset by 121 for circle posts if necessary
                    elif can_place_displaced_piece:
                        if displaced_player.has_general_stock("square"):
                            post_tensor[post_idx] = 1
                        if displaced_player.has_general_stock("circle"):
                            post_tensor[121 + post_idx] = 1  # Offset by 121 for circle posts
            #handle BM Move any2 or #handle BM Move 3:
            elif is_post_owned and ((game.waiting_for_bm_move_any_2) or (game.waiting_for_bm_move3 and post.owner != current_player)):
                if len(current_player.holding_pieces) < current_player.pieces_to_place:
                    post_tensor[post_idx] = 1
            else:
                valid_region = game.current_player.is_valid_region_transition(post.region)

                # Claim post action: check if the post is empty and region is valid
                if len(current_player.holding_pieces) < current_player.pieces_to_place and is_post_empty and valid_region:
                    if 'square' in current_player.holding_pieces and (not post.required_shape or post.required_shape == 'square'):
                        post_tensor[post_idx] = 1
                    if 'circle' in current_player.holding_pieces and (not post.required_shape or post.required_shape == 'circle'):
                        post_tensor[121 + post_idx] = 1  # Offset for circle posts

                # MOVE - if post is owned by current_player:
                elif is_post_owned and post.owner == current_player:
                    if len(current_player.holding_pieces) < current_player.book:
                        post_tensor[post_idx] = 1

                elif is_post_empty and check_brown_blue_priv(game, route):
                    if current_player.personal_supply_squares > 0 and (not post.required_shape or post.required_shape == "square"):
                        post_tensor[post_idx] = 1
                    if current_player.personal_supply_circles > 0 and (not post.required_shape or post.required_shape == "circle"):
                        post_tensor[121 + post_idx] = 1  # Offset by 121 for circle posts
                # DISPLACE - if post is owned by a different player:
                elif is_post_owned and post.owner != current_player and check_brown_blue_priv(game, route) and can_displace:
                    # Calculate the cost to displace based on the shape of the post's piece
                    displacement_cost = 2 if post.owner_piece_shape == "square" else 3
                    
                    # Check if the player has enough pieces to displace
                    if current_player.personal_supply_squares + current_player.personal_supply_circles >= displacement_cost:
                        # If the player can displace this post, mark it as a valid action
                        if post.required_shape == "square" or post.required_shape is None:
                            post_tensor[post_idx] = 1
                        if post.required_shape == "circle" or post.required_shape is None:
                            post_tensor[121 + post_idx] = 1  # Offset by 121 for circle posts
                else:
                    print (f"Invalid scenario detected. Post Info - Color: {COLOR_NAMES[post.color]}, Owner: {post.owner}, Region: {post.region}, ReqShape: {post.required_shape}")
            post_idx += 1

    return post_tensor

#should return - 40+150+90
def mask_claim_route(game):
    max_num_routes = 40  # Maximum number of routes
    max_num_cities = 30  # Maximum number of cities
    max_routes_per_city = 5  # Maximum number of routes per city
    max_routes_per_upgrade_city = 3  # Maximum routes for cities with upgrades

    # Initializing tensors for different actions
    claim_route_for_points_tensor = torch.zeros(max_num_routes, device=device, dtype=torch.uint8)
    claim_route_for_office_tensor = torch.zeros(max_num_cities * max_routes_per_city, device=device, dtype=torch.uint8)
    claim_route_for_upgrade_tensor = torch.zeros(max_num_cities * max_routes_per_upgrade_city, device=device, dtype=torch.uint8)

    route_idx = 0
    for route in game.selected_map.routes:
        if route.is_controlled_by(game.current_player):
            # Claim route for points
            claim_route_for_points_tensor[route_idx] = 1

            for city_idx, city in enumerate(route.cities):
                # Calculate indices for tensor
                action_index_office = city_idx * max_routes_per_city + route_idx
                action_index_upgrade = city_idx * max_routes_per_upgrade_city + route_idx

                # Check for office claim
                if city.has_empty_office():
                    next_open_office_color = city.get_next_open_office_color()
                    if game.current_player.player_can_claim_office(next_open_office_color) and city.color != DARK_GREEN:
                        if not city.has_required_piece_shape(game.current_player, route):
                            claim_route_for_office_tensor[action_index_office] = 1

                # Check for upgrade
                if city.upgrade_city_type:
                    claim_route_for_upgrade_tensor[action_index_upgrade] = 1

        route_idx += 1

    # Concatenate tensors to form a single tensor representing all claim route actions
    claim_route_tensor = torch.cat([claim_route_for_points_tensor, claim_route_for_office_tensor, claim_route_for_upgrade_tensor])
    return claim_route_tensor

def get_city_index(city, game):
    # Implement this function to return the index of a city in the game's city list
    return game.selected_map.cities.index(city)

def mask_income_actions(game):
    income_tensor = torch.zeros(5, device=device, dtype=torch.uint8)  # 5 options for income actions

    num_circles = game.current_player.general_stock_circles
    num_squares = game.current_player.general_stock_squares
    max_income = min(game.current_player.bank, 4)  # Limit to a maximum of 4 for circle income

    # Check if the player has no general stock circles or squares
    if num_circles == 0 and num_squares == 0:
        return income_tensor  # Return all zeros if no general stock pieces are available

    # Check if the player has no general stock circles
    if num_circles == 0:
        income_tensor[0] = 1  # Only valid action is to collect all squares
        return income_tensor

    # Set value for each possible income action
    for i in range(max_income + 1):
        if i <= num_circles:
            income_tensor[i] = 1  # Valid action if the player has enough circles

    return income_tensor

def mask_bm(game):
    bm_tensor = torch.zeros(8, device=device, dtype=torch.uint8)  # 8 possible permanent bonus markers
    bm_mapping = {
        "SwapOffice": 0,
        "Move3": 1,
        "UpgradeAbility": 2,
        "3Actions": 3,
        "4Actions": 4,
        "Exchange Bonus Marker": 5,
        "Tribute for Establish a Trading Post": 6,
        "Block Trade Route": 7
    }

    for bm in game.current_player.bonus_markers:
        bm_index = bm_mapping.get(bm.type)
        if bm_index is not None:
            bm_tensor[bm_index] = 1

    return bm_tensor

def mask_perm_bm(game):
    perm_bm_tensor = torch.zeros(5, device=device, dtype=torch.uint8)  # 5 possible permanent bonus markers
    bm_mapping = {
        "MoveAny2": 0,
        "+1Priv": 1,
        "ClaimGreenCity": 2,
        "Place2TradesmenFromRoute": 3,
        "Place2ScotlandOrWales": 4
    }

    for bm in game.current_player.bonus_markers:
        bm_index = bm_mapping.get(bm.type)
        if bm_index is not None:
            perm_bm_tensor[bm_index] = 1

    return perm_bm_tensor

def mask_replace_bm(game):
    max_num_routes = 40  # Maximum number of routes
    replace_bm_tensor = torch.zeros(max_num_routes, device=device, dtype=torch.uint8)  # Tensor for replace bonus marker actions

    for route_idx, route in enumerate(game.selected_map.routes):
        if not (route.bonus_marker or route.permanent_bonus_marker) \
                and not route.has_tradesmen() \
                and route.has_empty_office_in_cities():
            replace_bm_tensor[route_idx] = 1

    return replace_bm_tensor

def mask_end_turn(game):
    end_turn_tensor = torch.zeros(1, device=device, dtype=torch.uint8)
    if game.replace_bonus_marker == 0:
        end_turn_tensor[0] = 1
    return end_turn_tensor

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
        print (f"player_info[{i}] {player_info[i]}")
    
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