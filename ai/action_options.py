import torch
import time
from map_data.constants import GREEN, BLUE, PURPLE, RED, YELLOW, BLACKISH_BROWN, DARK_RED, DARK_GREEN, DARK_BLUE, GREY, COLOR_NAMES
from game.game_actions import claim_post_action, displace_action, move_action, displace_claim

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLAIM_POST_ACTIONS = 242
NUM_CLAIM_ROUTE_ACTIONS = 280
NUM_INCOME_ACTIONS = 5
NUM_BM_ACTIONS = 8
NUM_PERM_BM_ACTIONS = 5
NUM_REPLACE_BM_ACTIONS = 40
NUM_END_TURN_ACTIONS = 1

# Assuming max_prob_index is the index of the chosen action
def perform_action_from_index(game, max_prob_index):
    if max_prob_index < NUM_CLAIM_POST_ACTIONS:
        return map_claim_post_action(game, max_prob_index)
    elif max_prob_index < NUM_CLAIM_POST_ACTIONS + NUM_CLAIM_ROUTE_ACTIONS:
        return map_claim_route_action(game, max_prob_index - NUM_CLAIM_POST_ACTIONS)
    # ... Repeat for other categories
    # elif max_prob_index < TOTAL_ACTIONS - NUM_END_TURN_ACTIONS:
    #     return map_end_turn_action(max_prob_index - (TOTAL_ACTIONS - NUM_END_TURN_ACTIONS))

    # Handle default or error case
    return None

def map_claim_post_action(game, index):
    current_player = game.current_player

    post_type = 'square' if index < 121 else 'circle'
    ai_post_selection = index % 121  # Get the actual post index
    post_idx = 0
    for route in game.selected_map.routes:
        for post in route.posts:
            post_idx += 1
            if post_idx == ai_post_selection:
                break

    # Assuming you have a way to get the route and the specific post in the route
    # post = route.posts[post_in_route_index]

    is_post_owned = post.is_owned()
    is_post_empty = not is_post_owned

    can_displace = False

    if current_player.personal_supply_squares + current_player.personal_supply_circles > 1:
        can_displace = True

    # CLAIM AS DISPLACED PLAYER - if post is empty
    if game.waiting_for_displaced_player:
        if post in game.all_empty_posts:
            displace_claim(game, post, post_type)
    #handle BM Move any2 or #handle BM Move 3:
    elif is_post_owned and ((game.waiting_for_bm_move_any_2) or (game.waiting_for_bm_move3 and post.owner != current_player)):
        if len(current_player.holding_pieces) < current_player.pieces_to_place:
            move_action(game, post, post_type)
    else:
        # Claim post with MOVE action: check if the post is empty and region is valid
        if len(current_player.holding_pieces) < current_player.pieces_to_place and is_post_empty:
            if post_type in current_player.holding_pieces and (not post.required_shape or post.required_shape == post_type):
                move_action(game, post, post_type)

        # MOVE - if post is owned by current_player:
        elif is_post_owned and post.owner == current_player:
            if len(current_player.holding_pieces) < current_player.book:
                move_action(game, post, post_type)

        elif is_post_empty:
            if current_player.has_personal_supply(post_type) and (not post.required_shape or post.required_shape == post_type):
                claim_post_action(game, route, post, post_type)
        # DISPLACE - if post is owned by a different player:
        elif is_post_owned and post.owner != current_player and check_brown_blue_priv(game, route) and can_displace:
                displace_action(game, post, route, post_type)
        else:
            print (f"something invalid happened with post index {post_idx} of shape {post_type}")

    # Add other conditions as needed
def map_claim_route_action(index):
    # Logic to map index to a specific Claim Post action
    # Example: return claim_post_actions[index]
    pass
def map_income_action(index):
    # Logic to map index to a specific Claim Post action
    # Example: return claim_post_actions[index]
    pass
def map_bm_action(index):
    # Logic to map index to a specific Claim Post action
    # Example: return claim_post_actions[index]
    pass
def map_perm_bm_action(index):
    # Logic to map index to a specific Claim Post action
    # Example: return claim_post_actions[index]
    pass
def map_replace_bm_action(index):
    # Logic to map index to a specific Claim Post action
    # Example: return claim_post_actions[index]
    pass
def map_end_turn_action(index):
    # Logic to map index to a specific Claim Post action
    # Example: return claim_post_actions[index]
    pass



def masking_out_invalid_actions(game):
    
    claim_post_tensor = mask_post_action(game) #size 242 (claiming with a square or circle))
    claim_route_tensor = mask_claim_route(game) #size 280 (claim for points, office, or upgrade)
    income_tensor = mask_income_actions(game) #size 5 (0-4 circles + leftover squares)
    bonus_marker_tensor = mask_bm(game) #size 8 (8 total BM types to use)
    permanent_bonus_marker_tensor = mask_perm_bm(game) #size 5
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
                # Claim post action: check if the post is empty and region is valid
                if len(current_player.holding_pieces) < current_player.pieces_to_place and is_post_empty:
                    if game.current_player.is_valid_region_transition(current_player.holding_pieces[0].region, post.region):
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
                # else:
                #     print (f"Invalid scenario detected. Post Info - Circle Color: {COLOR_NAMES[post.circle_color]}, Square Color: {COLOR_NAMES[post.square_color]}, Owner: {post.owner}, Region: {post.region}, ReqShape: {post.required_shape}")
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