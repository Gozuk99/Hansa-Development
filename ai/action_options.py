import sys
import torch
import time
from map_data.constants import DARK_GREEN, COLOR_NAMES, UPGRADE_MAX_VALUES, MAX_CITIES, MAX_ROUTES, MAX_POSTS
from game.game_actions import claim_post_action, displace_action, move_action, displace_claim, assign_new_bonus_marker_on_route, claim_route_for_office, claim_route_for_upgrade, claim_route_for_points, buy_tile

#debugging
from drawing.drawing_utils import redraw_window
import pygame

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLAIM_POST_ACTIONS = 242        # Actions for claiming posts, index range: 0 - 241
NUM_CLAIM_ROUTE_ACTIONS = 280       # Actions for claiming routes, index range: 242 - 521
NUM_INCOME_ACTIONS = 5              # Actions for income, index range: 522 - 526
NUM_BM_ACTIONS = 8                  # Actions for BM, index range: 527 - 534
NUM_BUY_TILE_ACTIONS = 8            # Actions for buying a tile, index range: 535 - 542
NUM_REPLACE_BM_ACTIONS = 40         # Actions for replacing BM, index range: 543 - 582
NUM_BM_CITY_ACTIONS = 30            # Actions for BM related to cities, index range: 583 - 612
NUM_BM_UPGRADE = 5                  # Actions for BM upgrades, index range: 613 - 617
NUM_END_TURN_ACTIONS = 1            # Action to end turn, index range: 618

# Calculating the total actions
TOTAL_ACTIONS = (NUM_CLAIM_POST_ACTIONS + NUM_CLAIM_ROUTE_ACTIONS + NUM_INCOME_ACTIONS + 
                 NUM_BM_ACTIONS + NUM_BUY_TILE_ACTIONS + NUM_REPLACE_BM_ACTIONS + 
                 NUM_BM_CITY_ACTIONS + NUM_BM_UPGRADE + NUM_END_TURN_ACTIONS)

def perform_action_from_index(game, max_prob_index):
    if max_prob_index < NUM_CLAIM_POST_ACTIONS:
        map_claim_post_action(game, max_prob_index)
    elif max_prob_index < NUM_CLAIM_POST_ACTIONS + NUM_CLAIM_ROUTE_ACTIONS:
        map_claim_route_action(game, max_prob_index - NUM_CLAIM_POST_ACTIONS)
    elif max_prob_index < NUM_CLAIM_POST_ACTIONS + NUM_CLAIM_ROUTE_ACTIONS + NUM_INCOME_ACTIONS:
        map_income_action(game, max_prob_index - NUM_CLAIM_POST_ACTIONS - NUM_CLAIM_ROUTE_ACTIONS)
    elif max_prob_index < NUM_CLAIM_POST_ACTIONS + NUM_CLAIM_ROUTE_ACTIONS + NUM_INCOME_ACTIONS + NUM_BM_ACTIONS:
        map_bm_action(game, max_prob_index - NUM_CLAIM_POST_ACTIONS - NUM_CLAIM_ROUTE_ACTIONS - NUM_INCOME_ACTIONS)
    elif max_prob_index < NUM_CLAIM_POST_ACTIONS + NUM_CLAIM_ROUTE_ACTIONS + NUM_INCOME_ACTIONS + NUM_BM_ACTIONS + NUM_BUY_TILE_ACTIONS:
        map_buy_tile_action(game, max_prob_index - NUM_CLAIM_POST_ACTIONS - NUM_CLAIM_ROUTE_ACTIONS - NUM_INCOME_ACTIONS - NUM_BM_ACTIONS)
    elif max_prob_index < NUM_CLAIM_POST_ACTIONS + NUM_CLAIM_ROUTE_ACTIONS + NUM_INCOME_ACTIONS + NUM_BM_ACTIONS + NUM_BUY_TILE_ACTIONS + NUM_REPLACE_BM_ACTIONS:
        map_replace_bm_action(game, max_prob_index - NUM_CLAIM_POST_ACTIONS - NUM_CLAIM_ROUTE_ACTIONS - NUM_INCOME_ACTIONS - NUM_BM_ACTIONS - NUM_BUY_TILE_ACTIONS)
    elif max_prob_index < NUM_CLAIM_POST_ACTIONS + NUM_CLAIM_ROUTE_ACTIONS + NUM_INCOME_ACTIONS + NUM_BM_ACTIONS + NUM_BUY_TILE_ACTIONS + NUM_REPLACE_BM_ACTIONS + NUM_BM_CITY_ACTIONS:
        map_bm_city_actions(game, max_prob_index - NUM_CLAIM_POST_ACTIONS - NUM_CLAIM_ROUTE_ACTIONS - NUM_INCOME_ACTIONS - NUM_BM_ACTIONS - NUM_BUY_TILE_ACTIONS - NUM_REPLACE_BM_ACTIONS)
    elif max_prob_index < NUM_CLAIM_POST_ACTIONS + NUM_CLAIM_ROUTE_ACTIONS + NUM_INCOME_ACTIONS + NUM_BM_ACTIONS + NUM_BUY_TILE_ACTIONS + NUM_REPLACE_BM_ACTIONS + NUM_BM_CITY_ACTIONS + NUM_BM_UPGRADE:
        map_bm_upgrade_ability(game, max_prob_index - NUM_CLAIM_POST_ACTIONS - NUM_CLAIM_ROUTE_ACTIONS - NUM_INCOME_ACTIONS - NUM_BM_ACTIONS - NUM_BUY_TILE_ACTIONS - NUM_REPLACE_BM_ACTIONS - NUM_BM_CITY_ACTIONS)
    elif max_prob_index == TOTAL_ACTIONS - 1:
        map_end_turn_action(game)

    if not (game.waiting_for_bm_swap_office or game.waiting_for_bm_upgrade_ability or game.waiting_for_bm_move_any_2 or \
            game.waiting_for_bm_move3 or game.waiting_for_bm_exchange_bm or game.waiting_for_bm_tribute_trading_post or\
            game.waiting_for_bm_block_trade_route or game.waiting_for_bm_green_city or game.waiting_for_place2_in_scotland_or_wales):
        # print(f"BM flag waiting_for_bm_swap_office: {game.waiting_for_bm_swap_office}")
        # print(f"BM flag waiting_for_bm_upgrade_ability: {game.waiting_for_bm_upgrade_ability}")
        # print(f"BM flag waiting_for_bm_move_any_2: {game.waiting_for_bm_move_any_2}")
        # print(f"BM flag waiting_for_bm_move3: {game.waiting_for_bm_move3}")
        # print(f"BM flag waiting_for_bm_exchange_bm: {game.waiting_for_bm_exchange_bm}")
        # print(f"BM flag waiting_for_bm_tribute_trading_post: {game.waiting_for_bm_tribute_trading_post}")
        # print(f"BM flag waiting_for_bm_block_trade_route: {game.waiting_for_bm_block_trade_route}")
        # print(f"BM flag waiting_for_bm_green_city: {game.waiting_for_bm_green_city}")
        # print(f"BM flag waiting_for_place2_in_scotland_or_wales: {game.waiting_for_place2_in_scotland_or_wales}")
        game.switch_player_if_needed()
    # Handle default or error case
    return None

def error_exit(game, route):
    win = pygame.display.set_mode((game.selected_map.map_width+800, game.selected_map.map_height))
    # viewable_window = pygame.display.set_mode((1800, 1350))
    pygame.display.set_caption('Hansa Sample Game')
    win.fill((210, 180, 140))

    if route:
        print(f"Route between cities {route.cities[0].name} and {route.cities[1].name} has an error.")
        for i, post in enumerate(route.posts):
            if post.owner:
                print(f"[{i}] Post Owner: {COLOR_NAMES[post.owner.color]}, Post Owner Piece Shape: {post.owner_piece_shape}")
            else:
                print(f"[{i}] Post Owner: None, Post Owner Piece Shape: {post.owner_piece_shape}")
    # Redraw the window (if necessary)
    redraw_window(win, game)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()      

        # Update the display
        pygame.display.update()

def map_claim_post_action(game, index):
    current_player = game.current_player
    post_type = 'square' if index < MAX_POSTS else 'circle'
    ai_post_selection = index % MAX_POSTS  # Get the actual post index
    selected_post = None
    selected_route = None

    # Iterate over all routes and their posts
    post_idx = 0
    for route in game.selected_map.routes:
        for post in route.posts:
            if post_idx == ai_post_selection:
                selected_post = post
                selected_route = route
                # print(f"Selected post on route between: Route between {selected_route.cities[0].name} and {selected_route.cities[1].name}.")
                break
            post_idx += 1
        if selected_post:
            break

    if not selected_post:
        print(f"ERROR: This should have been masked out index:{index}, ai_post_index: {ai_post_selection}")
        error_exit(game, selected_route)
        return

    is_post_owned = selected_post.is_owned()
    is_post_empty = not is_post_owned
    # print(f"Post status - Owned: {is_post_owned}, Empty: {is_post_empty}, Type: {post_type}")

    can_displace = current_player.personal_supply_squares + current_player.personal_supply_circles > 1
    # print(f"Player able to displace if post owned? {can_displace}")
    # print(f"Holding pieces {len(current_player.holding_pieces)} and pieces to place: {current_player.pieces_to_pickup}")

    # CLAIM AS DISPLACED PLAYER - if post is empty
    if game.waiting_for_displaced_player:
        if selected_post.owner == game.displaced_player.player and game.displaced_player.is_general_stock_empty() and game.displaced_player.is_personal_supply_empty():
            print(f"DISPLACE - Performing PICKUP action for {post_type} on post {post_idx} because GS and PS is empty")
            displace_claim(game, selected_post, post_type)
        elif selected_post in game.all_empty_posts:
            print(f"DISPLACE - Performing displaced claim action for {post_type} on post {post_idx} between {selected_route.cities[0].name} and {selected_route.cities[1].name}")
            displace_claim(game, selected_post, post_type)
        else:
            print(f"DISPLACE ERROR - Selected Post NOT in game.all_empty_posts")
            exit()

    elif game.waiting_for_bm_move_any_2:
        print(f"Performing BM Move Any 2 action for {post_type} on post {post_idx}")
        move_action(game, selected_route, selected_post, post_type)

    elif game.waiting_for_place2_in_scotland_or_wales:
        if selected_post in game.all_empty_posts:
            print(f"Performing BM Place2 in Scotland or Wales action for {post_type} on post {post_idx}")
            move_action(game, selected_route, selected_post, post_type)
        else:
            print(f"ERROR: This should have been masked out index:{index}, ai_post_index: {ai_post_selection}")
            error_exit(game, selected_route)
    
    elif game.waiting_for_bm_move3:
        if selected_post.owner != current_player:
            print(f"Performing BM Move3 action for {post_type} on post {post_idx}")
            move_action(game, selected_route, selected_post, post_type)
        else:
            print(f"ERROR: This should have been masked out index:{index}, ai_post_index: {ai_post_selection}")
    elif game.waiting_for_bm_tribute_trading_post:
        current_player.personal_supply_squares -= 1
        selected_route.establish_tribute_on_route(current_player)
        game.waiting_for_bm_block_trade_route = False
    elif game.waiting_for_bm_block_trade_route:
        current_player.personal_supply_squares -= 1
        selected_route.establish_blocked_route()
        game.waiting_for_bm_block_trade_route = False
    else:
        # Claim post with MOVE action: check if the post is empty
        # Check if the desired post type is available in holding pieces
        post_type_available = any(piece[0] == post_type for piece in current_player.holding_pieces)

        # # Now use this condition in your if statement
        # if current_player.holding_pieces:
        #     if is_post_empty and post_type_available:
        #         # print(f"MOVE - Attempting to claim empty post {post_idx} with type {post_type}")
        #         if selected_post.required_shape is None or selected_post.required_shape == post_type:
        #             move_action(game, selected_route, post, post_type)
        #         else:
        #             print(f"MOVE CLAIM ERROR: Post Type {post_type}, shape: {selected_post.required_shape}")
        #             print(f"MOVE CLAIM ERROR: holding_pieces {current_player.holding_pieces[0]}")            
        #             error_exit(game, selected_route)
        #     # MOVE - if post is owned by current_player:
        #     elif is_post_owned and selected_post.owner == current_player and current_player.pieces_to_pickup > 0:
        #         # print(f"MOVE - Attempting MOVE action on owned post {post_idx} with type {post_type}")
        #         if len(current_player.holding_pieces) < current_player.book:
        #             move_action(game, selected_route, selected_post, post_type)
        #         else:
        #             print(f"MOVE PICKUP ERROR: Post Type {post_type}, shape: {selected_post.required_shape}")
        #             error_exit(game, selected_route)
        #     else:
        #         print(f"MOVE ERROR: Post Type {post_type}, shape: {selected_post.required_shape}")
        #         print(f"MOVE ERROR: Pieces held {len(current_player.holding_pieces)}")
        #         error_exit(game, selected_route)

        # # MOVE - if post is owned by current_player:
        # elif is_post_owned and selected_post.owner == current_player:
        #     # print(f"MOVE - Attempting MOVE action on owned post {post_idx} with type {post_type}")
        #     if len(current_player.holding_pieces) < current_player.book:
        #         move_action(game, selected_route, selected_post, post_type)
        #     else:
        #         print(f"MOVE PICKUP ERROR: Post Type {post_type}, shape: {selected_post.required_shape}")
        #         error_exit(game, selected_route)

        # Claim post if it's empty
        if is_post_empty and check_brown_blue_priv(game, selected_route):
            # print(f"CLAIM - Attempting to claim empty post {post_idx} with type {post_type}")
            if current_player.has_personal_supply(post_type) and (selected_post.required_shape is None or selected_post.required_shape == post_type):
                claim_post_action(game, selected_route, selected_post, post_type)
            else:
                print(f"CLAIM ERROR: Trying to claim with {post_type}, shape: {selected_post.required_shape}")
                print(f"CLAIM ERROR: current_player.personal_supply_squares {current_player.personal_supply_squares}")
                print(f"CLAIM ERROR: current_player.personal_supply_circles {current_player.personal_supply_circles}")
                print(f"CLAIM ERROR: selected_post.required_shape {selected_post.required_shape}")
                print(f"CLAIM ERROR: post_type {post_type}")
        # DISPLACE - if post is owned by a different player:
        elif is_post_owned and selected_post.owner != current_player and check_brown_blue_priv(game, selected_route) and can_displace:
            print(f"Attempting DISPLACE action on post {post_idx} between {selected_route.cities[0].name} and {selected_route.cities[1].name}")
            print(f"Post owned by {COLOR_NAMES[selected_post.owner.color]} {selected_post.owner_piece_shape}")
            displace_action(game, selected_post, selected_route, post_type)
        else:
            print(f"something invalid happened with post index {post_idx} of shape {post_type}")
            error_exit(game, selected_route)

# 40+80+160=280
def map_claim_route_action(game, index):
    # Size of each action type
    num_points_actions = MAX_ROUTES  #40
    num_office_actions = MAX_ROUTES * 2  # e.g., 80
    num_upgrade_actions = MAX_ROUTES * 2 * 2  # e.g., 160

    # Claim route for points
    if index < num_points_actions:
        route_idx = index
        route = game.selected_map.routes[route_idx]
        if route.is_controlled_by(game.current_player):
            claim_route_for_points(game, route)
        else:
            print(f"ERROR - Cannot claim route for points! Route {route_idx} not controlled by {COLOR_NAMES[game.current_player.color]}.")
            error_exit(game, route)

    # Claim an office in a city
    elif index < num_points_actions + num_office_actions:
        adjusted_index = index - num_points_actions
        route_idx = adjusted_index // 2  # Two cities per route
        city_idx = adjusted_index % 2  # Which city on the route
        route = game.selected_map.routes[route_idx]
        city = route.cities[city_idx]
        if city.has_empty_office() and route.is_controlled_by(game.current_player):
            claim_route_for_office(game, city, route)
        else:
            print(f"ERROR - Cannot claim office in {city.name}! Route not controlled or no empty office.")
            error_exit(game, route)

    # Upgrade in a city
    elif index < num_points_actions + num_office_actions + num_upgrade_actions:
        adjusted_index = index - (num_points_actions + num_office_actions)
        route_idx = adjusted_index // 4  # Four upgrade possibilities per route (2 cities × 2 upgrades each)
        city_idx = (adjusted_index // 2) % 2  # Which city on the route
        upgrade_idx = adjusted_index % 2  # Which upgrade in the city

        route = game.selected_map.routes[route_idx]
        city = route.cities[city_idx]
        if city.upgrade_city_type and len(city.upgrade_city_type) > upgrade_idx:
            upgrade_choice = city.upgrade_city_type[upgrade_idx]
            if route.is_controlled_by(game.current_player):
                claim_route_for_upgrade(game, city, route, upgrade_choice)
            else:
                print(f"ERROR - Cannot upgrade in {city.name}! Route not controlled.")
                error_exit(game, route)
        else:
            print(f"ERROR - Invalid upgrade index or no upgrades available in {city.name}.")
            error_exit(game, route)

    else:
        print("Invalid index for claim route action.")
        error_exit(game)

def map_income_action(game, index):
    current_player = game.current_player
    num_circles = current_player.general_stock_circles
    num_squares = current_player.general_stock_squares

    # Check if the player has no general stock circles or squares
    if num_circles == 0 and num_squares == 0:
        print(f"{COLOR_NAMES[current_player.color]} doesn't have any GS Circles or Squares - this should have been masked out")
        return

    # Check if the player has no general stock circles
    if num_circles == 0:
        print(f"[{current_player.actions_remaining}] {COLOR_NAMES[current_player.color]} INCOME - Squares: {min(num_squares, current_player.bank)}, Circles: 0.")
        current_player.income_action(min(num_squares, current_player.bank), 0)
        return

    # Determine the number of circles and squares for each index
    if index == 0:  # All squares, no circles
        squares_to_take = min(num_squares, current_player.bank)
        circles_to_take = 0
    else:  # Take 'index' number of circles and the rest squares
        circles_to_take = min(num_circles, index)
        squares_to_take = min(num_squares, current_player.bank - circles_to_take)

    # Perform the income action
    print(f"[{current_player.actions_remaining}] {COLOR_NAMES[current_player.color]} INCOME - Squares: {squares_to_take}, Circles: {circles_to_take}.")
    current_player.income_action(squares_to_take, circles_to_take)

def map_bm_action(game, index):
    current_player = game.current_player

    bm_mapping = {
        "SwapOffice": 0,
        "Move3": 1,
        "UpgradeAbility": 2,
        "3Actions": 3,
        "4Actions": 4,
        "ExchangeBonusMarker": 5,
        "Tribute4EstablishingTP": 6,
        "BlockTradeRoute": 7
    }

    if game.waiting_for_bm_exchange_bm:
        for player in game.players:
            if player != current_player:
                for bm in player.bonus_markers:
                    bm_index = bm_mapping.get(bm.type)
                    if bm_index is not None:
                        if bm_index == index:
                            player.bonus_markers.remove(bm)
                            current_player.bonus_markers.append(bm)
                            game.waiting_for_bm_exchange_bm = False
                            return
    else:
        selected_bm = None
        for bm in game.current_player.bonus_markers:
            bm_index = bm_mapping.get(bm.type)
            if bm_index is not None:
                if bm_index == index:
                    selected_bm = bm
                    print(f"1Selected BM: {selected_bm.type}")
                    break
        print(f"2Selected BM: {selected_bm.type}")
        if selected_bm.type == "SwapOffice":
            game.waiting_for_bm_swap_office = True
        elif selected_bm.type == "Move3":
            selected_bm.handle_move3(game)
            print(f"3Selected BM: {selected_bm.type}, waiting_for_bm_move3: {game.waiting_for_bm_move3}, pieces_to_pickup: {current_player.pieces_to_pickup}")
        elif selected_bm.type == "UpgradeAbility":
            game.waiting_for_bm_upgrade_ability = True
        elif selected_bm.type == "3Actions":
            selected_bm.handle_3_actions(current_player)
        elif selected_bm.type == "4Actions":
            selected_bm.handle_4_actions(current_player)

        elif selected_bm.type == "ExchangeBonusMarker":
            game.waiting_for_bm_exchange_bm = True
        elif selected_bm.type == "Tribute4EstablishingTP" and current_player.personal_supply_squares > 0:
            game.waiting_for_bm_tribute_trading_post = True
        elif selected_bm.type == "BlockTradeRoute" and current_player.personal_supply_squares > 0:
            game.waiting_for_bm_block_trade_route = True

        game.current_player.bonus_markers.remove(selected_bm)
    return

def map_buy_tile_action(game, index):
    current_player = game.current_player
    tile_mapping = {
        1: 'DisplaceAnywhere',
        2: '+1Action',
        3: '+1IncomeIfOthersIncome',
        4: '+1DisplacedPiece',
        5: '+4PtsPerOwnedCity',
        6: '+7PtsPerCompletedAbility'
    }
    bm_mapping = {
        "SwapOffice": 0,
        "Move3": 1,
        "UpgradeAbility": 2,
        "3Actions": 3,
        "4Actions": 4,
        "ExchangeBonusMarker": 5,
        "Tribute4EstablishingTP": 6,
        "BlockTradeRoute": 7
    }

    if game.waiting_for_buy_tile_with_bm:
        if game.first_bm_to_spend:
            second_bm_to_spend = bm_mapping.get(index)
            buy_tile(game, game.tile_to_buy, game.first_bm_to_spend, second_bm_to_spend)

            game.waiting_for_buy_tile_with_bm = False
            game.first_bm_to_spend = None
            game.tile_to_buy = None
        else:
            game.first_bm_to_spend = bm_mapping.get(index)

    bm_to_spend = [None, None]

    if len(current_player.bonus_markers) == 2:
        for i, bm in enumerate(current_player.bonus_markers):
            bm_to_spend[i] = bm
        buy_tile(game, tile_mapping.get(index), bm_to_spend[0], bm_to_spend[1])
    
    elif len(current_player.bonus_markers) > 2:
        game.waiting_for_buy_tile_with_bm = True
        game.tile_to_buy = tile_mapping.get(index)

    else:
        print(f"{COLOR_NAMES[current_player.color]} has less than 2 bonus markers to spend on tiles - this shoulda been masked out.")

    return

def map_replace_bm_action(game, index):
    current_player = game.current_player

    print(f"Replace BM action for index: {index}")
    print(f"Current Player: {COLOR_NAMES[current_player.color]}, Actions Remaining: {current_player.actions_remaining}, Ending Turn: {current_player.ending_turn}, Replace BM: {game.replace_bonus_marker}")
    # Check if the conditions are met to replace a bonus marker
    if current_player.actions_remaining == 0 and current_player.ending_turn and game.replace_bonus_marker > 0:
        # Check if the index is within the range of the number of routes
        if index < MAX_ROUTES:
            route = game.selected_map.routes[index]
            if not (route.bonus_marker or route.permanent_bonus_marker) and \
                    not route.has_tradesmen() and route.has_empty_office_in_cities():
                # Place the bonus marker on the selected route
                assign_new_bonus_marker_on_route(game, route)
                # Check if all bonus markers have been placed
                if game.replace_bonus_marker == 0:
                    current_player.ending_turn = False
                    game.switch_player_if_needed()
                if game.replace_bonus_marker < 0:
                    print(f"Invalid number of bonus markers to replace. Remaining: {game.replace_bonus_marker}")
                    error_exit(game, None)
            else:
                print(f"Invalid route selected for bonus marker placement. Route Index: {index}")
        else:
            print(f"Index out of range for route selection. Index: {index}")
    else:
        print(f"{COLOR_NAMES[current_player.color]} has actions remaining or conditions not met for bonus marker replacement.")

def map_bm_city_actions(game, index):
    for city_idx, city in enumerate(game.selected_map.cities):
        if city_idx == index:
            if game.waiting_for_bm_swap_office:
                if city.check_if_eligible_to_swap_offices(game.current_player):
                    print ("Valid City to swap offices")
                    city.swap_offices(game.current_player)
                    game.waiting_for_bm_swap_office = False
            elif game.waiting_for_bm_green_city:
                if city.color == DARK_GREEN:
                    city.claim_green_city(game)
                    game.waiting_for_bm_green_city = False
            break
    
def map_bm_upgrade_ability(game, index):
    for upgrade_idx, upgrade_city in enumerate(game.selected_map.upgrade_cities):
        if upgrade_idx == index:
            if game.waiting_for_bm_upgrade_ability:
                upgrade_type = upgrade_city.upgrade_type
                game.current_player.perform_upgrade(upgrade_type)
                game.waiting_for_bm_upgrade_ability = False 

def map_end_turn_action(game):

    if game.current_player.actions_remaining == 0:
        if ((game.current_player.ending_turn and game.replace_bonus_marker > 0) or
            (game.replace_bonus_marker > 0 and not check_if_player_has_usable_BMs(game))):
            print("forcing player to replace bonus markers")
            game.current_player.ending_turn = True
        elif not check_if_player_has_usable_BMs(game) and game.replace_bonus_marker == 0:
            # Player has no BMs to use or only has 'PlaceAdjacent', switch player
            print("Ending turn for player.")
            print(f"replace_bonus_marker: {game.replace_bonus_marker}, check_if_player_has_usable_BMs: {check_if_player_has_usable_BMs(game)}")
            game.switch_player_if_needed()
    else:
        print("Cannot end the turn as Bonus Markers need to be replaced on the Map!")
        error_exit(game, None)

def masking_out_invalid_actions(game):
    
    claim_post_tensor = mask_post_action(game) #size 242 (claiming with a square or circle)
    claim_route_tensor = mask_claim_route(game) #size 280 (claim for points, office, or upgrade)
    # print(f"claim_route_tensor - {claim_route_tensor.size()}")
    income_tensor = mask_income_actions(game) #size 5 (0-4 circles + leftover squares)
    bonus_marker_tensor = mask_bm(game) #size 8 (8 total BM types to use)
    buy_tile_tensor = mask_buy_tile(game) #size 8 (6 tiles or 8 BMs to pay for the tiles)
    replace_bm_tensor = mask_replace_bm(game) #size 40
    bm_city_actions_tensor = mask_bm_city_actions(game) #size 30
    bm_upgrade_ability_tensor = mask_bm_upgrade_ability(game) #size 5
    end_turn_tensor = mask_end_turn(game) #size 1 (allowed to end turn if no bonus markers to replace)

    # Concatenate all tensors into one big tensor representing all possible actions
    all_actions_tensor = torch.cat([claim_post_tensor, claim_route_tensor, income_tensor, bonus_marker_tensor, buy_tile_tensor,
                                    replace_bm_tensor, bm_city_actions_tensor, bm_upgrade_ability_tensor, end_turn_tensor], dim=0)
    # print(f"all_actions_tensor.size - {all_actions_tensor.size()}")
    
    return all_actions_tensor

def check_if_any_post_BM_flag_set(game):
    return game.waiting_for_displaced_player or game.waiting_for_bm_move_any_2 or game.waiting_for_bm_move3

def check_if_any_action_BM_flag_set(game):
    return game.waiting_for_bm_swap_office or game.waiting_for_bm_upgrade_ability or game.waiting_for_bm_green_city

def mask_post_action(game):
    current_player = game.current_player

    post_tensor = torch.zeros(MAX_POSTS * 2, device=device, dtype=torch.uint8)  # 121 max posts * 2 for squares and circles

    if (game.waiting_for_bm_move_any_2 or game.waiting_for_place2_in_scotland_or_wales or \
        game.waiting_for_bm_move3 or game.waiting_for_bm_tribute_trading_post or game.waiting_for_bm_block_trade_route):
        print("BM flag set.")
    elif current_player.actions_remaining == 0 or check_if_any_action_BM_flag_set(game):
        return post_tensor
    
    can_displace = current_player.personal_supply_squares + current_player.personal_supply_circles > 1
    
    post_idx = 0
    for route in game.selected_map.routes:
        # print(f"Route between cities {route.cities[0].name} and {route.cities[1].name}.")
        for post in route.posts:  # Make sure to iterate over posts in each route
            # if post.owner:
            #     print(f"Post {post_idx} - Owner: {COLOR_NAMES[post.owner.color] if post.owner else None}, owner_piece_shape: {post.owner_piece_shape}, waiting_for_bm_move3: {game.waiting_for_bm_move3}")
            # Common checks for valid region transition and post not being owned
            is_post_owned = post.is_owned()
            is_post_empty = not is_post_owned

            # CLAIM AS DISPLACED PLAYER - if post is empty
            if game.waiting_for_displaced_player:
                displaced_player = game.displaced_player
                must_use_displaced_piece = (not displaced_player.played_displaced_shape) and (displaced_player.total_pieces_to_place == 1)
                has_pieces_in_general_stock = (displaced_player.player.general_stock_squares + displaced_player.player.general_stock_circles) > 0
                has_pieces_in_personal_supply = (displaced_player.player.personal_supply_squares + displaced_player.player.personal_supply_circles) > 0
                
                if not has_pieces_in_general_stock and not has_pieces_in_personal_supply:
                    if post in game.all_empty_posts and not displaced_player.played_displaced_shape:
                        if displaced_player.displaced_shape == "square" and (post.required_shape == displaced_player.displaced_shape or post.required_shape is None):
                            post_tensor[post_idx] = 1  # Offset by 121 for circle posts if necessary
                        elif displaced_player.displaced_shape == "circle" and (post.required_shape == displaced_player.displaced_shape or post.required_shape is None):
                            post_tensor[MAX_POSTS + post_idx] = 1  # Offset by 121 for circle posts if necessary
                    elif post in game.all_empty_posts and displaced_player.played_displaced_shape and displaced_player.player.pieces_to_pickup == 0:
                        if displaced_player.displaced_shape == "square" and (post.required_shape == displaced_player.displaced_shape or post.required_shape is None):
                            post_tensor[post_idx] = 1  # Offset by 121 for circle posts if necessary
                        elif displaced_player.displaced_shape == "circle" and (post.required_shape == displaced_player.displaced_shape or post.required_shape is None):
                            post_tensor[MAX_POSTS + post_idx] = 1  # Offset by 121 for circle posts if necessary
                    elif post.owner == displaced_player.player and displaced_player.played_displaced_shape and displaced_player.player.pieces_to_pickup > 0:
                        post_tensor[post_idx] = 1
                        post_tensor[MAX_POSTS + post_idx] = 1  # Offset by 121 for circle posts if necessary
                    else:
                        if post.owner == displaced_player.player and displaced_player.played_displaced_shape and displaced_player.player.pieces_to_pickup > 0:
                            print(f"DISPLACE MASK ERROR: Displaced Shape: {displaced_player.played_displaced_shape}, index: {post_idx}")
                            print(f"displaced_player.player = {COLOR_NAMES[displaced_player.player.color]}, pieces_to_pickup = {displaced_player.player.pieces_to_pickup}")
                            error_exit(game, route)
                        # elif post in game.all_empty_posts and displaced_player.played_displaced_shape and displaced_player.player.pieces_to_pickup > 0:
                        #     print(f"Waiting to pick up remaining pieces: {displaced_player.player.pieces_to_pickup}")

                elif post in game.all_empty_posts:
                    if must_use_displaced_piece:
                        if displaced_player.displaced_shape == "square" and (post.required_shape == displaced_player.displaced_shape or post.required_shape is None):
                            post_tensor[post_idx] = 1  # Offset by 121 for circle posts if necessary
                        elif displaced_player.displaced_shape == "circle" and (post.required_shape == displaced_player.displaced_shape or post.required_shape is None):
                            post_tensor[MAX_POSTS + post_idx] = 1  # Offset by 121 for circle posts if necessary
                    
                    elif displaced_player.played_displaced_shape:
                        if has_pieces_in_general_stock:
                            if displaced_player.player.has_general_stock("square") and (not post.required_shape or post.required_shape == "square"):
                                post_tensor[post_idx] = 1
                            if displaced_player.player.has_general_stock("circle") and (not post.required_shape or post.required_shape == "circle"):
                                post_tensor[MAX_POSTS + post_idx] = 1
                        elif has_pieces_in_personal_supply:
                            if displaced_player.player.has_personal_supply("square") and (not post.required_shape or post.required_shape == "square"):
                                post_tensor[post_idx] = 1
                            if displaced_player.player.has_personal_supply("circle") and (not post.required_shape or post.required_shape == "circle"):
                                post_tensor[MAX_POSTS + post_idx] = 1
                        # else:
                        #     print (f"1If displaced player has an empty GS and PS, eventually need to handle moving pieces.")

                    elif not displaced_player.played_displaced_shape:
                        if displaced_player.displaced_shape == "square" and (not post.required_shape or post.required_shape == "square"):
                            post_tensor[post_idx] = 1
                        elif displaced_player.displaced_shape == "circle" and (not post.required_shape or post.required_shape == "circle"):
                            post_tensor[MAX_POSTS + post_idx] = 1  # Offset by 121 for circle posts
                        
                        if has_pieces_in_general_stock:
                            if displaced_player.player.has_general_stock("square") and (not post.required_shape or post.required_shape == "square"):
                                post_tensor[post_idx] = 1
                            if displaced_player.player.has_general_stock("circle") and (not post.required_shape or post.required_shape == "circle"):
                                post_tensor[MAX_POSTS + post_idx] = 1
                        elif has_pieces_in_personal_supply:
                            if displaced_player.player.has_personal_supply("square") and (not post.required_shape or post.required_shape == "square"):
                                post_tensor[post_idx] = 1
                            if displaced_player.player.has_personal_supply("circle") and (not post.required_shape or post.required_shape == "circle"):
                                post_tensor[MAX_POSTS + post_idx] = 1
                        # else:
                        #     print (f"2If displaced player has an empty GS and PS, eventually need to handle moving pieces.")
                    else:
                      print(f"DISPLACE MASK ERROR: Displaced Shape: {displaced_player.displaced_shape}, index: {post_idx}")  
                # else:
                #     print(f"DISPLACE MASK ERROR: Post Type {post.required_shape}, index: {post_idx}")
            #handle BM Move any2 or #handle BM Move 3:
            elif game.waiting_for_bm_move_any_2:
                if post.is_owned() and current_player.pieces_to_pickup > 0:
                    post_tensor[post_idx] = 1
                elif not post.is_owned() and current_player.pieces_to_pickup == 0:
                    post_tensor[post_idx] = 1

            elif game.waiting_for_place2_in_scotland_or_wales:
                if post.region == "Scotland" or post.region == "Wales":
                    if not post.is_owned() and current_player.pieces_to_place > 0:
                        post_tensor[post_idx] = 1
            
            elif game.waiting_for_bm_move3:
                if post.is_owned() and post.owner != current_player and current_player.pieces_to_pickup > 0:
                    post_tensor[post_idx] = 1
                elif not post.is_owned() and current_player.pieces_to_pickup == 0:
                    post_tensor[post_idx] = 1
            elif game.waiting_for_bm_tribute_trading_post:
                post_tensor[post_idx] = 1
            elif game.waiting_for_bm_block_trade_route:
                post_tensor[post_idx] = 1

            else:
                # # Now use this condition in your if statement
                # if current_player.holding_pieces:
                #     if is_post_empty:
                #         shape_to_place, owner_to_place, origin_region = current_player.holding_pieces[0]
                #         if current_player.is_valid_region_transition(origin_region, post.region):
                #             if shape_to_place == 'square' and (not post.required_shape or post.required_shape == 'square'):
                #                 post_tensor[post_idx] = 1
                #             elif shape_to_place == 'circle' and (not post.required_shape or post.required_shape == 'circle'):
                #                 post_tensor[MAX_POSTS + post_idx] = 1  # Offset for circle posts
                #         # else:
                #             # Invalid action scenario, handle accordingly
                #             # print(f"Invalid action: Cannot move {shape_to_place} piece from {origin_region} to {post.region}, or shape mismatch.")
                #     elif is_post_owned and post.owner == current_player and current_player.pieces_to_pickup > 0:
                #         if len(current_player.holding_pieces) < current_player.book:
                #             post_tensor[post_idx] = 1
                #     # else:
                #     #     print(f"MOVE ERROR: Next Shape to place: {shape_to_place}, shape: {post.required_shape}")
                #     #     print(f"MOVE ERROR: Pieces held {len(current_player.holding_pieces)}")

                # # MOVE - if post is owned by current_player:
                # elif is_post_owned and post.owner == current_player:
                #     if len(current_player.holding_pieces) < current_player.book:
                #         post_tensor[post_idx] = 1
                ## Claim post action: check if the post is empty and region is valid
                if is_post_empty and check_brown_blue_priv(game, route):
                    if current_player.personal_supply_squares > 0 and (not post.required_shape or post.required_shape == "square"):
                        # print(f"MASK OK - claim empty post {post_idx} with square")
                        post_tensor[post_idx] = 1
                    if current_player.personal_supply_circles > 0 and (not post.required_shape or post.required_shape == "circle"):
                        # print(f"MASK OK - claim empty post {post_idx} with circle")
                        post_tensor[MAX_POSTS + post_idx] = 1  # Offset by 121 for circle posts
                # DISPLACE - if post is owned by a different player:
                elif is_post_owned and post.owner != current_player and check_brown_blue_priv(game, route) and can_displace:
                    # Calculate the cost to displace based on the shape of the post's piece
                    displacement_cost = 2 if post.owner_piece_shape == "square" else 3
                    
                    # Check if the player has enough pieces to displace
                    if current_player.personal_supply_squares + current_player.personal_supply_circles >= displacement_cost:
                        # If the player can displace this post, mark it as a valid action
                        if post.required_shape == "square" or post.required_shape is None and current_player.personal_supply_squares > 0:
                            # print(f"MASK OK - displace taken post {post_idx} with square - owned by {COLOR_NAMES[post.owner.color]}")
                            post_tensor[post_idx] = 1
                        if post.required_shape == "circle" or post.required_shape is None and current_player.personal_supply_circles > 0:
                            # print(f"MASK OK - displace taken post {post_idx} with circle - owned by {COLOR_NAMES[post.owner.color]}")
                            post_tensor[MAX_POSTS + post_idx] = 1  # Offset by 121 for circle posts
                # else:
                #     print (f"Invalid scenario detected. Post Info - Circle Color: {COLOR_NAMES[post.circle_color]}, Square Color: {COLOR_NAMES[post.square_color]}, Owner: {post.owner}, Region: {post.region}, ReqShape: {post.required_shape}")
            post_idx += 1
    # print(f"mask_post_action: {post_tensor}")

    return post_tensor

#should return - 40+80+160=280
def mask_claim_route(game):
    max_num_routes = MAX_ROUTES  # Maximum number of routes
    two_cities_per_route = 2  # Maximum number of routes per city
    max_upgrades_per_city = 2  # Maximum upgrades per city

    # Initializing tensors for different actions
    claim_route_for_points_tensor = torch.zeros(max_num_routes, device=device, dtype=torch.uint8)
    claim_route_for_office_tensor = torch.zeros(max_num_routes * two_cities_per_route, device=device, dtype=torch.uint8)
    claim_route_for_upgrade_tensor = torch.zeros(max_num_routes * two_cities_per_route * max_upgrades_per_city, device=device, dtype=torch.uint8)

    if game.current_player.actions_remaining == 0 or game.current_player.holding_pieces or \
       check_if_any_post_BM_flag_set(game) or check_if_any_action_BM_flag_set(game):
        claim_route_tensor = torch.cat([claim_route_for_points_tensor, claim_route_for_office_tensor, claim_route_for_upgrade_tensor])
        return claim_route_tensor

    route_idx = 0
    for route in game.selected_map.routes:
        if route.is_controlled_by(game.current_player):
            # Claim route for points
            claim_route_for_points_tensor[route_idx] = 1

            for city_idx, city in enumerate(route.cities):
                # Calculate indices for tensor
                base_index_office = route_idx * two_cities_per_route + city_idx
                
                if city.has_empty_office():
                    next_open_office_color = city.get_next_open_office_color()
                    if game.current_player.player_can_claim_office(next_open_office_color) and city.color != DARK_GREEN:
                        if city.has_required_piece_shape(game.current_player, route):
                            claim_route_for_office_tensor[base_index_office] = 1
                            # print(f"{route_idx} City: {city.name}")
                            # print(f"{city_idx} Route between {route.cities[0].name} and {route.cities[1].name}")
                            # for i, post in enumerate(route.posts):
                            #     if post.owner:
                            #         print(f"[{i}] Post Owner: {COLOR_NAMES[post.owner.color]}, Post Owner Piece Shape: {post.owner_piece_shape}")
                            #     else:
                            #         print(f"[{i}] Post Owner: None, Post Owner Piece Shape: {post.owner_piece_shape}")

                # Check for upgrade options
                if city.upgrade_city_type:
                    for upgrade_idx, upgrade in enumerate(city.upgrade_city_type):
                        if upgrade_idx < max_upgrades_per_city:
                            # Calculate the unique index for this upgrade option
                            action_index_upgrade = (route_idx * two_cities_per_route * max_upgrades_per_city) + (city_idx * max_upgrades_per_city) + upgrade_idx
                            claim_route_for_upgrade_tensor[action_index_upgrade] = 1
                            # print(f"action_index_upgrade: {action_index_upgrade}")
                            # print(f"{route_idx} City: {city.name}")
                            # print(f"{city_idx} Route between {route.cities[0].name} and {route.cities[1].name}")
                            # for i, post in enumerate(route.posts):
                            #     if post.owner:
                            #         print(f"[{i}] Post Owner: {COLOR_NAMES[post.owner.color]}, Post Owner Piece Shape: {post.owner_piece_shape}")
                            #     else:
                            #         print(f"[{i}] Post Owner: None, Post Owner Piece Shape: {post.owner_piece_shape}")

        route_idx += 1

    # Concatenate tensors to form a single tensor representing all claim route actions
    claim_route_tensor = torch.cat([claim_route_for_points_tensor, claim_route_for_office_tensor, claim_route_for_upgrade_tensor])
    # print(f"{claim_route_for_upgrade_tensor}")
    return claim_route_tensor

def get_city_index(city, game):
    # Implement this function to return the index of a city in the game's city list
    return game.selected_map.cities.index(city)

def mask_income_actions(game):
    income_tensor = torch.zeros(5, device=device, dtype=torch.uint8)  # 5 options for income actions

    if game.current_player.actions_remaining == 0 or game.current_player.holding_pieces or \
       check_if_any_post_BM_flag_set(game) or check_if_any_action_BM_flag_set(game):
        return income_tensor

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
        "ExchangeBonusMarker": 5,
        "Tribute4EstablishingTP": 6,
        "BlockTradeRoute": 7
    }

    if game.waiting_for_bm_exchange_bm:
        for player in game.players:
            if player != game.current_player and player.has_bonus_markers():
                for bm in player.bonus_markers:
                    bm_index = bm_mapping.get(bm.type)
                    if bm_index is not None:
                        bm_tensor[bm_index] = 1
    else:
        for bm in game.current_player.bonus_markers:
            bm_index = bm_mapping.get(bm.type)
            if bm_index is not None:
                #TODO: Check if the BM is valid to use
                if bm.type == "SwapOffice":
                    for city in game.selected_map.cities:
                        if city.check_if_eligible_to_swap_offices(game.current_player):
                            bm_tensor[bm_index] = 1
                #todo check if can exchange bm with no one else having a used bm
                else:
                    bm_tensor[bm_index] = 1

    return bm_tensor

def mask_buy_tile(game):
    current_player = game.current_player

    buy_tile_tensor = torch.zeros(8, device=device, dtype=torch.uint8)  # 6 possible tiles to buy and 8 BMs
    tile_mapping = {
        'DisplaceAnywhere': 1,
        '+1Action': 2,
        '+1IncomeIfOthersIncome': 3,
        '+1DisplacedPiece': 4,
        '+4PtsPerOwnedCity': 5,
        '+7PtsPerCompletedAbility': 6
    }
    bm_mapping = {
        "SwapOffice": 0,
        "Move3": 1,
        "UpgradeAbility": 2,
        "3Actions": 3,
        "4Actions": 4,
        "ExchangeBonusMarker": 5,
        "Tribute4EstablishingTP": 6,
        "BlockTradeRoute": 7
    }

    if game.waiting_for_buy_tile_with_bm == True:
        for bm in current_player.bonus_markers:
            bm_index = bm_mapping.get(bm.type)
            if bm_index is not None:
                buy_tile_tensor[bm_index] = 1


    if current_player.actions_remaining == current_player.actions and len(current_player.bonus_markers) >= 2:
        for tile in game.tile_pool:
            buy_tile_tensor[tile_mapping.get(tile.type)] = 1

    return buy_tile_tensor

def mask_replace_bm(game):
    max_num_routes = 40  # Maximum number of routes
    replace_bm_tensor = torch.zeros(max_num_routes, device=device, dtype=torch.uint8)  # Tensor for replace bonus marker actions

    # print(f"mask_replace_bm: actions_remaining: {game.current_player.actions_remaining}, ending_turn: {game.current_player.ending_turn}, replace_bonus_marker: {game.replace_bonus_marker}")
    # Only allow replacing bonus markers if the player has no actions remaining and needs to replace a bonus marker
    if game.current_player.actions_remaining == 0 and game.replace_bonus_marker > 0 and game.current_player.ending_turn:
        for route_idx, route in enumerate(game.selected_map.routes):
            if not (route.bonus_marker or route.permanent_bonus_marker) and \
               not route.has_tradesmen() and \
               route.has_empty_office_in_cities():
                replace_bm_tensor[route_idx] = 1

    return replace_bm_tensor

def mask_bm_city_actions(game):
    bm_city_tensor = torch.zeros(MAX_CITIES, device=device, dtype=torch.uint8)  # 30 possible cities to claim a green city or swap office
    if not game.waiting_for_bm_swap_office and not game.waiting_for_bm_green_city:
        return bm_city_tensor
    
    city_idx = 0
    for city in game.selected_map.cities:
        if game.waiting_for_bm_swap_office:
            if city.check_if_eligible_to_swap_offices(game.current_player):
                bm_city_tensor[city_idx] = 1
        elif game.waiting_for_bm_green_city:
            if city.color == DARK_GREEN:
                print (f"Valid City to claim green city: {city.name}, {city_idx}")
                bm_city_tensor[city_idx] = 1
        city_idx += 1
    
    return bm_city_tensor

def mask_bm_upgrade_ability(game):
    bm_upgrade_tensor = torch.zeros(5, device=device, dtype=torch.uint8)  # 5 possible upgrades

    if not game.waiting_for_bm_upgrade_ability:
        return bm_upgrade_tensor
    
    upgrade_idx = 0
    for upgrade_city in game.selected_map.upgrade_cities:
        upgrade_type = upgrade_city.upgrade_type
        current_player_value = getattr(game.current_player, upgrade_type.lower())
        max_value = UPGRADE_MAX_VALUES.get(upgrade_type)

        if current_player_value != max_value:
            bm_upgrade_tensor[upgrade_idx] = 1

        upgrade_idx += 1
    return bm_upgrade_tensor

def mask_end_turn(game):
    end_turn_tensor = torch.zeros(1, device=device, dtype=torch.uint8)

    # print(f"mask_end_turn: actions_remaining: {game.current_player.actions_remaining}, ending_turn: {game.current_player.ending_turn}, holding_pieces: {game.current_player.holding_pieces}")
    if game.current_player.actions_remaining > 0 or game.current_player.ending_turn or game.current_player.holding_pieces or \
       check_if_any_post_BM_flag_set(game) or check_if_any_action_BM_flag_set(game):
        # print("mask_end_turn: returning 0's")
        return end_turn_tensor

    if (check_if_player_has_usable_BMs(game) and game.current_player.actions_remaining == 0) or \
        (game.replace_bonus_marker > 0 and game.current_player.actions_remaining == 0):
        print(f"mask_end_turn: returning valid to end turn")
        print(f"game.replace_bonus_marker = {game.replace_bonus_marker}, game.current_player.ending_turn = {game.current_player.ending_turn}")
        end_turn_tensor[0] = 1

    if game.replace_bonus_marker > 0:
        end_turn_tensor[0] = 1
    
    return end_turn_tensor

def check_if_player_has_usable_BMs(game):
    if not game.current_player.bonus_markers:
        return False

    for bm in game.current_player.bonus_markers:
        if bm.type == 'PlaceAdjacent':
            continue
        elif bm.type == 'SwapOffice':
            # Check if there's a city where the player is eligible to swap offices
            swap_office_possible = False
            for city in game.selected_map.cities:
                if city.check_if_eligible_to_swap_offices(game.current_player):
                    swap_office_possible = True
                    break
            if not swap_office_possible:
                continue
        elif bm.type == 'ExchangeBonusMarker':
            # Check if there's a player to exchange bonus markers with
            exchange_possible = False
            for player in game.players:
                if player != game.current_player and player.used_bonus_markers:
                    exchange_possible = True
                    break
            if not exchange_possible:
                continue
        else:
            return True

    return False

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