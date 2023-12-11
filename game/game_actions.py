import sys
from map_data.constants import COLOR_NAMES, WHITE, GREEN, BLUE, PURPLE, RED, YELLOW, TAN, DARK_GREEN

def claim_post_action(game, route, post, piece_to_play):
    player = game.current_player

    if not game.check_brown_blue_priv(route):
        print(f"CLAIM ERROR - Incorrect Privilige to claim in brown or blue")
        return
    else:
        if route.region == "Wales":
            game.current_player.brown_priv_count -= 1
        elif route.region == "Scotland":
            game.current_player.blue_priv_count -= 1
    
    if (player.personal_supply_squares + player.personal_supply_circles) == 0:
        print("CLAIM ERROR - No pieces left to place!")
        print(f"personal_supply_squares={player.personal_supply_squares} personal_supply_circles={player.personal_supply_circles}")
        sys.exit()
        return
    
    if route.has_bonus_marker:
        player.reward += player.reward_structure.post_with_bm
    elif route.has_permanent_bm_type:
        player.reward += player.reward_structure.post_with_perm_bm
    if route.cities[0].upgrade_city_type or route.cities[1].upgrade_city_type:
        player.reward += player.reward_structure.post_adjacent_to_upgrade_city
    else:
        player.reward += player.reward_structure.post_with_nothing

    city_names = ' and '.join([city.name for city in route.cities])
    region_info = f" in {route.region}" if route.region else ""

    if piece_to_play == "square" and player.personal_supply_squares > 0:
        post.claim(player, "square")
        print(f"[{player.actions_remaining}] {COLOR_NAMES[player.color]} placed a 'square' between {city_names}{region_info}")
        player.actions_remaining -= 1
        player.personal_supply_squares -= 1

    elif piece_to_play == "circle" and player.personal_supply_circles > 0:
        post.claim(player, "circle")
        print(f"[{player.actions_remaining}] {COLOR_NAMES[player.color]} placed a 'circle' between {city_names}{region_info}")
        player.actions_remaining -= 1
        player.personal_supply_circles -= 1
    else:
        print("ERROR in claim_post_action")
        sys.exit()

def displace_action(game, post, route, displacing_piece_shape):
    current_player = game.current_player

    # Determine the shape of the piece that will be displaced
    displaced_piece_shape = post.owner_piece_shape
    current_displaced_player = post.owner
    game.active_player = current_displaced_player.order-1

    # Calculate the cost to displace
    cost = 2 if displaced_piece_shape == "square" else 3

    # Check if the current player has enough tradesmen
    if current_player.personal_supply_squares + current_player.personal_supply_circles < cost:
        print(f"DISPLACE ERROR - Not enough tradesmen! PS Squares: {current_player.personal_supply_squares} PS Circles: {current_player.personal_supply_circles}")
        return  # Not enough tradesmen, action cannot be performed

    # Before displacing with a square or circle, check if the current player has that shape in their personal supply
    if not current_player.has_personal_supply(displacing_piece_shape):
        print(f"DISPLACE ERROR - Cannot displace with a {displaced_piece_shape} as current_player has PS Squares: {current_player.personal_supply_squares} PS Circles: {current_player.personal_supply_circles}")
        return  # Invalid move as there's no circle in the personal supply
    
    if not game.check_brown_blue_priv(route):
        print(f"DISPLACE ERROR - Incorrect Privilige to displace in brown or blue")
        return
    else:
        if route.region == "Wales":
            game.current_player.brown_priv_count -= 1
        elif route.region == "Scotland":
            game.current_player.blue_priv_count -= 1
    
    if route.has_bonus_marker:
        current_player.reward += current_player.reward_structure.post_with_bm-2
    elif route.has_permanent_bm_type:
        current_player.reward += current_player.reward_structure.post_with_perm_bm-2
    if route.cities[0].upgrade_city_type or route.cities[1].upgrade_city_type:
        current_player.reward += current_player.reward_structure.post_adjacent_to_upgrade_city-2
    else:
        current_player.reward += current_player.reward_structure.post_with_nothing

    # Handle the piece being placed
    if displacing_piece_shape == "square" and current_player.personal_supply_squares > 0:
        current_player.personal_supply_squares -= 1
        post.circle_color = TAN
        post.square_color = current_player.color
    elif displacing_piece_shape == "circle" and current_player.personal_supply_circles > 0:
        current_player.personal_supply_circles -= 1
        post.circle_color = current_player.color
        post.square_color = TAN
    else:
        print("ERROR in displace_action")
        sys.exit()

    # Handle the cost of displacement with priority to squares
    squares_to_pay = min(current_player.personal_supply_squares, cost - 1)  # Subtract 1 for the piece being placed
    circles_to_pay = cost - 1 - squares_to_pay

    current_player.personal_supply_squares -= squares_to_pay
    current_player.personal_supply_circles -= circles_to_pay

    current_player.general_stock_squares += squares_to_pay
    current_player.general_stock_circles += circles_to_pay

    if current_player.personal_supply_squares < 0 or current_player.personal_supply_circles < 0:
        print(f"displace_action - personal_supply_squares={current_player.personal_supply_squares} personal_supply_circles={current_player.personal_supply_circles}")
        sys.exit()

    post.owner_piece_shape = displacing_piece_shape
    post.owner = current_player

    # Find empty posts on adjacent routes
    game.all_empty_posts = gather_empty_posts(route)
    for post in game.all_empty_posts:
        post.valid_post_to_displace_to()
    # Check conditions based on displaced_piece_shape and number of game.all_empty_posts
    if ((displaced_piece_shape == "square" and len(game.all_empty_posts) == 1) or
        (displaced_piece_shape == "circle" and 1 <= len(game.all_empty_posts) <= 2)):
        game.original_route_of_displacement = route

    game.waiting_for_displaced_player = True
    game.displaced_player.populate_displaced_player(current_displaced_player, displaced_piece_shape)
    print(f"Waiting for Displaced Player {COLOR_NAMES[game.displaced_player.player.color]} to place {game.displaced_player.total_pieces_to_place} tradesmen (circle or square) from their general_stock, one must be {game.displaced_player.displaced_shape}.")

def gather_empty_posts(start_route):
    if not start_route:
        print("Error: start_route is None in gather_empty_posts")
        return []

    visited_routes = [start_route]  # Mark the start route as visited immediately
    queue = get_adjacent_routes(start_route, start_route.region)  # Start with the routes adjacent to the given route, considering the region
    
    while queue:
        level_size = len(queue)
        empty_posts = []
        next_level_routes = []  # Routes for the next level

        for i in range(level_size):
            current_route = queue.pop(0)
            if current_route in visited_routes:
                continue
            visited_routes.append(current_route)

            adjacent_routes = get_adjacent_routes(current_route, start_route.region)
            for route in adjacent_routes:
                if route not in visited_routes and route not in next_level_routes:
                    next_level_routes.append(route)

            for post in current_route.posts:
                if not post.is_owned():
                    empty_posts.append(post)

        if empty_posts:
            return empty_posts

        queue.extend(next_level_routes)

    return []

def get_adjacent_routes(current_route, start_route_region):
    if not current_route:
        print("Error: current_route is None in get_adjacent_routes")
        return []

    adjacent_routes = []
    for city in current_route.cities:
        for adjacent_route in city.routes:
            if adjacent_route != current_route and adjacent_route not in adjacent_routes:
                if valid_region_transition(start_route_region, adjacent_route.region):
                    adjacent_routes.append(adjacent_route)
    return adjacent_routes

def valid_region_transition(start_region, target_region):
    if start_region is None:
        # Routes with no specific region can consider only routes with no specific region
        return target_region is None
    elif start_region in ["Scotland", "Wales"]:
        # Brown and blue can consider their own and None regions
        return target_region in [start_region, None]
    return False

def move_action(game, route, post, shape):
    player = game.current_player

     # If no post was found, simply return without doing anything
    if post is None:
        print("No post found!")
        return
    
    # player.reward += 0.1

    if ((game.waiting_for_bm_move3 and post.is_owned() and post.owner != player) or
        (game.waiting_for_bm_move_any_2 and post.is_owned())):
        if player.pieces_to_place > 0:
            player.pick_up_piece(post)
            print(f"[{player.actions_remaining}] {COLOR_NAMES[player.color]} BM - picked up a piece")
    # If the player is picking up pieces
    elif post.owner == player:
        # If we have not started a move yet, start one
        if player.pieces_to_place == 0:
            player.start_move()
        # Attempt to pick up a piece if within the pieces to move limit (book)
        player.pick_up_piece(post)

        if route.has_bonus_marker:
            player.reward -= player.reward_structure.post_with_bm
        elif route.has_permanent_bm_type:
            player.reward -= player.reward_structure.post_with_perm_bm
        if route.cities[0].upgrade_city_type or route.cities[1].upgrade_city_type:
            player.reward -= player.reward_structure.post_adjacent_to_upgrade_city
        # else:
        #     player.reward -= player.reward_structure.post_with_nothing

        print(f"[{player.actions_remaining}] {COLOR_NAMES[player.color]} MOVE - picked up a piece")

    # If the player has pieces in hand to place
    elif player.holding_pieces:
        if not post.is_owned():
            shape_to_place, owner_to_place, origin_region = player.holding_pieces[0]
            player.place_piece(post, shape)

            if owner_to_place == player:
                if route.has_bonus_marker:
                    player.reward += player.reward_structure.post_with_bm
                elif route.has_permanent_bm_type:
                    player.reward += player.reward_structure.post_with_perm_bm
                if route.cities[0].upgrade_city_type or route.cities[1].upgrade_city_type:
                    player.reward += player.reward_structure.post_adjacent_to_upgrade_city
                # else:
                #     player.reward -= player.reward_structure.post_with_nothing
            
            # If no pieces are left to place, finish the move
            if not player.holding_pieces:
                if player.pieces_to_place > 0:
                    player.reward -= 10
                # else:
                #     player.reward += 10
                player.finish_move()
                # Deduct an action if it's a standard move (not a BM Move3 or MoveAny2)
                if not (game.waiting_for_bm_move3 or game.waiting_for_bm_move_any_2):
                    print(f"[{player.actions_remaining}] {COLOR_NAMES[player.color]} finished their move action.")
                    player.actions_remaining -= 1
        else:
            print("Cannot place a piece here. The post is already occupied.")
    else:
        print(f"ERROR: Holding pieces {len(player.holding_pieces)}, shape: {post.shape}")

def displace_claim(game, post, desired_shape):
    displaced_player = game.displaced_player

    # Check if the player is forced to use the displaced shape
    must_use_displaced_piece = False
    if not displaced_player.played_displaced_shape:
        if displaced_player.displaced_shape == "circle" and displaced_player.total_pieces_to_place == 1:
            must_use_displaced_piece = True
        elif displaced_player.displaced_shape == "square" and displaced_player.total_pieces_to_place == 1:
            must_use_displaced_piece = True

    if must_use_displaced_piece and desired_shape != displaced_player.displaced_shape:
        print("Invalid action. Must use the displaced piece.")
        return

    wants_to_use_displaced_piece = (not displaced_player.played_displaced_shape) and (desired_shape == displaced_player.displaced_shape)
    
    if wants_to_use_displaced_piece:
        print(f"Attempting to place the Displaced Piece '{desired_shape}' while Displaced Shape has NOT been played yet.")
        displace_to(game, post, desired_shape, use_displaced_piece=True)
    else:
        displace_to(game, post, desired_shape)
    
    if game.displaced_player.all_pieces_placed():
        for post in game.all_empty_posts:
            post.reset_post()
        game.all_empty_posts.clear()
        game.original_route_of_displacement = None
        game.displaced_player.reset_displaced_player()
        game.waiting_for_displaced_player = False
        print("No longer waiting for the displaced player.")
        game.current_player.actions_remaining -= 1
        game.active_player = game.current_player.order-1
        # game.switch_player_if_needed()
    elif not game.all_empty_posts:
        print("No empty posts found initially. Searching for adjacent routes...")  # Debugging log
        game.all_empty_posts = gather_empty_posts(game.original_route_of_displacement)
        if not game.all_empty_posts:
            print("ERROR::No empty posts found in adjacent routes either!")  # Debugging log
            sys.exit()
        
        post_count = 0  # Counter for the number of posts processed
        for post in game.all_empty_posts:
            post.valid_post_to_displace_to()
            post_count += 1
        print(f"Processed {post_count} posts.")  # Debugging log

def displace_to(game, post, shape, use_displaced_piece=False):
    displaced_player = game.displaced_player
    if use_displaced_piece:
        print(f"Placed the displaced piece {shape} - no affect to the GS or PS.")
        claim_and_update(game, post, shape, use_displaced_piece=True)
    else:
        if displaced_player.has_general_stock(shape):
            print(f"Placed a {shape} from general_stock, because use_displaced is false.")
            claim_and_update(game, post, shape)
        elif displaced_player.is_general_stock_empty() and displaced_player.has_personal_supply(shape):
            print(f"Placed a {shape} from personal_supply, because use_displaced is false and general stock is empty.")
            claim_and_update(game, post, shape, from_personal_supply=True)
        else:
            print(f"Cannot place a {shape} because the general stock and personal supply are empty.")

def claim_and_update(game, post, shape, use_displaced_piece=False, from_personal_supply=False):
    displaced_player = game.displaced_player
    post.claim(displaced_player.player, shape)
    game.all_empty_posts.remove(post)
    
    if use_displaced_piece:
        displaced_player.played_displaced_shape = True
    elif not from_personal_supply:
        if shape == "square":
            displaced_player.player.general_stock_squares -= 1
        else:
            displaced_player.player.general_stock_circles -= 1
    else:
        if shape == "square":
            displaced_player.player.personal_supply_squares -= 1
        else:
            displaced_player.player.personal_supply_circles -= 1
    
    if displaced_player.player.personal_supply_squares < 0 or displaced_player.player.personal_supply_circles < 0:
        print(f"claim_and_update - personal_supply_squares={displaced_player.player.personal_supply_squares} personal_supply_circles={displaced_player.player.personal_supply_circles}")
        sys.exit()
    
    displaced_player.total_pieces_to_place -= 1

def assign_new_bonus_marker_on_route(game, route):
    if not route:
        print("Invalid BM Placement: Clicked position does not correspond to any route.")
        return

    if route.bonus_marker or route.permanent_bonus_marker:
        print(f"Invalid BM Placement: Route between {route.cities[0].name} and {route.cities[1].name} already has a bonus marker.")
        return

    if route.has_tradesmen():
        print(f"Invalid BM Placement: Route between {route.cities[0].name} and {route.cities[1].name} has tradesmen on it.")
        return

    if not route.has_empty_office_in_cities():
        print(f"Invalid BM Placement: Both cities at the ends of the route between {route.cities[0].name} and {route.cities[1].name} have no empty offices.")
        return

    if game.selected_map.bonus_marker_pool:
        bm_type = game.selected_map.bonus_marker_pool.pop()
        route.assign_map_new_bonus_marker(bm_type)  # Create a new BonusMarker instance with the type
        print(f"Bonus marker '{bm_type}' has been placed on the route between {route.cities[0].name} and {route.cities[1].name}.")
        game.replace_bonus_marker -= 1
    else:
        print("No action taken: No bonus markers available in the pool.")

def score_route(route):
    # Allocate points
    for city in route.cities:
        player = city.get_controller()
        if player is not None:
            player.score += 1
            player.reward += player.reward_structure.route_complete_got_points
            print(f"Player {COLOR_NAMES[player.color]} scored for controlling {city.name}, total score: {player.score}")

def claim_route_for_office(game, city, route):
    current_player = game.current_player
    next_open_office_color = city.get_next_open_office_color()
    # print(f"Next open office color: {next_open_office_color}")

    if current_player.player_can_claim_office(next_open_office_color) and city.color != DARK_GREEN:
        if not city.has_required_piece_shape(current_player, route):
            required_shape = city.get_next_open_office_shape()
            print(f"{COLOR_NAMES[current_player.color]} tried to claim an office in {city.name} but doesn't have the required {required_shape} shape on the route.")
        else:
            current_player.reward += current_player.reward_structure.city_claim_office
            score_route(route)
            placed_piece_shape = city.get_next_open_office_shape()
            print(f"[{current_player.actions_remaining}] {COLOR_NAMES[current_player.color]} placed a {placed_piece_shape.upper()} into an office of {city.name}")
            city.update_next_open_office_ownership(game)
            finalize_route_claim(game, route, placed_piece_shape)
    elif 'PlaceAdjacent' in (bm.type for bm in current_player.bonus_markers):
        current_player.reward += current_player.reward_structure.bm_place_adjacent
        score_route(route)
        city.claim_office_with_bonus_marker(current_player)
        print(f"[{current_player.actions_remaining}] {COLOR_NAMES[current_player.color]} placed a square into a NEW office of {city.name}.")
        finalize_route_claim(game, route, "square")
    elif city.color == DARK_GREEN:
        print(f"{COLOR_NAMES[current_player.color]} cannot claim a GREEN City ({city.name}) without a PlaceAdjacent BM.")
    else:
        print(f"{COLOR_NAMES[current_player.color]} doesn't have the correct privilege - {current_player.privilege} - to claim an office in {city.name}.")

def claim_route_for_upgrade(game, city, route, upgrade_choice):
    current_player = game.current_player
    specialprestigepoints_city = game.selected_map.specialprestigepoints

    if "SpecialPrestigePoints" in city.upgrade_city_type and route.contains_a_circle():
        if specialprestigepoints_city.claim_highest_prestige(route):
            current_player.reward += current_player.reward_structure.upgraded_bonus_points
            score_route(route)
            finalize_route_claim(game, route, "circle")
    elif any(upgrade_type in ["Keys", "Privilege", "Book", "Actions", "Bank"] for upgrade_type in city.upgrade_city_type):
        if upgrade_choice and current_player.perform_upgrade(upgrade_choice):
            print(f"[{current_player.actions_remaining}] {COLOR_NAMES[current_player.color]} upgraded {upgrade_choice}!")
            current_player.reward += current_player.reward_structure.upgraded_keys
            score_route(route)
            finalize_route_claim(game, route)
    else:
        print(f"Upgrade not detected")

def claim_route_for_points(game, route):
    current_player = game.current_player
    print(f"[{current_player.actions_remaining}] {COLOR_NAMES[current_player.color]} claimed a route for Points/BM!")
    score_route(route)
    finalize_route_claim(game, route)

def finalize_route_claim(game, route, placed_piece_shape=None):
    reset_pieces = update_stock_and_reset(route, game.current_player, placed_piece_shape)
    handle_bonus_marker(game, game.current_player, route, reset_pieces)
    game.current_player.actions_remaining -= 1
    game.check_for_east_west_connection()
    # game.switch_player_if_needed()

def handle_bonus_marker(game, player, route, reset_pieces):
    if route.bonus_marker:
        player.reward += player.reward_structure.route_complete_receive_bm
        route.bonus_marker.owner = player
        player.bonus_markers.append(route.bonus_marker)
        route.bonus_marker = None
        game.replace_bonus_marker += 1
    elif route.permanent_bonus_marker:
        player.reward += player.reward_structure.route_complete_perm_bm
        perm_bm_type = route.permanent_bonus_marker.type
        print(f"Waiting for Player to handle {route.permanent_bonus_marker.type} BM")
        if perm_bm_type == 'MoveAny2':
            game.current_player.pieces_to_place = 2
            game.waiting_for_bm_move_any_2 = True
            print(f"BM: Please pick up, upto {game.current_player.pieces_to_place} pieces to move!")
        elif perm_bm_type == 'Place2TradesmenFromRoute':
            game.current_player.pieces_to_place = 2
            game.current_player.holding_pieces = reset_pieces
            print(f"BM: Please place {game.current_player.pieces_to_place} pieces of {reset_pieces} on valid posts!")
            game.waiting_for_bm_move_any_2 = True
        elif perm_bm_type == '+1Priv':
            game.current_player.upgrade_privilege()
        # handle_permanent_bonus_marker(route.permanent_bonus_marker.type, reset_pieces)

def update_stock_and_reset(route, player, placed_piece_shape=None):
    """Update player's general stock based on pieces on the route and reset those posts."""
    circles_on_route = sum(1 for post in route.posts if post.owner == player and post.owner_piece_shape == "circle")
    squares_on_route = sum(1 for post in route.posts if post.owner == player and post.owner_piece_shape == "square")

    if placed_piece_shape == "circle":
        circles_on_route -= 1
    elif placed_piece_shape == "square":
        squares_on_route -= 1

    # Update the player's general supply
    player.general_stock_circles += circles_on_route
    player.general_stock_squares += squares_on_route

    # Create reset_pieces list
    reset_pieces = []
    for post in route.posts:
        if post.owner == player:
            reset_pieces.append((post.owner_piece_shape, player, post.region))

    # Remove one piece of placed_piece_shape (if not None) from reset_pieces
    if placed_piece_shape:
        for i, piece in enumerate(reset_pieces):
            if piece[0] == placed_piece_shape:
                reset_pieces.pop(i)
                break

    for post in route.posts:
        if post.owner == player:
            post.reset_post()

    return reset_pieces