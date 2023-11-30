import sys
from map_data.constants import COLOR_NAMES, WHITE, GREEN, BLUE, PURPLE, RED, YELLOW, TAN

def claim_post_action(game, route, post, piece_to_play):
    player = game.current_player

    if not game.check_brown_blue_priv(route):
        return

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

def displace_action(game, post, route, displacing_piece_shape):

    # Determine the shape of the piece that will be displaced
    displaced_piece_shape = post.owner_piece_shape
    current_displaced_player = post.owner

    # Calculate the cost to displace
    cost = 2 if displaced_piece_shape == "square" else 3

    # Check if the current player has enough tradesmen
    if game.current_player.personal_supply_squares + game.current_player.personal_supply_circles < cost:
        return  # Not enough tradesmen, action cannot be performed

    # Before displacing with a square, check if the current player has a square in their personal supply
    if displacing_piece_shape == "square" and game.current_player.personal_supply_squares == 0:
        return  # Invalid move as there's no square in the personal supply

    # Before displacing with a circle, check if the current player has a circle in their personal supply
    if displacing_piece_shape == "circle" and game.current_player.personal_supply_circles == 0:
        return  # Invalid move as there's no circle in the personal supply
    
    if not game.check_brown_blue_priv(route):
        return

    # Handle the cost of displacement with priority to squares
    squares_to_pay = min(game.current_player.personal_supply_squares, cost - 1)  # Subtract 1 for the piece being placed
    circles_to_pay = cost - 1 - squares_to_pay

    game.current_player.personal_supply_squares -= squares_to_pay
    game.current_player.personal_supply_circles -= circles_to_pay

    game.current_player.general_stock_squares += squares_to_pay
    game.current_player.general_stock_circles += circles_to_pay

    # Handle the piece being placed
    if displacing_piece_shape == "square":
        game.current_player.personal_supply_squares -= 1
        post.circle_color = TAN
        post.square_color = game.current_player.color
    elif displacing_piece_shape == "circle":
        game.current_player.personal_supply_circles -= 1
        post.circle_color = game.current_player.color
        post.square_color = TAN
    else:
        sys.exit()

    post.owner_piece_shape = displacing_piece_shape
    post.owner = game.current_player

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

def move_action(game, post, shape):
    player = game.current_player

     # If no post was found, simply return without doing anything
    if post is None:
        return

    if ((game.waiting_for_bm_move3 and post.is_owned() and post.owner != player) or
        (game.waiting_for_bm_move_any_2 and post.is_owned())):
        if player.pieces_to_place > 0:
            player.pick_up_piece(post)
    # If the player is picking up pieces
    elif post.owner == player:
        # If we have not started a move yet, start one
        if player.pieces_to_place == 0:
            player.start_move()
        # Attempt to pick up a piece if within the pieces to move limit (book)
        player.pick_up_piece(post)

    # If the player has pieces in hand to place
    elif player.holding_pieces:
        if not post.is_owned():
            player.place_piece(post, shape)

            # If no pieces are left to place, finish the move
            if not player.holding_pieces:
                player.finish_move()
                # Deduct an action if it's a standard move (not a BM Move3 or MoveAny2)
                if not (game.waiting_for_bm_move3 or game.waiting_for_bm_move_any_2):
                    player.actions_remaining -= 1
        else:
            print("Cannot place a piece here. The post is already occupied.")

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
        print(f"Attempting to place a {desired_shape} while Displaced Shape has NOT been played yet")
        displace_to(game, post, desired_shape, use_displaced_piece=True)
    else:
        displace_to(game, post, desired_shape)

def displace_to(game, post, shape, use_displaced_piece=False):
    displaced_player = game.displaced_player
    if use_displaced_piece:
        print(f"Attempting to use displaced piece {shape} - no affect to the GS or PS")
        claim_and_update(game, post, shape, use_displaced_piece=True)
    else:
        if displaced_player.has_general_stock(shape):
            print(f"Attempting to place a {shape} from general_stock, because use_displaced is false")
            claim_and_update(game, post, shape)
        elif displaced_player.is_general_stock_empty() and displaced_player.has_personal_supply(shape):
            print(f"Attempting to place a {shape} from personal_supply, because use_displaced is false")
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
    
    displaced_player.total_pieces_to_place -= 1