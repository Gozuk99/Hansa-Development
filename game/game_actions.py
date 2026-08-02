from collections import deque

from map_data.constants import ACTIONS_MAX_VALUES, DARK_GREEN


class InvalidActionError(RuntimeError):
    """Raised when an interaction violates the current game rules."""


def claim_post_action(game, route, post, piece_to_play):
    player = game.current_player

    if not player.has_personal_supply(piece_to_play):
        raise InvalidActionError(f"No {piece_to_play} is available in personal supply")
    if not post.can_be_claimed_by(piece_to_play):
        raise InvalidActionError("The selected post cannot accept that piece")
    if not game.check_brown_blue_priv(route):
        raise InvalidActionError("The player lacks the required regional privilege")

    block_cost = len(route.block_marker_owners)
    available_squares = player.personal_supply_squares - int(piece_to_play == "square")
    available_circles = player.personal_supply_circles - int(piece_to_play == "circle")
    if available_squares + available_circles < block_cost:
        raise InvalidActionError("Not enough pieces are available to pay the route block cost")

    if route.has_bonus_marker:
        player.reward += player.reward_structure.post_with_bm
    elif route.has_permanent_bm_type:
        player.reward += player.reward_structure.post_with_perm_bm
    if route.cities[0].upgrade_city_type or route.cities[1].upgrade_city_type:
        player.reward += player.reward_structure.post_adjacent_to_upgrade_city
    else:
        player.reward += player.reward_structure.post_with_nothing

    game.consume_region_privilege(route)
    if block_cost:
        squares_to_pay = min(available_squares, block_cost)
        circles_to_pay = block_cost - squares_to_pay
        player.personal_supply_squares -= squares_to_pay
        player.personal_supply_circles -= circles_to_pay
        player.general_stock_squares += squares_to_pay
        player.general_stock_circles += circles_to_pay

    post.claim(player, piece_to_play)
    if piece_to_play == "square":
        player.personal_supply_squares -= 1
    else:
        player.personal_supply_circles -= 1
    player.spend_action()


def displace_action(game, post, route, displacing_piece_shape):
    current_player = game.current_player
    displaced_piece_shape = post.owner_piece_shape
    current_displaced_player = post.owner
    if current_displaced_player is None or displaced_piece_shape is None:
        raise InvalidActionError("The selected post has no piece to displace")
    if current_displaced_player is current_player:
        raise InvalidActionError("A player cannot displace their own piece")

    cost = 2 if displaced_piece_shape == "square" else 3
    if current_player.personal_supply_squares + current_player.personal_supply_circles < cost:
        raise InvalidActionError("Not enough pieces are available to displace")
    if not current_player.has_personal_supply(displacing_piece_shape):
        raise InvalidActionError(
            f"No {displacing_piece_shape} is available to perform the displacement"
        )
    if not game.check_brown_blue_priv(route):
        raise InvalidActionError("The player lacks the required regional privilege")

    game.active_player = current_displaced_player.order - 1
    game.consume_region_privilege(route)

    if route.has_bonus_marker:
        current_player.reward += current_player.reward_structure.post_with_bm - 2
    elif route.has_permanent_bm_type:
        current_player.reward += current_player.reward_structure.post_with_perm_bm - 2
    if route.cities[0].upgrade_city_type or route.cities[1].upgrade_city_type:
        current_player.reward += current_player.reward_structure.post_adjacent_to_upgrade_city - 2
    else:
        current_player.reward += current_player.reward_structure.post_with_nothing

    if displacing_piece_shape == "square":
        current_player.personal_supply_squares -= 1
    else:
        current_player.personal_supply_circles -= 1
    post.claim(current_player, displacing_piece_shape)

    squares_to_pay = min(current_player.personal_supply_squares, cost - 1)
    circles_to_pay = cost - 1 - squares_to_pay

    current_player.personal_supply_squares -= squares_to_pay
    current_player.personal_supply_circles -= circles_to_pay

    current_player.general_stock_squares += squares_to_pay
    current_player.general_stock_circles += circles_to_pay

    game.original_route_of_displacement = route
    game.waiting_for_displaced_player = True
    game.displaced_player.populate_displaced_player(
        game, current_displaced_player, displaced_piece_shape
    )

    refresh_displacement_targets(game)


def _post_accepts_any_shape(post, shapes):
    return not post.is_owned() and (post.required_shape is None or post.required_shape in shapes)


def gather_all_empty_posts(game, required_shapes=("square", "circle")):
    all_empty_posts = []
    for route in game.selected_map.routes:
        for post in route.posts:
            if _post_accepts_any_shape(post, required_shapes):
                all_empty_posts.append(post)
    return all_empty_posts


def gather_empty_adjacent_posts(
    start_route,
    required_shapes=("square", "circle"),
):
    """Return compatible empty posts at the nearest reachable route distance."""
    if not start_route:
        raise InvalidActionError("Displacement has no originating route")

    visited_routes = {start_route}
    queue = deque(get_adjacent_routes(start_route, start_route.region))

    while queue:
        level_size = len(queue)
        empty_posts = []
        next_level_routes = []

        for _ in range(level_size):
            current_route = queue.popleft()
            if current_route in visited_routes:
                continue
            visited_routes.add(current_route)

            adjacent_routes = get_adjacent_routes(current_route, start_route.region)
            for route in adjacent_routes:
                if route not in visited_routes and route not in next_level_routes:
                    next_level_routes.append(route)

            for post in current_route.posts:
                if _post_accepts_any_shape(post, required_shapes):
                    empty_posts.append(post)

        if empty_posts:
            return empty_posts

        queue.extend(next_level_routes)

    return []


def displacement_shapes_to_place(game):
    """Return shapes available among all pieces remaining in the sequence.

    The displaced piece is mandatory, but it need not be placed first. Optional
    pieces therefore participate in the same nearest-distance search until the
    displaced piece has been placed. General Stock retains priority over
    Personal Supply.
    """
    displaced = game.displaced_player
    player = game.displaced_player.player
    if player.holding_pieces:
        return (player.holding_pieces[0][0],)
    if displaced.use_optional_displaced_shape:
        return (displaced.displaced_shape,)

    shapes = []
    if not displaced.played_displaced_shape:
        shapes.append(displaced.displaced_shape)
        if displaced.total_pieces_to_place == 1:
            return tuple(shapes)

    if displacement_uses_board_fallback(game):
        return tuple(shapes)

    if not displaced.is_general_stock_empty():
        source_counts = (
            ("square", player.general_stock_squares),
            ("circle", player.general_stock_circles),
        )
    else:
        source_counts = (
            ("square", player.personal_supply_squares),
            ("circle", player.personal_supply_circles),
        )

    for shape, count in source_counts:
        # Pieces have no identity beyond shape. When an optional source contains
        # the displaced shape, one post action represents both choices; applying
        # it to the mandatory piece first preserves every possible board result
        # and leaves the identical optional piece available (and declinable).
        if count and shape not in shapes:
            shapes.append(shape)
    return tuple(shapes)


def displacement_uses_board_fallback(game):
    displaced = game.displaced_player
    return (
        displaced.played_displaced_shape
        and displaced.is_general_stock_empty()
        and displaced.is_personal_supply_empty()
    )


def displacement_piece_available(game, shape):
    """Whether a shape exists in the current rule-ordered source."""
    return shape in displacement_shapes_to_place(game)


def optional_displacement_piece_available(game, shape):
    """Whether the current optional source contains a piece of this shape."""
    displaced = game.displaced_player
    if displaced.is_general_stock_empty():
        return displaced.player.has_personal_supply(shape)
    return displaced.has_general_stock(shape)


def select_optional_displaced_shape(game):
    """Use an optional source piece before the identical mandatory piece."""
    displaced = game.displaced_player
    if (
        displaced.played_displaced_shape
        or displaced.use_optional_displaced_shape
        or displaced.total_pieces_to_place <= 1
        or not optional_displacement_piece_available(game, displaced.displaced_shape)
    ):
        raise RuntimeError("No identical optional displacement piece is available")
    displaced.use_optional_displaced_shape = True
    refresh_displacement_targets(game)


def can_pick_up_displacement_fallback(game, post):
    displaced = game.displaced_player
    return (
        displacement_uses_board_fallback(game)
        and displaced.player.pieces_to_pickup > 0
        and post.owner == displaced.player
    )


def can_place_displacement_piece(game, post, shape):
    """Apply nearest-target, shape, and source rules at one boundary."""
    return (
        post in game.all_empty_posts
        and not post.is_owned()
        and shape in displacement_shapes_to_place(game)
        and post.required_shape in (None, shape)
        and displacement_piece_available(game, shape)
    )


def refresh_displacement_targets(game):
    """Mark only the nearest posts legal for the piece currently being placed."""
    for candidate in game.all_empty_posts:
        candidate.reset_post()

    available_shapes = displacement_shapes_to_place(game)
    if game.DisplaceAnywhereOwner == game.displaced_player.player:
        targets = gather_all_empty_posts(game, available_shapes)
    else:
        targets = gather_empty_adjacent_posts(
            game.original_route_of_displacement,
            available_shapes,
        )

    game.all_empty_posts = targets
    for candidate in targets:
        candidate.valid_post_to_displace_to()
    return targets


def get_adjacent_routes(current_route, start_route_region):
    if not current_route:
        raise InvalidActionError("Cannot find adjacent routes without a starting route")

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

    if post is None:
        raise InvalidActionError("No post was selected")

    if (game.waiting_for_bm_move3 and post.is_owned() and post.owner != player) or (
        game.waiting_for_bm_move_any_2 and post.is_owned()
    ):
        if player.pieces_to_pickup > 0:
            player.pick_up_piece(post)
        else:
            raise InvalidActionError("No additional pieces may be picked up")
    elif post.owner == player:
        if player.pieces_to_pickup == 0:
            player.start_move()
        player.pick_up_piece(post)

        if route.has_bonus_marker:
            player.reward -= player.reward_structure.post_with_bm
        elif route.has_permanent_bm_type:
            player.reward -= player.reward_structure.post_with_perm_bm
        if route.cities[0].upgrade_city_type or route.cities[1].upgrade_city_type:
            player.reward -= player.reward_structure.post_adjacent_to_upgrade_city
    elif player.holding_pieces:
        if not post.is_owned():
            shape_to_place, owner_to_place, origin_region = player.holding_pieces[0]
            player.place_piece(post, shape_to_place)

            if owner_to_place == player:
                if route.has_bonus_marker:
                    player.reward += player.reward_structure.post_with_bm
                elif route.has_permanent_bm_type:
                    player.reward += player.reward_structure.post_with_perm_bm
                if route.cities[0].upgrade_city_type or route.cities[1].upgrade_city_type:
                    player.reward += player.reward_structure.post_adjacent_to_upgrade_city
            if not player.holding_pieces:
                if player.pieces_to_pickup > 0:
                    player.reward -= 10
                player.finish_move()
                if not (
                    game.waiting_for_bm_move3
                    or game.waiting_for_bm_move_any_2
                    or game.waiting_for_place2_from_route
                    or game.waiting_for_place2_in_scotland_or_wales
                ):
                    player.spend_action()
                else:
                    if game.waiting_for_bm_move3:
                        game.waiting_for_bm_move3 = False
                    elif game.waiting_for_bm_move_any_2:
                        game.waiting_for_bm_move_any_2 = False
                    elif game.waiting_for_place2_from_route:
                        game.waiting_for_place2_from_route = False
                    elif game.waiting_for_place2_in_scotland_or_wales:
                        game.waiting_for_place2_in_scotland_or_wales = False
        else:
            raise InvalidActionError("The selected post is occupied")
    else:
        raise InvalidActionError("The selected post cannot be used during this move")


def displace_move_action(game, post):
    displaced_player = game.displaced_player.player
    if post is None:
        raise InvalidActionError("No displacement post was selected")

    if post.owner == displaced_player and displaced_player.pieces_to_pickup > 0:
        displaced_player.pick_up_piece(post)

        refresh_displacement_targets(game)

    # If the player has pieces in hand to place
    elif displaced_player.holding_pieces:
        if not post.is_owned() and post in game.all_empty_posts:
            shape_to_place, owner_to_place, origin_region = displaced_player.holding_pieces[0]
            displaced_player.place_piece(post, shape_to_place)
            game.displaced_player.total_pieces_to_place -= 1
            game.all_empty_posts.remove(post)
        else:
            raise InvalidActionError("The displacement destination is not currently legal")


def displace_claim(game, post, desired_shape):
    displaced_player = game.displaced_player

    # Check if the player is forced to use the displaced shape
    must_use_displaced_piece = False
    if not displaced_player.played_displaced_shape:
        if (
            displaced_player.displaced_shape == "circle"
            and displaced_player.total_pieces_to_place == 1
        ):
            must_use_displaced_piece = True
        elif (
            displaced_player.displaced_shape == "square"
            and displaced_player.total_pieces_to_place == 1
        ):
            must_use_displaced_piece = True

    if must_use_displaced_piece and desired_shape != displaced_player.displaced_shape:
        raise InvalidActionError("The mandatory displaced piece must be placed")

    is_board_fallback_pickup = can_pick_up_displacement_fallback(game, post)
    if not is_board_fallback_pickup and not can_place_displacement_piece(game, post, desired_shape):
        raise InvalidActionError(
            "The selected displacement piece must use a nearest compatible post"
        )

    wants_to_use_displaced_piece = (
        not displaced_player.played_displaced_shape
        and desired_shape == displaced_player.displaced_shape
        and not displaced_player.use_optional_displaced_shape
    )
    refresh_displacement_targets(game)

    if wants_to_use_displaced_piece:
        displace_to(game, post, desired_shape, use_displaced_piece=True)
        if (
            displaced_player.is_general_stock_empty()
            and displaced_player.is_personal_supply_empty()
        ):
            displaced_player.player.pieces_to_pickup = displaced_player.total_pieces_to_place - len(
                displaced_player.player.holding_pieces
            )
    elif (
        displaced_player.played_displaced_shape == True
        and displaced_player.is_general_stock_empty()
        and displaced_player.is_personal_supply_empty()
    ):
        displaced_player.player.pieces_to_pickup = displaced_player.total_pieces_to_place - len(
            displaced_player.player.holding_pieces
        )
        displace_move_action(game, post)
    else:
        displace_to(game, post, desired_shape)

    displaced_player.use_optional_displaced_shape = False

    if game.displaced_player.all_pieces_placed():
        for post in game.all_empty_posts:
            post.reset_post()
        game.all_empty_posts.clear()
        game.original_route_of_displacement = None
        game.displaced_player.reset_displaced_player()
        game.waiting_for_displaced_player = False
        game.current_player.spend_action()
        game.active_player = game.current_player.order - 1
    else:
        refresh_displacement_targets(game)


def finish_displacement(game):
    """Decline optional extra pieces after replacing the displaced piece."""
    displaced = game.displaced_player
    if not game.waiting_for_displaced_player:
        raise RuntimeError("No displacement is in progress")
    if not displaced.played_displaced_shape:
        raise RuntimeError("The displaced piece itself must be placed")
    if displaced.player.holding_pieces:
        raise RuntimeError("Held pieces must be placed before finishing displacement")

    for candidate in game.all_empty_posts:
        candidate.reset_post()
    game.all_empty_posts.clear()
    game.original_route_of_displacement = None
    displaced.reset_displaced_player()
    game.waiting_for_displaced_player = False
    game.current_player.spend_action()
    game.active_player = game.current_player.order - 1


def displace_to(game, post, shape, use_displaced_piece=False):
    displaced_player = game.displaced_player
    if use_displaced_piece:
        claim_and_update(game, post, shape, use_displaced_piece=True)
    else:
        if displaced_player.has_general_stock(shape):
            claim_and_update(game, post, shape)
        elif displaced_player.is_general_stock_empty() and displaced_player.has_personal_supply(
            shape
        ):
            claim_and_update(game, post, shape, from_personal_supply=True)
        else:
            raise InvalidActionError(f"No {shape} is available from the required source")


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

    if (
        displaced_player.player.personal_supply_squares < 0
        or displaced_player.player.personal_supply_circles < 0
    ):
        raise InvalidActionError("Displacement produced a negative personal supply")

    displaced_player.total_pieces_to_place -= 1


def assign_new_bonus_marker_on_route(game, route):
    if not route:
        return False

    if route.bonus_marker or route.permanent_bonus_marker:
        return False

    if game.map_num == 3 and route.region in ("Wales", "Scotland"):
        return False

    if route.has_tradesmen():
        return False

    if not route.has_empty_office_in_cities():
        return False

    if game.pending_bonus_markers:
        bm_type = game.pending_bonus_markers.pop(0)
        route.assign_map_new_bonus_marker(bm_type)
        game.replace_bonus_marker -= 1
        game.switch_player_if_needed()
        return True
    return False


def score_route(current_player, route):
    # Allocate points
    for city in route.cities:
        player = city.get_controller()
        if player is not None:
            player.score += 1
            if current_player.color == player.color:
                current_player.reward += player.reward_structure.route_complete_got_points
            else:
                current_player.reward -= player.reward_structure.route_complete_got_points


def claim_route_for_office(game, city, route):
    current_player = game.current_player
    next_open_office_color = city.get_next_open_office_color()
    if current_player.player_can_claim_office(next_open_office_color) and city.color != DARK_GREEN:
        if city.has_required_piece_shape(current_player, route):
            current_player.reward += current_player.reward_structure.city_claim_office
            score_route(current_player, route)
            placed_piece_shape = city.get_next_open_office_shape()
            city.update_next_open_office_ownership(game, placed_piece_shape)
            finalize_route_claim(game, route, placed_piece_shape)
            route.award_tributes(game)
    elif "PlaceAdjacent" in (bm.type for bm in current_player.bonus_markers):
        current_player.reward += current_player.reward_structure.bm_place_adjacent
        score_route(current_player, route)
        city.claim_office_with_bonus_marker(current_player)
        finalize_route_claim(game, route, "square")
        route.award_tributes(game)


def claim_route_for_additional_office(game, city, route, shape):
    player = game.current_player
    if not city.can_claim_additional_office(player, route, shape):
        raise ValueError("Additional Trading Post choice is no longer legal")
    score_route(player, route)
    city.claim_office_with_bonus_marker(player, shape)
    finalize_route_claim(game, route, shape)
    route.award_tributes(game)
    game.waiting_for_bm_place_adjacent = False


def claim_route_for_upgrade(game, city, route, upgrade_choice, prestige_value=None):
    current_player = game.current_player
    specialprestigepoints_city = game.selected_map.specialprestigepoints

    if "SpecialPrestigePoints" in city.upgrade_city_type and route.contains_a_circle():
        claimed = (
            specialprestigepoints_city.claim_prestige(current_player, prestige_value)
            if prestige_value is not None
            else specialprestigepoints_city.claim_highest_prestige(current_player)
        )
        if claimed:
            current_player.reward += current_player.reward_structure.upgraded_bonus_points
            score_route(current_player, route)
            finalize_route_claim(game, route, "circle")
    elif any(
        upgrade_type in ["Keys", "Privilege", "Book", "Actions", "Bank"]
        for upgrade_type in city.upgrade_city_type
    ):
        upgrade_rewards = {
            "Keys": "upgraded_keys",
            "Privilege": "upgraded_privilege",
            "Book": "upgraded_circles",
            "Actions": "upgraded_actions",
            "Bank": "upgraded_bank",
        }
        if upgrade_choice and current_player.perform_upgrade(upgrade_choice):
            current_player.reward += getattr(
                current_player.reward_structure, upgrade_rewards[upgrade_choice]
            )
            score_route(current_player, route)
            finalize_route_claim(game, route)


def claim_route_for_points(game, route):
    current_player = game.current_player
    score_route(current_player, route)
    finalize_route_claim(game, route)


def finalize_route_claim(game, route, placed_piece_shape=None):
    reset_pieces = update_stock_and_reset(route, game.current_player, placed_piece_shape)
    handle_bonus_marker(game, game.current_player, route, reset_pieces)
    game.current_player.spend_action()
    game.check_for_east_west_connection()


def handle_bonus_marker(game, player, route, reset_pieces):
    if route.bonus_marker:
        player.reward += player.reward_structure.route_complete_receive_bm
        route.bonus_marker.owner = player
        player.bonus_markers.append(route.bonus_marker)
        route.bonus_marker = None
        route.has_bonus_marker = False
        if game.selected_map.bonus_marker_pool:
            game.pending_bonus_markers.append(game.selected_map.bonus_marker_pool.pop())
            game.replace_bonus_marker += 1
        else:
            game.bonus_pool_exhausted_during_claim = True
    elif route.permanent_bonus_marker:
        player.reward += player.reward_structure.route_complete_perm_bm
        perm_bm_type = route.permanent_bonus_marker.type
        if perm_bm_type == "MoveAny2":
            game.current_player.pieces_to_pickup = 2
            game.waiting_for_bm_move_any_2 = True
        elif perm_bm_type == "+1Priv":
            game.current_player.perform_upgrade("Privilege")
        elif perm_bm_type == "ClaimGreenCity":
            if (
                game.current_player.personal_supply_squares > 0
                or game.current_player.personal_supply_circles > 0
            ):
                game.waiting_for_bm_green_city = True
            else:
                player.reward -= 20
        elif perm_bm_type == "Place2TradesmenFromRoute":
            game.pending_route_piece_choices = reset_pieces
        elif perm_bm_type == "Place2ScotlandOrWales":
            game.pending_britannia_place2 = True
            game.current_player.pieces_to_place = 2


def update_stock_and_reset(route, player, placed_piece_shape=None):
    """Update player's general stock based on pieces on the route and reset those posts."""
    circles_on_route = sum(
        1 for post in route.posts if post.owner == player and post.owner_piece_shape == "circle"
    )
    squares_on_route = sum(
        1 for post in route.posts if post.owner == player and post.owner_piece_shape == "square"
    )

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


def buy_tile(game, tile_type, bm_payment1=None, bm_payment2=None):
    player = game.current_player

    if tile_type not in game.tile_pool:
        raise ValueError(f"Emperor's Favour tile is unavailable: {tile_type}")
    if player.actions_remaining != player.actions_at_turn_start:
        raise ValueError("Emperor's Favour may only be bought at the start of a turn")
    payments = [bm_payment1, bm_payment2]
    if any(marker is None for marker in payments):
        raise ValueError("Exactly two unused bonus markers are required")
    if payments[0] is payments[1]:
        raise ValueError("Two distinct unused bonus markers are required")
    if any(marker not in player.bonus_markers for marker in payments):
        raise ValueError("Payment must use the buyer's unused bonus markers")

    for marker in payments:
        player.bonus_markers.remove(marker)
        player.used_bonus_markers.append(marker)
    game.tile_pool.remove(tile_type)
    player.tiles.append(tile_type)

    if tile_type == "DisplaceAnywhere":
        game.DisplaceAnywhereOwner = player
    elif tile_type == "+1Action":
        game.OneActionOwner = player
    elif tile_type == "+1IncomeIfOthersIncome":
        game.OneIncomeIfOthersIncomeOwner = player
    elif tile_type == "+1DisplacedPiece":
        game.OneDisplacedPieceOwner = player
    elif tile_type == "+4PtsPerOwnedCity":
        game.FourPtsPerOwnedCityOwner = player
    elif tile_type == "+7PtsPerCompletedAbility":
        game.SevenPtsPerCompletedAbilityOwner = player

    player.forfeit_remaining_actions()
