from game.action_schema import (
    BONUS_MARKER_PAYMENT_TYPES,
    BONUS_MARKER_SLOT_BY_TYPE,
    TILE_SLOT_BY_TYPE,
)
from game.game_actions import (
    can_pick_up_displacement_fallback,
    can_place_displacement_piece,
    displacement_can_be_completed,
    optional_displacement_piece_available,
)
from game.turn_state import TurnPhase
from map_data.constants import DARK_GREEN, MAX_CITIES, MAX_POSTS, MAX_ROUTES, UPGRADE_MAX_VALUES


def _empty_mask(size):
    return [False] * size


def mask_place_adjacent(game):
    tensor = _empty_mask(1)
    if game.turn_phase == TurnPhase.DISPLACEMENT:
        displaced = game.displaced_player
        if (
            not displaced.played_displaced_shape
            and not displaced.use_optional_displaced_shape
            and displaced.total_pieces_to_place > 1
            and optional_displacement_piece_available(game, displaced.displaced_shape)
        ):
            tensor[0] = 1
        return tensor
    if game.turn_phase != TurnPhase.ACTIONS:
        return tensor
    if any(marker.type == "PlaceAdjacent" for marker in game.current_player.bonus_markers) and any(
        city.can_claim_additional_office(game.current_player, route, shape)
        for route in game.selected_map.routes
        if route.is_controlled_by(game.current_player)
        for city in route.cities
        for shape in ("square", "circle")
    ):
        tensor[0] = 1
    return tensor


def _has_pending_post_workflow(game):
    return (
        game.waiting_for_displaced_player
        or game.waiting_for_bm_move_any_2
        or game.waiting_for_bm_move3
    )


def _has_pending_action_choice(game):
    return (
        game.waiting_for_bm_swap_office
        or game.waiting_for_bm_upgrade_ability
        or game.waiting_for_bm_exchange_bm
        or game.waiting_for_bm_green_city
    )


def mask_post_action(game):
    current_player = game.current_player

    post_tensor = _empty_mask(MAX_POSTS * 2)

    special_post_workflow = (
        game.waiting_for_bm_move_any_2
        or game.waiting_for_place2_from_route
        or game.waiting_for_place2_in_scotland_or_wales
        or game.waiting_for_bm_move3
        or game.waiting_for_bm_tribute_trading_post
        or game.waiting_for_bm_block_trade_route
    )
    if not special_post_workflow and (
        current_player.actions_remaining == 0 or _has_pending_action_choice(game)
    ):
        return post_tensor

    can_displace = (
        current_player.personal_supply_squares + current_player.personal_supply_circles > 1
    )

    post_idx = 0
    for route in game.selected_map.routes:
        for post in route.posts:
            # Common checks for valid region transition and post not being owned
            is_post_owned = post.is_owned()
            is_post_empty = not is_post_owned

            if game.waiting_for_displaced_player:
                displaced_player = game.displaced_player
                if can_pick_up_displacement_fallback(game, post):
                    # Any owned route piece may be selected; after pickup its
                    # actual shape becomes the placement requirement.
                    post_tensor[post_idx] = 1
                    post_tensor[MAX_POSTS + post_idx] = 1
                else:
                    for available_shape in ("square", "circle"):
                        if can_place_displacement_piece(game, post, available_shape):
                            shape_offset = MAX_POSTS if available_shape == "circle" else 0
                            post_tensor[shape_offset + post_idx] = 1
            elif game.waiting_for_bm_move_any_2:
                if post.is_owned() and current_player.pieces_to_pickup > 0:
                    post_tensor[post_idx] = 1
                    post_tensor[MAX_POSTS + post_idx] = 1
                elif (
                    not post.is_owned()
                    and current_player.pieces_to_pickup == 0
                    and current_player.holding_pieces
                ):
                    shape_to_place, _, origin_region = current_player.holding_pieces[0]
                    if (
                        origin_region == post.region
                        if game.map_num == 3
                        else current_player.is_valid_region_transition(origin_region, post.region)
                    ):
                        if shape_to_place == "square" and (
                            not post.required_shape or post.required_shape == "square"
                        ):
                            post_tensor[post_idx] = 1
                        elif shape_to_place == "circle" and (
                            not post.required_shape or post.required_shape == "circle"
                        ):
                            post_tensor[MAX_POSTS + post_idx] = 1

            elif game.waiting_for_place2_in_scotland_or_wales:
                if (
                    post.region in ("Scotland", "Wales")
                    and not post.is_owned()
                    and current_player.holding_pieces
                ):
                    shape = current_player.holding_pieces[0][0]
                    if post.required_shape in (None, shape):
                        post_tensor[post_idx + (MAX_POSTS if shape == "circle" else 0)] = 1

            elif game.waiting_for_bm_move3:
                if (
                    post.is_owned()
                    and post.owner != current_player
                    and current_player.pieces_to_pickup > 0
                ):
                    post_tensor[post_idx] = 1
                    post_tensor[MAX_POSTS + post_idx] = 1
                elif (
                    not post.is_owned()
                    and current_player.pieces_to_pickup == 0
                    and current_player.holding_pieces
                ):
                    shape_to_place, _, origin_region = current_player.holding_pieces[0]
                    if game.map_num != 3 or origin_region == post.region:
                        if shape_to_place == "square" and (
                            not post.required_shape or post.required_shape == "square"
                        ):
                            post_tensor[post_idx] = 1
                        elif shape_to_place == "circle" and (
                            not post.required_shape or post.required_shape == "circle"
                        ):
                            post_tensor[MAX_POSTS + post_idx] = 1
            elif game.waiting_for_place2_from_route:
                if not post.is_owned() and current_player.holding_pieces:
                    shape = current_player.holding_pieces[0][0]
                    if post.required_shape in (None, shape):
                        post_tensor[post_idx + (MAX_POSTS if shape == "circle" else 0)] = 1
            elif game.waiting_for_bm_tribute_trading_post:
                if post is route.posts[0] and current_player.personal_supply_squares > 0:
                    post_tensor[post_idx] = 1
            elif game.waiting_for_bm_block_trade_route:
                if post is route.posts[0] and current_player.personal_supply_squares > 0:
                    post_tensor[post_idx] = 1

            else:
                # MOVE: first pick up owned pieces, then place held pieces on
                # compatible empty posts. The selected post already determines
                # the piece's shape during pickup.
                if current_player.holding_pieces:
                    if is_post_empty:
                        shape_to_place, _, origin_region = current_player.holding_pieces[0]
                        if current_player.is_valid_region_transition(origin_region, post.region):
                            if shape_to_place == "square" and (
                                not post.required_shape or post.required_shape == "square"
                            ):
                                post_tensor[post_idx] = 1
                            elif shape_to_place == "circle" and (
                                not post.required_shape or post.required_shape == "circle"
                            ):
                                post_tensor[MAX_POSTS + post_idx] = 1
                    elif (
                        post.owner == current_player
                        and current_player.pieces_to_pickup > 0
                        and len(current_player.holding_pieces) < current_player.book
                    ):
                        post_tensor[post_idx] = 1
                        post_tensor[MAX_POSTS + post_idx] = 1

                elif is_post_owned and post.owner == current_player:
                    post_tensor[post_idx] = 1
                    post_tensor[MAX_POSTS + post_idx] = 1
                elif is_post_empty and game.check_brown_blue_priv(route):
                    block_cost = len(route.block_marker_owners)
                    total_supply = (
                        current_player.personal_supply_squares
                        + current_player.personal_supply_circles
                    )
                    if (
                        current_player.personal_supply_squares > 0
                        and total_supply > block_cost
                        and (not post.required_shape or post.required_shape == "square")
                    ):
                        post_tensor[post_idx] = 1
                    if (
                        current_player.personal_supply_circles > 0
                        and total_supply > block_cost
                        and (not post.required_shape or post.required_shape == "circle")
                    ):
                        post_tensor[MAX_POSTS + post_idx] = 1
                elif (
                    is_post_owned
                    and post.owner != current_player
                    and game.check_brown_blue_priv(route)
                    and can_displace
                ):
                    displacement_cost = 2 if post.owner_piece_shape == "square" else 3
                    if (
                        current_player.personal_supply_squares
                        + current_player.personal_supply_circles
                        >= displacement_cost
                        and displacement_can_be_completed(
                            game, route, post.owner, post.owner_piece_shape
                        )
                    ):
                        if (
                            post.required_shape in (None, "square")
                            and current_player.personal_supply_squares > 0
                        ):
                            post_tensor[post_idx] = 1
                        if (
                            post.required_shape in (None, "circle")
                            and current_player.personal_supply_circles > 0
                        ):
                            post_tensor[MAX_POSTS + post_idx] = 1
            post_idx += 1

    return post_tensor


def mask_claim_route(game):
    max_num_routes = MAX_ROUTES
    two_cities_per_route = 2
    max_upgrades_per_city = 2

    claim_route_for_points_tensor = _empty_mask(max_num_routes)
    claim_route_for_office_tensor = _empty_mask(max_num_routes * two_cities_per_route)
    claim_route_for_upgrade_tensor = _empty_mask(
        max_num_routes * two_cities_per_route * max_upgrades_per_city
    )

    if game.waiting_for_bm_place_adjacent:
        for route_idx, route in enumerate(game.selected_map.routes):
            if not route.is_controlled_by(game.current_player):
                continue
            for city_idx, city in enumerate(route.cities):
                for shape_idx, shape in enumerate(("square", "circle")):
                    if city.can_claim_additional_office(game.current_player, route, shape):
                        claim_route_for_upgrade_tensor[route_idx * 4 + city_idx * 2 + shape_idx] = 1
        return (
            claim_route_for_points_tensor
            + claim_route_for_office_tensor
            + claim_route_for_upgrade_tensor
        )

    if (
        game.current_player.actions_remaining == 0
        or game.current_player.holding_pieces
        or _has_pending_post_workflow(game)
        or _has_pending_action_choice(game)
    ):
        claim_route_tensor = (
            claim_route_for_points_tensor
            + claim_route_for_office_tensor
            + claim_route_for_upgrade_tensor
        )
        return claim_route_tensor

    route_idx = 0
    for route in game.selected_map.routes:
        if route.is_controlled_by(game.current_player):
            claim_route_for_points_tensor[route_idx] = 1

            special_city = next(
                (
                    city
                    for city in route.cities
                    if "SpecialPrestigePoints" in city.upgrade_city_type
                ),
                None,
            )
            if special_city is not None and route.contains_a_circle():
                prestige_values = [7, 8, 9, 11]
                for prestige_index, prestige_value in enumerate(prestige_values):
                    if game.selected_map.specialprestigepoints.can_claim_prestige(
                        game.current_player, prestige_value
                    ):
                        claim_route_for_upgrade_tensor[route_idx * 4 + prestige_index] = 1

            for city_idx, city in enumerate(route.cities):
                base_index_office = route_idx * two_cities_per_route + city_idx

                if city.has_empty_office():
                    next_open_office_color = city.get_next_open_office_color()
                    if (
                        game.current_player.player_can_claim_office(next_open_office_color)
                        and city.color != DARK_GREEN
                    ):
                        if city.has_required_piece_shape(game.current_player, route):
                            claim_route_for_office_tensor[base_index_office] = 1
                if city.upgrade_city_type and special_city is None:
                    for upgrade_idx, upgrade in enumerate(city.upgrade_city_type):
                        if upgrade_idx < max_upgrades_per_city:
                            action_index_upgrade = (
                                (route_idx * two_cities_per_route * max_upgrades_per_city)
                                + (city_idx * max_upgrades_per_city)
                                + upgrade_idx
                            )
                            if upgrade == "SpecialPrestigePoints":
                                if (
                                    route.contains_a_circle()
                                    and game.selected_map.specialprestigepoints.can_claim_prestige(
                                        game.current_player
                                    )
                                ):
                                    claim_route_for_upgrade_tensor[action_index_upgrade] = 1
                            else:
                                current_value = getattr(game.current_player, upgrade.lower())
                                if current_value != UPGRADE_MAX_VALUES[upgrade.lower()]:
                                    claim_route_for_upgrade_tensor[action_index_upgrade] = 1

        route_idx += 1

    claim_route_tensor = (
        claim_route_for_points_tensor
        + claim_route_for_office_tensor
        + claim_route_for_upgrade_tensor
    )
    return claim_route_tensor


def mask_income_actions(game):
    income_tensor = _empty_mask(5)

    if game.pending_britannia_place2:
        player = game.current_player
        board_circles = sum(
            post.owner == player and post.owner_piece_shape == "circle"
            for route in game.selected_map.routes
            for post in route.posts
        )
        board_squares = sum(
            post.owner == player and post.owner_piece_shape == "square"
            for route in game.selected_map.routes
            for post in route.posts
        )
        pools = (
            (player.general_stock_squares, player.general_stock_circles),
            (player.personal_supply_squares, player.personal_supply_circles),
            (board_squares, board_circles),
        )
        remaining_slots = 2
        quotas = []
        for squares, circles in pools:
            quota = min(remaining_slots, squares + circles)
            quotas.append(quota)
            remaining_slots -= quota
        if remaining_slots:
            return income_tensor
        for requested_circles in range(3):
            for first in range(quotas[0] + 1):
                for second in range(quotas[1] + 1):
                    third = requested_circles - first - second
                    candidates = (first, second, third)
                    if 0 <= third <= quotas[2] and all(
                        circles >= selected_circles and squares >= quota - selected_circles
                        for (squares, circles), quota, selected_circles in zip(
                            pools, quotas, candidates
                        )
                    ):
                        income_tensor[requested_circles] = 1
                        break
                if income_tensor[requested_circles]:
                    break
        return income_tensor

    if game.pending_route_piece_choices:
        circles = sum(
            shape == "circle" for shape, _owner, _region in game.pending_route_piece_choices
        )
        squares = len(game.pending_route_piece_choices) - circles
        for circle_count in range(3):
            if circle_count <= circles and 2 - circle_count <= squares:
                income_tensor[circle_count] = 1
        return income_tensor

    if game.pending_tribute_income_owners:
        owner = game.pending_tribute_income_owners[0]
        amount = min(2, owner.general_stock_squares + owner.general_stock_circles)
        for circles in range(amount + 1):
            squares = amount - circles
            if circles <= owner.general_stock_circles and squares <= owner.general_stock_squares:
                income_tensor[circles] = 1
        return income_tensor

    if (
        game.current_player.actions_remaining == 0
        or game.current_player.holding_pieces
        or _has_pending_post_workflow(game)
        or _has_pending_action_choice(game)
    ):
        return income_tensor

    num_circles = game.current_player.general_stock_circles
    num_squares = game.current_player.general_stock_squares
    max_income = min(game.current_player.bank, 4)

    if num_circles == 0 and num_squares == 0:
        return income_tensor

    if num_circles == 0:
        income_tensor[0] = 1
        return income_tensor

    # Set only actions that transfer at least one piece. A changed deterministic
    # game path can otherwise expose a zero-piece income choice when only the
    # unavailable shape is requested.
    for i in range(max_income + 1):
        circles_to_take = min(num_circles, i)
        squares_to_take = min(num_squares, game.current_player.bank - circles_to_take)
        if i <= num_circles and circles_to_take + squares_to_take > 0:
            income_tensor[i] = 1

    return income_tensor


def mask_bm(game):
    bm_tensor = _empty_mask(len(BONUS_MARKER_SLOT_BY_TYPE))

    if (
        game.waiting_for_displaced_player
        or (_has_pending_post_workflow(game) and not game.waiting_for_bm_exchange_bm)
        or (_has_pending_action_choice(game) and not game.waiting_for_bm_exchange_bm)
    ):
        return bm_tensor

    if game.waiting_for_bm_exchange_bm:
        if game.exchange_target_player is not None:
            for bm in game.exchange_target_player.used_bonus_markers:
                bm_index = BONUS_MARKER_SLOT_BY_TYPE.get(bm.type)
                if bm_index is not None:
                    bm_tensor[bm_index] = 1
    else:
        for bm in game.current_player.bonus_markers:
            bm_index = BONUS_MARKER_SLOT_BY_TYPE.get(bm.type)
            if bm_index is not None:
                if bm.type == "SwapOffice":
                    for city in game.selected_map.cities:
                        if city.check_if_eligible_to_swap_offices(game.current_player, game):
                            bm_tensor[bm_index] = 1
                elif bm.type == "ExchangeBonusMarker":
                    if any(
                        player is not game.current_player and player.used_bonus_markers
                        for player in game.players
                    ):
                        bm_tensor[bm_index] = 1
                elif bm.type == "UpgradeAbility":
                    if any(
                        getattr(game.current_player, ability) != maximum
                        for ability, maximum in UPGRADE_MAX_VALUES.items()
                    ):
                        bm_tensor[bm_index] = 1
                elif bm.type in ("Tribute4EstablishingTP", "BlockTradeRoute"):
                    if game.current_player.personal_supply_squares > 0:
                        bm_tensor[bm_index] = 1
                else:
                    bm_tensor[bm_index] = 1

    return bm_tensor


def mask_buy_tile(game):
    current_player = game.current_player

    buy_tile_tensor = _empty_mask(9)
    payment_slot_by_type = {
        marker_type: slot for slot, marker_type in enumerate(BONUS_MARKER_PAYMENT_TYPES)
    }

    if game.pending_income_favour_owner is not None:
        owner = game.pending_income_favour_owner
        buy_tile_tensor[0] = int(owner.general_stock_squares > 0)
        buy_tile_tensor[1] = int(owner.general_stock_circles > 0)
        buy_tile_tensor[2] = 1
        return buy_tile_tensor

    if game.waiting_for_buy_tile_with_bm:
        for bm in current_player.bonus_markers:
            if bm is game.first_bm_to_spend_on_tile:
                continue
            bm_index = payment_slot_by_type.get(bm.type)
            if bm_index is not None:
                buy_tile_tensor[bm_index] = 1
        return buy_tile_tensor

    # Buying an Emperor's Favour tile is a new turn-level interaction. It
    # cannot begin while another bonus-marker or piece workflow is waiting for
    # its required follow-up choice.
    if game.pending_workflows:
        return buy_tile_tensor

    if (
        game.use_emperors_favour
        and current_player.actions_remaining == current_player.actions_at_turn_start
        and len(current_player.bonus_markers) >= 2
    ):
        for tile in game.tile_pool:
            tile_index = TILE_SLOT_BY_TYPE.get(tile)
            if tile_index is not None:
                buy_tile_tensor[tile_index] = 1

    return buy_tile_tensor


def mask_replace_bm(game):
    max_num_routes = MAX_ROUTES
    replace_bm_tensor = _empty_mask(max_num_routes)

    if (
        game.current_player.actions_remaining == 0
        and game.replace_bonus_marker > 0
        and game.current_player.ending_turn
    ):
        for route_idx, route in enumerate(game.selected_map.routes):
            if (
                not (route.bonus_marker or route.permanent_bonus_marker)
                and not (game.map_num == 3 and route.region in ("Wales", "Scotland"))
                and not route.has_tradesmen()
                and route.has_empty_office_in_cities()
            ):
                replace_bm_tensor[route_idx] = 1

    return replace_bm_tensor


def mask_bm_city_actions(game):
    bm_city_tensor = _empty_mask(MAX_CITIES)
    if game.waiting_for_bm_exchange_bm and game.exchange_target_player is None:
        for index, player in enumerate(game.players):
            if player is not game.current_player and player.used_bonus_markers:
                bm_city_tensor[index] = 1
        return bm_city_tensor

    if game.waiting_for_bm_swap_office:
        pairs = [
            (city, pair)
            for city in game.selected_map.cities
            for pair in city.eligible_swap_pairs(game.current_player, game)
        ]
        bm_city_tensor[: len(pairs)] = [True] * len(pairs)
        return bm_city_tensor

    if game.waiting_for_bm_green_city:
        choices = [
            (city, shape)
            for city in game.selected_map.cities
            if city.color == DARK_GREEN
            for shape in ("square", "circle")
            if game.current_player.has_personal_supply(shape)
        ]
        bm_city_tensor[: len(choices)] = [True] * len(choices)
        return bm_city_tensor

    return bm_city_tensor


def mask_bm_upgrade_ability(game):
    bm_upgrade_tensor = _empty_mask(5)

    if not game.waiting_for_bm_upgrade_ability:
        return bm_upgrade_tensor

    for upgrade_idx, upgrade_city in enumerate(game.selected_map.upgrade_cities):
        upgrade_type = upgrade_city.upgrade_type
        current_player_value = getattr(game.current_player, upgrade_type.lower())
        max_value = UPGRADE_MAX_VALUES.get(upgrade_type)

        if current_player_value != max_value:
            bm_upgrade_tensor[upgrade_idx] = 1

    return bm_upgrade_tensor


def mask_end_turn(game):
    end_turn_tensor = _empty_mask(1)

    if (
        game.waiting_for_bm_move3 or game.waiting_for_bm_move_any_2
    ) and game.current_player.pieces_to_pickup > 0:
        end_turn_tensor[0] = 1
        return end_turn_tensor

    if (
        game.waiting_for_displaced_player
        and game.displaced_player.played_displaced_shape
        and not game.displaced_player.player.holding_pieces
    ):
        end_turn_tensor[0] = 1
        return end_turn_tensor

    if (
        game.current_player.actions_remaining > 0
        or game.current_player.ending_turn
        or game.current_player.holding_pieces
        or _has_pending_post_workflow(game)
        or _has_pending_action_choice(game)
    ):
        return end_turn_tensor

    if (_has_usable_bonus_marker(game) and game.current_player.actions_remaining == 0) or (
        game.replace_bonus_marker > 0 and game.current_player.actions_remaining == 0
    ):
        end_turn_tensor[0] = 1

    if game.replace_bonus_marker > 0:
        end_turn_tensor[0] = 1

    return end_turn_tensor


def _has_usable_bonus_marker(game):
    if not game.current_player.bonus_markers:
        return False

    for bm in game.current_player.bonus_markers:
        if bm.type == "PlaceAdjacent":
            continue
        elif bm.type == "SwapOffice":
            # Check if there's a city where the player is eligible to swap offices
            swap_office_possible = False
            for city in game.selected_map.cities:
                if city.check_if_eligible_to_swap_offices(game.current_player, game):
                    swap_office_possible = True
                    break
            if not swap_office_possible:
                continue
        elif bm.type == "ExchangeBonusMarker":
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
