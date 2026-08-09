from map_data.constants import DARK_GREEN
from game.action_schema import (
    BONUS_MARKER_PAYMENT_TYPES,
    BONUS_MARKER_SLOT_BY_TYPE,
    TILE_TYPES,
)
from game.game_actions import (
    InvalidActionError,
    claim_post_action,
    displace_action,
    move_action,
    displace_claim,
    can_pick_up_displacement_fallback,
    can_place_displacement_piece,
    optional_displacement_piece_available,
    select_optional_displaced_shape,
    finish_displacement,
    assign_new_bonus_marker_on_route,
    claim_route_for_office,
    claim_route_for_additional_office,
    claim_route_for_upgrade,
    claim_route_for_points,
    buy_tile,
)
from game.turn_state import TurnPhase


def _raise_invalid_action(game, route=None):
    route_description = "unknown route"
    if route is not None:
        route_description = " - ".join(city.name for city in route.cities)

    raise InvalidActionError(
        f"Invalid action on {route_description}; "
        f"current_player={game.current_player_index}, active_player={game.active_player}"
    )


def resolve_post_interaction(game, post_slot, post_type):
    current_player = game.current_player
    numbered_posts = enumerate(
        (route, post) for route in game.selected_map.routes for post in route.posts
    )
    selected = next((item for slot, item in numbered_posts if slot == post_slot), None)
    if selected is None:
        raise InvalidActionError(f"Post slot does not exist: {post_slot}")
    selected_route, selected_post = selected

    is_post_owned = selected_post.is_owned()
    is_post_empty = not is_post_owned
    can_displace = (
        current_player.personal_supply_squares + current_player.personal_supply_circles > 1
    )

    if game.waiting_for_displaced_player:
        if (
            selected_post.owner == game.displaced_player.player
            and game.displaced_player.is_general_stock_empty()
            and game.displaced_player.is_personal_supply_empty()
        ):
            displace_claim(game, selected_post, post_type)
        elif selected_post in game.all_empty_posts:
            displace_claim(game, selected_post, post_type)
        else:
            _raise_invalid_action(game, selected_route)

    elif game.waiting_for_bm_move_any_2:
        move_action(game, selected_route, selected_post, post_type)

    elif game.waiting_for_place2_from_route:
        move_action(game, selected_route, selected_post, post_type)

    elif game.waiting_for_place2_in_scotland_or_wales:
        if not selected_post.is_owned() and selected_post.region in ("Scotland", "Wales"):
            move_action(game, selected_route, selected_post, post_type)
        else:
            _raise_invalid_action(game, selected_route)

    elif game.waiting_for_bm_move3:
        if selected_post.owner != current_player:
            move_action(game, selected_route, selected_post, post_type)
        else:
            _raise_invalid_action(game, selected_route)
    elif game.waiting_for_bm_tribute_trading_post:
        current_player.personal_supply_squares -= 1
        selected_route.establish_tribute_on_route(current_player)
        game.waiting_for_bm_tribute_trading_post = False
    elif game.waiting_for_bm_block_trade_route:
        current_player.personal_supply_squares -= 1
        selected_route.establish_blocked_route(current_player)
        game.waiting_for_bm_block_trade_route = False
    else:
        post_type_available = any(piece[0] == post_type for piece in current_player.holding_pieces)

        if current_player.holding_pieces:
            if (
                is_post_empty
                and post_type_available
                and (
                    selected_post.required_shape is None
                    or selected_post.required_shape == post_type
                )
            ):
                move_action(game, selected_route, selected_post, post_type)
            elif (
                is_post_owned
                and selected_post.owner == current_player
                and current_player.pieces_to_pickup > 0
                and len(current_player.holding_pieces) < current_player.book
            ):
                move_action(game, selected_route, selected_post, post_type)
            else:
                _raise_invalid_action(game, selected_route)
        elif is_post_owned and selected_post.owner == current_player:
            move_action(game, selected_route, selected_post, post_type)
        elif is_post_empty and game.check_brown_blue_priv(selected_route):
            if current_player.has_personal_supply(post_type) and (
                selected_post.required_shape is None or selected_post.required_shape == post_type
            ):
                claim_post_action(game, selected_route, selected_post, post_type)
            else:
                _raise_invalid_action(game, selected_route)
        elif (
            is_post_owned
            and selected_post.owner != current_player
            and game.check_brown_blue_priv(selected_route)
            and can_displace
        ):
            displace_action(game, selected_post, selected_route, post_type)
        else:
            _raise_invalid_action(game, selected_route)


def resolve_route_interaction(game, route_idx, interaction_slot):
    """Resolve one structured interaction with a completed route."""
    route = game.selected_map.routes[route_idx]

    if interaction_slot == 0:
        if route.is_controlled_by(game.current_player):
            claim_route_for_points(game, route)
        else:
            _raise_invalid_action(game, route)

    elif interaction_slot <= 2:
        city_idx = interaction_slot - 1
        city = route.cities[city_idx]
        if city.has_empty_office() and route.is_controlled_by(game.current_player):
            claim_route_for_office(game, city, route)
        else:
            _raise_invalid_action(game, route)

    elif interaction_slot <= 6:
        adjusted_index = interaction_slot - 3
        city_idx, upgrade_idx = divmod(adjusted_index, 2)

        city = route.cities[city_idx]
        if game.waiting_for_bm_place_adjacent:
            shape = ("square", "circle")[upgrade_idx]
            claim_route_for_additional_office(game, city, route, shape)
            game.check_for_game_end()
            return
        special_city = next(
            (
                candidate
                for candidate in route.cities
                if "SpecialPrestigePoints" in candidate.upgrade_city_type
            ),
            None,
        )
        if special_city is not None:
            prestige_value = [7, 8, 9, 11][adjusted_index % 4]
            if route.is_controlled_by(game.current_player):
                claim_route_for_upgrade(
                    game,
                    special_city,
                    route,
                    "SpecialPrestigePoints",
                    prestige_value=prestige_value,
                )
            else:
                _raise_invalid_action(game, route)
        elif city.upgrade_city_type and len(city.upgrade_city_type) > upgrade_idx:
            upgrade_choice = city.upgrade_city_type[upgrade_idx]
            if route.is_controlled_by(game.current_player):
                claim_route_for_upgrade(game, city, route, upgrade_choice)
            else:
                _raise_invalid_action(game, route)
        else:
            _raise_invalid_action(game, route)

    else:
        raise InvalidActionError(f"Invalid route interaction slot: {interaction_slot}")

    game.check_for_game_end()


def resolve_income_interaction(game, index):
    if game.pending_britannia_place2:
        player = game.current_player
        requested_circles = index
        board_posts = [
            post
            for route in game.selected_map.routes
            for post in route.posts
            if post.owner == player
        ]
        pools = [
            ["general_stock", player.general_stock_squares, player.general_stock_circles, []],
            ["personal_supply", player.personal_supply_squares, player.personal_supply_circles, []],
            [
                "board",
                sum(post.owner_piece_shape == "square" for post in board_posts),
                sum(post.owner_piece_shape == "circle" for post in board_posts),
                board_posts,
            ],
        ]
        remaining_slots = 2
        quotas = []
        for _source, squares, circles, _posts in pools:
            quota = min(remaining_slots, squares + circles)
            quotas.append(quota)
            remaining_slots -= quota
        if remaining_slots:
            raise InvalidActionError("Fewer than two pieces are available")

        circle_allocations = None
        for first in range(quotas[0] + 1):
            for second in range(quotas[1] + 1):
                third = requested_circles - first - second
                candidates = (first, second, third)
                if 0 <= third <= quotas[2] and all(
                    circles >= selected_circles and squares >= quota - selected_circles
                    for (_source, squares, circles, _posts), quota, selected_circles in zip(
                        pools, quotas, candidates
                    )
                ):
                    circle_allocations = candidates
                    break
            if circle_allocations is not None:
                break
        if circle_allocations is None:
            raise InvalidActionError("Selected Britannia piece composition is unavailable")

        selected = []
        for (source, _squares, _circles, posts), quota, circles in zip(
            pools, quotas, circle_allocations
        ):
            shapes = ["circle"] * circles + ["square"] * (quota - circles)
            for shape in shapes:
                if source == "board":
                    post = next(post for post in posts if post.owner_piece_shape == shape)
                    posts.remove(post)
                    selected.append((shape, player, post.region, source, post))
                else:
                    selected.append((shape, player, None, source))

        holding = []
        for piece in selected:
            shape, _owner, region, source, *post = piece
            if source == "general_stock":
                if shape == "square":
                    player.general_stock_squares -= 1
                else:
                    player.general_stock_circles -= 1
            elif source == "personal_supply":
                if shape == "square":
                    player.personal_supply_squares -= 1
                else:
                    player.personal_supply_circles -= 1
            else:
                post[0].reset_post()
            holding.append((shape, player, region))
        player.holding_pieces = holding
        player.pieces_to_pickup = 0
        game.pending_britannia_place2 = False
        game.waiting_for_place2_in_scotland_or_wales = True
        return

    if game.pending_route_piece_choices:
        available_circles = sum(
            shape == "circle" for shape, _owner, _region in game.pending_route_piece_choices
        )
        available_squares = len(game.pending_route_piece_choices) - available_circles
        circles = index
        squares = 2 - circles
        if circles > available_circles or squares > available_squares:
            raise InvalidActionError("Selected route-piece composition is unavailable")
        chosen = []
        remaining = list(game.pending_route_piece_choices)
        for shape, count in (("circle", circles), ("square", squares)):
            for _ in range(count):
                piece = next(piece for piece in remaining if piece[0] == shape)
                remaining.remove(piece)
                chosen.append(piece)
                if shape == "circle":
                    game.current_player.general_stock_circles -= 1
                else:
                    game.current_player.general_stock_squares -= 1
        game.current_player.holding_pieces = chosen
        game.current_player.pieces_to_place = 2
        game.current_player.pieces_to_pickup = 0
        game.pending_route_piece_choices = []
        game.waiting_for_place2_from_route = True
        return

    if game.pending_tribute_income_owners:
        game.resolve_tribute_income(index)
        return

    current_player = game.current_player
    num_circles = current_player.general_stock_circles
    num_squares = current_player.general_stock_squares

    if num_circles == 0 and num_squares == 0:
        raise InvalidActionError("Income selected with an empty general stock")

    if num_circles == 0:
        current_player.income_action(min(num_squares, current_player.bank), 0)
        game.begin_income_favour_response(current_player)
        return

    if index == 0:
        squares_to_take = min(num_squares, current_player.bank)
        circles_to_take = 0
    else:
        circles_to_take = min(num_circles, index)
        squares_to_take = min(num_squares, current_player.bank - circles_to_take)

    current_player.income_action(squares_to_take, circles_to_take)
    game.begin_income_favour_response(current_player)


def resolve_bonus_marker_interaction(game, index):
    if game.waiting_for_buy_tile_with_bm and game.tile_to_buy is not None:
        resolve_tile_interaction(game, index)
        return
    current_player = game.current_player

    if game.waiting_for_bm_exchange_bm:
        target = game.exchange_target_player
        if target is None:
            raise InvalidActionError("An exchange target must be selected first")
        exchanged_marker = next(
            (
                marker
                for marker in target.used_bonus_markers
                if BONUS_MARKER_SLOT_BY_TYPE.get(marker.type) == index
            ),
            None,
        )
        if exchanged_marker is None:
            raise InvalidActionError("Selected used bonus marker is unavailable")
        target.used_bonus_markers.remove(exchanged_marker)
        current_player.bonus_markers.append(exchanged_marker)
        game.pending_exchange_marker.owner = target
        target.used_bonus_markers.append(game.pending_exchange_marker)
        game.pending_exchange_marker = None
        game.exchange_target_player = None
        game.waiting_for_bm_exchange_bm = False
        return
    else:
        selected_bm = next(
            (
                marker
                for marker in current_player.bonus_markers
                if BONUS_MARKER_SLOT_BY_TYPE.get(marker.type) == index
            ),
            None,
        )
        if selected_bm is None:
            raise InvalidActionError("Selected bonus marker is unavailable")
        if selected_bm.type == "SwapOffice":
            game.waiting_for_bm_swap_office = True
        elif selected_bm.type == "Move3":
            selected_bm.handle_move3(game)
        elif selected_bm.type == "UpgradeAbility":
            game.waiting_for_bm_upgrade_ability = True
        elif selected_bm.type == "3Actions":
            selected_bm.handle_3_actions(current_player)
        elif selected_bm.type == "4Actions":
            selected_bm.handle_4_actions(current_player)

        elif selected_bm.type == "ExchangeBonusMarker":
            game.waiting_for_bm_exchange_bm = True
            game.pending_exchange_marker = selected_bm
        elif (
            selected_bm.type == "Tribute4EstablishingTP"
            and current_player.personal_supply_squares > 0
        ):
            game.waiting_for_bm_tribute_trading_post = True
        elif selected_bm.type == "BlockTradeRoute" and current_player.personal_supply_squares > 0:
            game.waiting_for_bm_block_trade_route = True

        game.current_player.bonus_markers.remove(selected_bm)
        if selected_bm.type != "ExchangeBonusMarker":
            selected_bm.owner = current_player
            current_player.used_bonus_markers.append(selected_bm)
    return


def resolve_additional_office_marker(game):
    if game.waiting_for_buy_tile_with_bm and game.tile_to_buy is not None:
        resolve_tile_interaction(game, 8)
        return
    if not any(marker.type == "PlaceAdjacent" for marker in game.current_player.bonus_markers):
        raise InvalidActionError("No Additional Trading Post marker is available")
    game.waiting_for_bm_place_adjacent = True


def resolve_tile_interaction(game, index):
    current_player = game.current_player
    if game.pending_income_favour_owner is not None:
        game.resolve_income_favour({0: "square", 1: "circle", 2: None}[index])
        return

    if game.waiting_for_buy_tile_with_bm:
        marker_type = BONUS_MARKER_PAYMENT_TYPES[index]
        marker = next(
            (
                candidate
                for candidate in current_player.bonus_markers
                if candidate.type == marker_type and candidate is not game.first_bm_to_spend_on_tile
            ),
            None,
        )
        if marker is None:
            raise InvalidActionError("Selected bonus-marker payment is unavailable")
        if game.first_bm_to_spend_on_tile is None:
            game.first_bm_to_spend_on_tile = marker
        else:
            buy_tile(
                game,
                game.tile_to_buy,
                game.first_bm_to_spend_on_tile,
                marker,
            )
            game.waiting_for_buy_tile_with_bm = False
            game.first_bm_to_spend_on_tile = None
            game.tile_to_buy = None
        return

    if len(current_player.bonus_markers) == 2:
        buy_tile(
            game,
            TILE_TYPES[index],
            current_player.bonus_markers[0],
            current_player.bonus_markers[1],
        )
    elif len(current_player.bonus_markers) > 2:
        game.waiting_for_buy_tile_with_bm = True
        game.tile_to_buy = TILE_TYPES[index]
    else:
        raise InvalidActionError("Two unused bonus markers are required")

    return


def resolve_replacement_marker(game, index):
    current_player = game.current_player
    if not (
        current_player.actions_remaining == 0
        and current_player.ending_turn
        and game.replace_bonus_marker > 0
    ):
        raise InvalidActionError("Bonus-marker replacement is not pending")
    if not 0 <= index < len(game.selected_map.routes):
        raise InvalidActionError(f"Route slot does not exist: {index}")

    route = game.selected_map.routes[index]
    if (
        route.bonus_marker
        or route.permanent_bonus_marker
        or route.has_tradesmen()
        or not route.has_empty_office_in_cities()
    ):
        _raise_invalid_action(game, route)

    assign_new_bonus_marker_on_route(game, route)
    if game.replace_bonus_marker < 0:
        raise InvalidActionError("Bonus-marker replacement count became negative")
    if game.replace_bonus_marker == 0:
        current_player.ending_turn = False
        game.switch_player_if_needed()


def resolve_player_interaction(game, player_slot):
    if not game.waiting_for_bm_exchange_bm or game.exchange_target_player is not None:
        raise InvalidActionError("Player interaction has no active workflow")
    target = game.players[player_slot]
    if target is game.current_player or not target.used_bonus_markers:
        raise InvalidActionError("Selected player has no used marker to exchange")
    game.exchange_target_player = target


def resolve_city_interaction(game, action):
    if game.waiting_for_bm_swap_office:
        catalogue = [
            (city, (left, left + 1))
            for city in game.selected_map.cities
            for left in range(len(city.offices) - 1)
        ]
        city, pair = catalogue[action.city_interaction_slot]
        if not city.swap_office_pair(game.current_player, pair, game):
            raise InvalidActionError("Trading-post exchange is no longer legal")
        game.waiting_for_bm_swap_office = False
        return

    if game.waiting_for_bm_green_city:
        catalogue = [
            (city, shape)
            for city in game.selected_map.cities
            if city.color == DARK_GREEN
            for shape in ("square", "circle")
        ]
        city, shape = catalogue[action.city_interaction_slot - 46]
        if not city.claim_green_city(game, shape):
            raise InvalidActionError("Green-city choice is no longer legal")
        game.waiting_for_bm_green_city = False
        return

    raise InvalidActionError("City interaction has no active workflow")


def resolve_ability_interaction(game, index):
    for upgrade_idx, upgrade_city in enumerate(game.selected_map.upgrade_cities):
        if upgrade_idx == index:
            if game.waiting_for_bm_upgrade_ability:
                upgrade_type = upgrade_city.upgrade_type
                game.current_player.perform_upgrade(upgrade_type)
                game.waiting_for_bm_upgrade_ability = False


def resolve_control_interaction(game):
    if game.waiting_for_bm_move3 or game.waiting_for_bm_move_any_2:
        game.current_player.pieces_to_pickup = 0
        if not game.current_player.holding_pieces:
            game.current_player.finish_move()
            game.waiting_for_bm_move3 = False
            game.waiting_for_bm_move_any_2 = False
        return

    if game.turn_phase == TurnPhase.DISPLACEMENT:
        finish_displacement(game)
        game.switch_player_if_needed()
        return

    if game.current_player.actions_remaining != 0:
        raise InvalidActionError("Cannot end a turn while ordinary actions remain")

    game.current_player.ending_turn = True
    if game.replace_bonus_marker == 0:
        game.switch_player_if_needed()
