"""Human-readable descriptions for the engine's indexed legal actions."""

from __future__ import annotations

from game.turn_state import TurnPhase
from map_data.constants import DARK_GREEN, MAX_ROUTES

BONUS_MARKER_NAMES = (
    "Swap Office",
    "Move 3",
    "Upgrade Ability",
    "+3 Actions",
    "+4 Actions",
    "Exchange Bonus Marker",
    "Tribute Trading Post",
    "Block Trade Route",
)


def phase_prompt(game) -> str:
    """Describe what the current phase requires from the active human."""
    prompts = {
        TurnPhase.ACTIONS: f"Choose an action — {game.current_player.actions_remaining} remaining",
        TurnPhase.DISPLACEMENT: "Resolve displacement on the nearest legal route",
        TurnPhase.MOVE_PIECES: "Finish moving every picked-up tradesman",
        TurnPhase.BONUS_MARKER_CHOICE: "Complete the selected bonus-marker effect",
        TurnPhase.BUY_TILE_PAYMENT: "Choose two unused bonus markers as payment",
        TurnPhase.INCOME_FAVOUR_RESPONSE: "Take a Trader, take a Merchant, or decline",
        TurnPhase.TRIBUTE_INCOME_RESPONSE: "Choose the two-piece Tribute income mix",
        TurnPhase.PLACE_ADJACENT_ROUTE: "Choose city and piece for Additional Trading Post",
        TurnPhase.PERMANENT_ROUTE_PIECE_SELECTION: "Choose the required two-piece mix",
        TurnPhase.REPLACE_BONUS_MARKERS: "Place every pending marker on a legal empty route",
        TurnPhase.TURN_COMPLETE: "Use an optional marker or finish the turn",
        TurnPhase.GAME_OVER: "Game over",
    }
    return prompts[game.turn_phase]


def action_label(index: int, game=None) -> str:
    """Translate one engine action index using its current contextual meaning."""
    if index < 121:
        return f"Post {index}: Trader"
    if index < 242:
        return f"Post {index - 121}: Merchant"
    if index < 522:
        return _route_action_label(index, game)
    if index < 527:
        return _income_action_label(index, game)
    if index < 535:
        verb = "Take used" if getattr(game, "exchange_target_player", None) else "Use"
        return f"{verb} {BONUS_MARKER_NAMES[index - 527]}"
    if index < 543:
        return _tile_action_label(index, game)
    if index < 583 and game is not None:
        route = game.selected_map.routes[index - 543]
        return f"Place marker: {route.cities[0].name}—{route.cities[1].name}"
    if index < 613:
        return _city_context_label(index, game)
    if index < 618 and game is not None:
        choice = index - 613
        if choice < len(game.selected_map.upgrade_cities):
            return f"Upgrade {game.selected_map.upgrade_cities[choice].upgrade_type}"
        return f"Ability choice {choice + 1}"
    if index == 618:
        return "Finish / End turn"
    if index == 619:
        return "Use Additional Trading Post"
    return f"Action {index}"


def _income_action_label(index: int, game) -> str:
    circles = index - 522
    if game is not None and game.turn_phase == TurnPhase.PERMANENT_ROUTE_PIECE_SELECTION:
        return _piece_mix_label(total=2, circles=circles)
    if game is not None and game.turn_phase == TurnPhase.TRIBUTE_INCOME_RESPONSE:
        owner = game.pending_tribute_income_owners[0]
        total = min(2, owner.general_stock_squares + owner.general_stock_circles)
        return _piece_mix_label(total=total, circles=circles)
    if game is not None:
        player = game.current_player
        selected_circles = min(player.general_stock_circles, circles)
        selected_squares = min(
            player.general_stock_squares,
            player.bank - selected_circles,
        )
        return "Income: " + _piece_mix_label(
            total=selected_squares + selected_circles,
            circles=selected_circles,
        )
    return f"Income choice {circles}"


def _piece_mix_label(*, total: int, circles: int) -> str:
    traders = total - circles
    parts = []
    if traders:
        parts.append(f"{traders} Trader{'s' if traders != 1 else ''}")
    if circles:
        parts.append(f"{circles} Merchant{'s' if circles != 1 else ''}")
    return " + ".join(parts)


def _route_action_label(index: int, game) -> str:
    if game is None:
        return f"Route action {index}"
    relative = index - 242
    if relative < MAX_ROUTES:
        route = game.selected_map.routes[relative]
        return f"Score route: {route.cities[0].name}—{route.cities[1].name}"
    if relative < MAX_ROUTES * 3:
        route_index, city_index = divmod(relative - MAX_ROUTES, 2)
        route = game.selected_map.routes[route_index]
        return f"Office in {route.cities[city_index].name}"

    route_index, route_choice = divmod(relative - MAX_ROUTES * 3, 4)
    route = game.selected_map.routes[route_index]
    if getattr(game, "waiting_for_bm_place_adjacent", False):
        city_index, shape_index = divmod(route_choice, 2)
        return (
            f"Additional {('Trader', 'Merchant')[shape_index]} office "
            f"in {route.cities[city_index].name}"
        )
    special_city = next(
        (city for city in route.cities if "SpecialPrestigePoints" in city.upgrade_city_type),
        None,
    )
    if special_city is not None:
        return f"Prestige {(7, 8, 9, 11)[route_choice]}: {special_city.name}"
    city_index, upgrade_index = divmod(route_choice, 2)
    city = route.cities[city_index]
    if upgrade_index < len(city.upgrade_city_type):
        return f"Upgrade {city.upgrade_city_type[upgrade_index]} in {city.name}"
    return f"Route choice: {route.cities[0].name}—{route.cities[1].name}"


def _tile_action_label(index: int, game) -> str:
    choice = index - 535
    if getattr(game, "pending_income_favour_owner", None) is not None:
        labels = ("Income favour: Trader", "Income favour: Merchant", "Decline income favour")
        return labels[choice] if choice < len(labels) else f"Income favour choice {choice + 1}"
    if getattr(game, "waiting_for_buy_tile_with_bm", False):
        return f"Pay {BONUS_MARKER_NAMES[choice]}"
    tiles = (
        "Buy Displace Anywhere",
        "Buy +1 Action",
        "Buy Income Favour",
        "Buy +1 Displaced Piece",
        "Buy City Scoring",
        "Buy Ability Scoring",
    )
    return tiles[choice] if choice < len(tiles) else f"Tile choice {choice + 1}"


def _city_context_label(index: int, game) -> str:
    if game is None:
        return f"City choice {index - 582}"
    choice = index - 583
    if getattr(game, "waiting_for_bm_exchange_bm", False):
        if getattr(game, "exchange_target_player", None) is None:
            return f"Exchange with Player {choice + 1}"
    if getattr(game, "waiting_for_bm_swap_office", False):
        pairs = [
            (city, pair)
            for city in game.selected_map.cities
            for pair in city.eligible_swap_pairs(game.current_player)
        ]
        if choice < len(pairs):
            city, pair = pairs[choice]
            return f"Swap offices {pair[0] + 1}/{pair[1] + 1} in {city.name}"
    if getattr(game, "waiting_for_bm_green_city", False):
        choices = [
            (city, shape)
            for city in game.selected_map.cities
            if city.color == DARK_GREEN
            for shape in ("square", "circle")
            if game.current_player.has_personal_supply(shape)
        ]
        if choice < len(choices):
            city, shape = choices[choice]
            return f"Green office in {city.name}: {shape.title()}"
    return f"City choice {choice + 1}"
