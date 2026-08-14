"""Human-readable descriptions for codec-backed legal interactions."""

from __future__ import annotations

from game.action_codec import DEFAULT_ACTION_CODEC
from game.action_schema import (
    ADDITIONAL_TRADING_POST_SLOT,
    BONUS_MARKER_PAYMENT_TYPES,
    BONUS_MARKER_TYPES,
    EXCHANGED_BONUS_MARKER_START,
    GREEN_CITY_SLOT_START,
    PRESTIGE_VALUES,
    ROUTE_COMPLETE_SLOT,
    ROUTE_OFFICE_SLOT_START,
    ROUTE_OUTCOME_SLOT_START,
    TILE_TYPES,
    city_pair_catalogue,
    green_city_catalogue,
)
from game.structured_actions import (
    AbilityInteraction,
    BonusMarkerInteraction,
    CityInteraction,
    ControlInteraction,
    IncomeInteraction,
    PlayerInteraction,
    PostInteraction,
    RouteInteraction,
    SupplyInteraction,
    TileInteraction,
)
from game.turn_state import TurnPhase

BONUS_MARKER_LABELS = dict(
    zip(
        BONUS_MARKER_TYPES,
        (
            "Swap Office",
            "Move 3",
            "Upgrade Ability",
            "+3 Actions",
            "+4 Actions",
            "Exchange Bonus Marker",
            "Tribute Trading Post",
            "Block Trade Route",
        ),
        strict=True,
    )
)
TILE_LABELS = dict(
    zip(
        TILE_TYPES,
        (
            "Buy Displace Anywhere",
            "Buy +1 Action",
            "Buy Income Favour",
            "Buy +1 Displaced Piece",
            "Buy City Scoring",
            "Buy Ability Scoring",
        ),
        strict=True,
    )
)


def fit_text(font, text, max_width):
    """Truncate a rendered label to the available logical width."""
    if font.size(text)[0] <= max_width:
        return text
    suffix = "…"
    while text and font.size(text + suffix)[0] > max_width:
        text = text[:-1]
    return text.rstrip() + suffix


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
    """Describe one action index using its decoded structured interaction."""
    action = DEFAULT_ACTION_CODEC.decode(index)
    if isinstance(action, PostInteraction):
        return f"Post {action.post_slot}: {action.shape.value.title()}"
    if isinstance(action, RouteInteraction):
        return _route_action_label(action, game)
    if isinstance(action, IncomeInteraction):
        return _income_action_label(action, game)
    if isinstance(action, BonusMarkerInteraction):
        if action.marker_slot == ADDITIONAL_TRADING_POST_SLOT:
            if getattr(game, "waiting_for_buy_tile_with_bm", False):
                return "Pay Additional Trading Post"
            if getattr(game, "exchange_target_player", None):
                return "Take used Additional Trading Post"
            return "Use Additional Trading Post"
        marker_slot = action.marker_slot
        if marker_slot >= EXCHANGED_BONUS_MARKER_START:
            marker_slot = (marker_slot - EXCHANGED_BONUS_MARKER_START) % len(BONUS_MARKER_TYPES)
        marker_type = BONUS_MARKER_TYPES[marker_slot]
        if getattr(game, "waiting_for_buy_tile_with_bm", False):
            verb = "Pay"
        else:
            verb = "Take used" if getattr(game, "exchange_target_player", None) else "Use"
        return f"{verb} {BONUS_MARKER_LABELS[marker_type]}"
    if isinstance(action, TileInteraction):
        return _tile_action_label(action, game)
    if isinstance(action, PlayerInteraction):
        return f"Exchange with Player {action.player_slot + 1}"
    if isinstance(action, CityInteraction):
        return _city_context_label(action, game)
    if isinstance(action, AbilityInteraction):
        if game is not None and action.ability_slot < len(game.selected_map.upgrade_cities):
            upgrade = game.selected_map.upgrade_cities[action.ability_slot]
            return f"Upgrade {upgrade.upgrade_type}"
        return f"Ability choice {action.ability_slot + 1}"
    if isinstance(action, ControlInteraction):
        if game is not None and game.turn_phase == TurnPhase.DISPLACEMENT:
            return "Finish displacement (decline optional pieces)"
        return "Finish / End turn"
    if isinstance(action, SupplyInteraction):
        if game is not None and game.turn_phase == TurnPhase.DISPLACEMENT:
            shape = game.displaced_player.displaced_shape
            piece = "Merchant" if shape == "circle" else "Trader"
            return f"Place optional {piece} before displaced piece"
        return "Select player supply"
    return f"Action {index}"


def _income_action_label(action, game) -> str:
    circles = action.merchant_count
    if game is not None and game.turn_phase == TurnPhase.PERMANENT_ROUTE_PIECE_SELECTION:
        return _piece_mix_label(total=2, circles=circles)
    if game is not None and game.turn_phase == TurnPhase.TRIBUTE_INCOME_RESPONSE:
        owner = game.pending_tribute_income_owners[0]
        total = min(2, owner.general_stock_squares + owner.general_stock_circles)
        return _piece_mix_label(total=total, circles=circles)
    if game is not None:
        player = game.current_player
        selected_circles = min(player.general_stock_circles, circles)
        selected_squares = min(player.general_stock_squares, player.bank - selected_circles)
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


def _route_action_label(action, game) -> str:
    if game is None:
        return f"Route {action.route_slot}: interaction {action.interaction_slot}"
    route = game.selected_map.routes[action.route_slot]
    if game.turn_phase == TurnPhase.REPLACE_BONUS_MARKERS:
        return f"Place marker: {route.cities[0].name}—{route.cities[1].name}"
    if action.interaction_slot == ROUTE_COMPLETE_SLOT:
        return f"Complete route (no office): {route.cities[0].name}—{route.cities[1].name}"
    if action.interaction_slot < ROUTE_OUTCOME_SLOT_START:
        endpoint = action.interaction_slot - ROUTE_OFFICE_SLOT_START
        return f"Office in {route.cities[endpoint].name}"

    route_choice = action.interaction_slot - ROUTE_OUTCOME_SLOT_START
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
        return f"Prestige {PRESTIGE_VALUES[route_choice]}: {special_city.name}"
    city_index, upgrade_index = divmod(route_choice, 2)
    city = route.cities[city_index]
    if upgrade_index < len(city.upgrade_city_type):
        return f"Upgrade {city.upgrade_city_type[upgrade_index]} in {city.name}"
    return f"Route choice: {route.cities[0].name}—{route.cities[1].name}"


def _tile_action_label(action, game) -> str:
    choice = action.tile_slot
    if getattr(game, "pending_income_favour_owner", None) is not None:
        labels = ("Income favour: Trader", "Income favour: Merchant", "Decline income favour")
        return labels[choice] if choice < len(labels) else f"Income favour choice {choice + 1}"
    if getattr(game, "waiting_for_buy_tile_with_bm", False):
        marker_type = BONUS_MARKER_PAYMENT_TYPES[choice]
        if marker_type == "PlaceAdjacent":
            return "Pay Additional Trading Post"
        return f"Pay {BONUS_MARKER_LABELS[marker_type]}"
    if choice < len(TILE_TYPES):
        return TILE_LABELS[TILE_TYPES[choice]]
    return f"Tile choice {choice + 1}"


def _city_context_label(action, game) -> str:
    if game is None:
        return f"City choice {action.city_interaction_slot}"
    if game.waiting_for_bm_swap_office:
        catalogue = city_pair_catalogue(game.selected_map.cities)
        city, pair = catalogue[action.city_interaction_slot]
        return f"Swap offices {pair[0] + 1}/{pair[1] + 1} in {city.name}"
    if game.waiting_for_bm_green_city:
        catalogue = green_city_catalogue(game.selected_map.cities)
        city, shape = catalogue[action.city_interaction_slot - GREEN_CITY_SLOT_START]
        return f"Green office in {city.name}: {shape.title()}"
    return f"City choice {action.city_interaction_slot + 1}"
