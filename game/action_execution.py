from game.action_resolvers import (
    resolve_ability_interaction,
    resolve_additional_office_marker,
    resolve_bonus_marker_interaction,
    resolve_city_interaction,
    resolve_control_interaction,
    resolve_income_interaction,
    resolve_player_interaction,
    resolve_post_interaction,
    resolve_replacement_marker,
    resolve_route_interaction,
    resolve_tile_interaction,
)
from game.action_schema import BONUS_MARKER_PAYMENT_TYPES, BONUS_MARKER_TYPES
from game.game_actions import select_optional_displaced_shape
from game.turn_state import TurnPhase
from game.structured_actions import (
    AbilityInteraction,
    BonusMarkerInteraction,
    CityInteraction,
    ControlInteraction,
    IncomeInteraction,
    PieceShape,
    PlayerInteraction,
    PostInteraction,
    RouteInteraction,
    SupplyInteraction,
    TileInteraction,
)


def execute_action(game, action):
    """Execute one already-validated structured interaction directly."""
    if isinstance(action, PostInteraction):
        shape = "circle" if action.shape is PieceShape.MERCHANT else "square"
        resolve_post_interaction(game, action.post_slot, shape)
    elif isinstance(action, RouteInteraction):
        if game.turn_phase == TurnPhase.REPLACE_BONUS_MARKERS:
            resolve_replacement_marker(game, action.route_slot)
        else:
            resolve_route_interaction(game, action.route_slot, action.interaction_slot)
    elif isinstance(action, IncomeInteraction):
        resolve_income_interaction(game, action.merchant_count)
    elif isinstance(action, BonusMarkerInteraction):
        if action.marker_slot == 8:
            resolve_additional_office_marker(game)
        else:
            exchanged_start = len(BONUS_MARKER_PAYMENT_TYPES)
            marker_type_slot = action.marker_slot
            if marker_type_slot >= exchanged_start:
                marker_type_slot = (marker_type_slot - exchanged_start) % len(BONUS_MARKER_TYPES)
            resolve_bonus_marker_interaction(game, marker_type_slot)
    elif isinstance(action, TileInteraction):
        resolve_tile_interaction(game, action.tile_slot)
    elif isinstance(action, PlayerInteraction):
        resolve_player_interaction(game, action.player_slot)
    elif isinstance(action, CityInteraction):
        resolve_city_interaction(game, action)
    elif isinstance(action, AbilityInteraction):
        resolve_ability_interaction(game, action.ability_slot)
    elif isinstance(action, SupplyInteraction):
        select_optional_displaced_shape(game)
    elif isinstance(action, ControlInteraction):
        resolve_control_interaction(game)
    else:
        raise TypeError(f"Unsupported structured action: {action!r}")

    game.complete_deferred_game_end_if_ready()

    if not (
        game.waiting_for_bm_swap_office
        or game.waiting_for_bm_upgrade_ability
        or game.waiting_for_bm_move_any_2
        or game.waiting_for_bm_move3
        or game.waiting_for_bm_exchange_bm
        or game.waiting_for_bm_tribute_trading_post
        or game.waiting_for_bm_block_trade_route
        or game.waiting_for_bm_green_city
        or game.waiting_for_place2_in_scotland_or_wales
    ):
        game.switch_player_if_needed()
