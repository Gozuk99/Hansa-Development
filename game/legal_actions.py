"""Structured legal interactions backed by the existing rules predicates."""

from game.action_legality import (
    mask_bm,
    mask_bm_city_actions,
    mask_bm_upgrade_ability,
    mask_buy_tile,
    mask_claim_route,
    mask_end_turn,
    mask_income_actions,
    mask_place_adjacent,
    mask_post_action,
    mask_replace_bm,
)
from game.action_schema import BONUS_MARKER_PAYMENT_TYPES, BONUS_MARKER_TYPES
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
from game.turn_state import TurnPhase
from map_data.constants import DARK_GREEN, MAX_POSTS, MAX_ROUTES


def _enabled(mask):
    return (index for index, enabled in enumerate(mask) if enabled)


def _city_pair_catalogue(game):
    return [
        (city, (left, left + 1))
        for city in game.selected_map.cities
        for left in range(len(city.offices) - 1)
    ]


def _green_catalogue(game):
    return [
        (city, shape)
        for city in game.selected_map.cities
        if city.color == DARK_GREEN
        for shape in ("square", "circle")
    ]


def _post_actions(game):
    for local in _enabled(mask_post_action(game)):
        shape = PieceShape.MERCHANT if local >= MAX_POSTS else PieceShape.TRADER
        yield PostInteraction(local % MAX_POSTS, shape)


def _route_actions(game):
    office_start = MAX_ROUTES
    outcome_start = MAX_ROUTES * 3
    for local in _enabled(mask_claim_route(game)):
        if local < office_start:
            yield RouteInteraction(local, 0)
        elif local < outcome_start:
            route, endpoint = divmod(local - office_start, 2)
            yield RouteInteraction(route, endpoint + 1)
        else:
            route, outcome = divmod(local - outcome_start, 4)
            yield RouteInteraction(route, outcome + 3)


def _tile_and_marker_actions(game):
    tile_choices = _enabled(mask_buy_tile(game))
    if game.turn_phase == TurnPhase.INCOME_FAVOUR_RESPONSE:
        yield from (TileInteraction(local) for local in tile_choices)
        return
    if game.turn_phase == TurnPhase.BUY_TILE_PAYMENT:
        interaction = BonusMarkerInteraction if game.tile_to_buy is not None else TileInteraction
        yield from (interaction(local) for local in tile_choices)
        return

    yield from (TileInteraction(local) for local in tile_choices)
    for local in _enabled(mask_bm(game)):
        if game.waiting_for_bm_exchange_bm and game.exchange_target_player is not None:
            opponents = [player for player in game.players if player is not game.current_player]
            opponent = opponents.index(game.exchange_target_player)
            yield BonusMarkerInteraction(
                len(BONUS_MARKER_PAYMENT_TYPES) + opponent * len(BONUS_MARKER_TYPES) + local
            )
        else:
            yield BonusMarkerInteraction(local)


def _income_actions(game):
    yield from (IncomeInteraction(local) for local in _enabled(mask_income_actions(game)))


def _replacement_actions(game):
    yield from (RouteInteraction(route, 0) for route in _enabled(mask_replace_bm(game)))


def _ability_actions(game):
    yield from (AbilityInteraction(local) for local in _enabled(mask_bm_upgrade_ability(game)))


def _city_actions(game):
    city_mask = mask_bm_city_actions(game)
    if game.waiting_for_bm_exchange_bm and game.exchange_target_player is None:
        yield from (PlayerInteraction(local) for local in _enabled(city_mask))
    elif game.waiting_for_bm_swap_office:
        eligible = [
            (city, pair)
            for city in game.selected_map.cities
            for pair in city.eligible_swap_pairs(game.current_player, game)
        ]
        catalogue = _city_pair_catalogue(game)
        yield from (
            CityInteraction(catalogue.index(choice))
            for local, choice in enumerate(eligible)
            if bool(city_mask[local])
        )
    elif game.waiting_for_bm_green_city:
        catalogue = _green_catalogue(game)
        eligible = [
            (city, shape)
            for city, shape in catalogue
            if game.current_player.has_personal_supply(shape)
        ]
        yield from (
            CityInteraction(46 + catalogue.index(choice))
            for local, choice in enumerate(eligible)
            if bool(city_mask[local])
        )


def _control_actions(game):
    if not bool(mask_end_turn(game)[0]):
        return
    finish_phases = {
        TurnPhase.DISPLACEMENT,
        TurnPhase.MOVE_PIECES,
        TurnPhase.BONUS_MARKER_CHOICE,
    }
    yield ControlInteraction(0 if game.turn_phase in finish_phases else 1)


def _supply_actions(game):
    if not bool(mask_place_adjacent(game)[0]):
        return
    if game.turn_phase == TurnPhase.DISPLACEMENT:
        yield SupplyInteraction(0)
    else:
        yield BonusMarkerInteraction(8)


PHASE_BUILDERS = {
    TurnPhase.ACTIONS: (
        _post_actions,
        _route_actions,
        _income_actions,
        _tile_and_marker_actions,
        _replacement_actions,
        _city_actions,
        _ability_actions,
        _control_actions,
        _supply_actions,
    ),
    TurnPhase.DISPLACEMENT: (_post_actions, _control_actions, _supply_actions),
    TurnPhase.MOVE_PIECES: (_post_actions,),
    TurnPhase.BONUS_MARKER_CHOICE: (
        _post_actions,
        _tile_and_marker_actions,
        _city_actions,
        _ability_actions,
        _control_actions,
    ),
    TurnPhase.BUY_TILE_PAYMENT: (_tile_and_marker_actions,),
    TurnPhase.INCOME_FAVOUR_RESPONSE: (_tile_and_marker_actions,),
    TurnPhase.TRIBUTE_INCOME_RESPONSE: (_income_actions,),
    TurnPhase.PERMANENT_ROUTE_PIECE_SELECTION: (_income_actions,),
    TurnPhase.PLACE_ADJACENT_ROUTE: (_route_actions,),
    TurnPhase.REPLACE_BONUS_MARKERS: (_replacement_actions, _control_actions),
    TurnPhase.TURN_COMPLETE: (_tile_and_marker_actions, _control_actions),
}


def get_legal_actions(game):
    """Return every legal structured interaction for the current phase."""
    if game.turn_phase == TurnPhase.GAME_OVER:
        return []

    actions = [
        action for builder in PHASE_BUILDERS.get(game.turn_phase, ()) for action in builder(game)
    ]
    unique = list(dict.fromkeys(actions))
    if len(unique) != len(actions):
        raise RuntimeError("Legal interaction generation produced a duplicate")
    return unique
