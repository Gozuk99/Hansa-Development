"""Structured legal interactions backed by the existing rules predicates."""

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
from map_data.constants import DARK_GREEN


BM_TYPES = (
    "SwapOffice",
    "Move3",
    "UpgradeAbility",
    "3Actions",
    "4Actions",
    "ExchangeBonusMarker",
    "Tribute4EstablishingTP",
    "BlockTradeRoute",
)


def _enabled(tensor):
    return (index for index, value in enumerate(tensor) if bool(value))


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


def _allowed_in_phase(game, action):
    phase = game.turn_phase
    if phase == TurnPhase.ACTIONS:
        return True
    if phase == TurnPhase.DISPLACEMENT:
        return isinstance(action, (PostInteraction, SupplyInteraction, ControlInteraction))
    if phase == TurnPhase.MOVE_PIECES:
        return isinstance(action, PostInteraction)
    if phase == TurnPhase.BONUS_MARKER_CHOICE:
        return isinstance(
            action,
            (
                PostInteraction,
                BonusMarkerInteraction,
                CityInteraction,
                PlayerInteraction,
                AbilityInteraction,
                ControlInteraction,
            ),
        )
    if phase in (TurnPhase.BUY_TILE_PAYMENT, TurnPhase.INCOME_FAVOUR_RESPONSE):
        return isinstance(action, TileInteraction)
    if phase in (
        TurnPhase.TRIBUTE_INCOME_RESPONSE,
        TurnPhase.PERMANENT_ROUTE_PIECE_SELECTION,
    ):
        return isinstance(action, IncomeInteraction)
    if phase == TurnPhase.PLACE_ADJACENT_ROUTE:
        return isinstance(action, RouteInteraction) and action.interaction_slot >= 3
    if phase == TurnPhase.REPLACE_BONUS_MARKERS:
        return isinstance(action, (RouteInteraction, ControlInteraction))
    if phase == TurnPhase.TURN_COMPLETE:
        return isinstance(action, (BonusMarkerInteraction, ControlInteraction))
    return False


def get_legal_actions(game):
    """Return every currently legal stable interaction exactly once."""
    from ai.action_options import (
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

    if game.turn_phase == TurnPhase.GAME_OVER:
        return []

    actions = []

    for local in _enabled(mask_post_action(game)):
        shape = PieceShape.MERCHANT if local >= 121 else PieceShape.TRADER
        actions.append(PostInteraction(local % 121, shape))

    for local in _enabled(mask_claim_route(game)):
        if local < 40:
            actions.append(RouteInteraction(local, 0))
        elif local < 120:
            route, endpoint = divmod(local - 40, 2)
            actions.append(RouteInteraction(route, endpoint + 1))
        else:
            route, outcome = divmod(local - 120, 4)
            actions.append(RouteInteraction(route, outcome + 3))

    actions.extend(IncomeInteraction(local) for local in _enabled(mask_income_actions(game)))

    for local in _enabled(mask_bm(game)):
        if game.waiting_for_bm_exchange_bm and game.exchange_target_player is not None:
            opponents = [player for player in game.players if player is not game.current_player]
            opponent = opponents.index(game.exchange_target_player)
            actions.append(BonusMarkerInteraction(9 + opponent * 8 + local))
        else:
            actions.append(BonusMarkerInteraction(local))

    actions.extend(TileInteraction(local) for local in _enabled(mask_buy_tile(game)))

    for route in _enabled(mask_replace_bm(game)):
        actions.append(RouteInteraction(route, 0))

    city_mask = mask_bm_city_actions(game)
    if game.waiting_for_bm_exchange_bm and game.exchange_target_player is None:
        actions.extend(PlayerInteraction(local) for local in _enabled(city_mask))
    elif game.waiting_for_bm_swap_office:
        eligible = [
            (city, pair)
            for city in game.selected_map.cities
            for pair in city.eligible_swap_pairs(game.current_player)
        ]
        catalogue = _city_pair_catalogue(game)
        actions.extend(
            CityInteraction(catalogue.index(choice))
            for local, choice in enumerate(eligible)
            if bool(city_mask[local])
        )
    elif game.waiting_for_bm_green_city:
        eligible = [
            (city, shape)
            for city, shape in _green_catalogue(game)
            if game.current_player.has_personal_supply(shape)
        ]
        catalogue = _green_catalogue(game)
        actions.extend(
            CityInteraction(46 + catalogue.index(choice))
            for local, choice in enumerate(eligible)
            if bool(city_mask[local])
        )

    actions.extend(AbilityInteraction(local) for local in _enabled(mask_bm_upgrade_ability(game)))

    if bool(mask_end_turn(game)[0]):
        control = (
            0
            if game.turn_phase
            in (
                TurnPhase.DISPLACEMENT,
                TurnPhase.MOVE_PIECES,
                TurnPhase.BONUS_MARKER_CHOICE,
            )
            else 1
        )
        actions.append(ControlInteraction(control))

    if bool(mask_place_adjacent(game)[0]):
        if game.turn_phase == TurnPhase.DISPLACEMENT:
            actions.append(SupplyInteraction(0))
        else:
            actions.append(BonusMarkerInteraction(8))

    actions = [action for action in actions if _allowed_in_phase(game, action)]
    unique = list(dict.fromkeys(actions))
    if len(unique) != len(actions):
        raise RuntimeError("Legal interaction generation produced a duplicate")
    return unique


def to_legacy_index(game, action):
    """Translate one legal interaction for the unchanged production dispatcher."""
    if isinstance(action, PostInteraction):
        return action.post_slot + (121 if action.shape is PieceShape.MERCHANT else 0)
    if isinstance(action, RouteInteraction):
        if game.turn_phase == TurnPhase.REPLACE_BONUS_MARKERS:
            return 543 + action.route_slot
        if action.interaction_slot == 0:
            return 242 + action.route_slot
        if action.interaction_slot <= 2:
            return 242 + 40 + action.route_slot * 2 + action.interaction_slot - 1
        return 242 + 120 + action.route_slot * 4 + action.interaction_slot - 3
    if isinstance(action, IncomeInteraction):
        return 522 + action.merchant_count
    if isinstance(action, BonusMarkerInteraction):
        if action.marker_slot == 8:
            return 619
        marker_type_slot = (
            (action.marker_slot - 9) % 8 if action.marker_slot >= 9 else action.marker_slot
        )
        return 527 + marker_type_slot
    if isinstance(action, TileInteraction):
        return 535 + action.tile_slot
    if isinstance(action, PlayerInteraction):
        return 583 + action.player_slot
    if isinstance(action, CityInteraction):
        if game.waiting_for_bm_swap_office:
            catalogue = _city_pair_catalogue(game)
            choice = catalogue[action.city_interaction_slot]
            eligible = [
                (city, pair)
                for city in game.selected_map.cities
                for pair in city.eligible_swap_pairs(game.current_player)
            ]
            return 583 + eligible.index(choice)
        catalogue = _green_catalogue(game)
        choice = catalogue[action.city_interaction_slot - 46]
        eligible = [item for item in catalogue if game.current_player.has_personal_supply(item[1])]
        return 583 + eligible.index(choice)
    if isinstance(action, AbilityInteraction):
        return 613 + action.ability_slot
    if isinstance(action, SupplyInteraction):
        return 619
    if isinstance(action, ControlInteraction):
        return 618
    raise TypeError(f"Unsupported legal interaction: {action!r}")
