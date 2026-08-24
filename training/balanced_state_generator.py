"""Balanced generator with independent ending and strategic focuses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import random

from game.action_validation import ActionValidationError, validate_action_state
from game.game_config import GameConfiguration, human_players
from game.invariants import GameInvariantError, validate_game
from game.loaded_state_validation import validate_loaded_game
from game.persistence import load_game, save_game
from map_data.map_attributes import Map
from training.targeted_state_generator import (
    DEVELOPMENT_RANGES,
    EndGameScenario,
    StateGenerationError,
    _assign_some_emperor_tiles,
    _balance_projected_scores,
    _apply_upgrade,
    _can_claim_office,
    _complete_balanced_development,
    _configure_bonus_marker_scenario,
    _configure_player,
    _development_total,
    _divide_remaining_supply,
    _ensure_personal_piece,
    _prepare_bonus_marker_route,
    _prepare_britannia_region,
    _prepare_completed_cities,
    _prepare_east_west,
    _prepare_dual_east_west,
    _prepare_score_route,
    _prepare_special_prestige,
    _prepare_network_keys,
    _place_office,
    _upgrade_choices,
)


GENERATOR_VERSION = 8


class EndingCondition(str, Enum):
    NEAR_SCORE = "near_score"
    NEAR_BONUS_MARKERS = "near_bonus_markers"
    NEAR_COMPLETED_CITIES = "near_completed_cities"


class RegionalFocus(str, Enum):
    WALES = "britannia_wales"
    SCOTLAND = "britannia_scotland"
    ISLE_OF_MAN = "britannia_isle_of_man"


class StrategicFocus(str, Enum):
    NONE = "none"
    EAST_WEST = "east_west"
    DUAL_EAST_WEST = "dual_east_west"
    BLOCKED_EAST_WEST = "blocked_east_west"
    BLOCKED_DUAL_EAST_WEST = "blocked_dual_east_west"
    SPECIAL_PRESTIGE = "special_prestige"
    NETWORK_KEYS = "network_keys"


class StartingPosition(str, Enum):
    ONE_ROUND_BEFORE = "one_round_before"
    TWO_DECISIONS_BEFORE = "two_decisions_before"
    IMMEDIATE_FINISH = "immediate_finish"


class BonusMarkerSetup(str, Enum):
    DEFAULT = "default"
    ALL_PROMOS = "all_promos"
    MIXED = "mixed"


EAST_WEST_FOCUSES = {
    StrategicFocus.EAST_WEST,
    StrategicFocus.DUAL_EAST_WEST,
    StrategicFocus.BLOCKED_EAST_WEST,
    StrategicFocus.BLOCKED_DUAL_EAST_WEST,
}
DUAL_EAST_WEST_FOCUSES = {
    StrategicFocus.DUAL_EAST_WEST,
    StrategicFocus.BLOCKED_DUAL_EAST_WEST,
}
BLOCKED_EAST_WEST_FOCUSES = {
    StrategicFocus.BLOCKED_EAST_WEST,
    StrategicFocus.BLOCKED_DUAL_EAST_WEST,
}


REGIONAL_SCENARIOS = {
    RegionalFocus.WALES: EndGameScenario.BRITANNIA_WALES,
    RegionalFocus.SCOTLAND: EndGameScenario.BRITANNIA_SCOTLAND,
    RegionalFocus.ISLE_OF_MAN: EndGameScenario.BRITANNIA_ISLE_OF_MAN,
}


def _prepare_remaining_completed_cities(game, pools, rng, cities_below_limit):
    """Prepare the remaining city limit after strategic focuses filled some cities."""
    already_full = sum(city.city_is_full() for city in game.selected_map.cities)
    original_limit = game.selected_map.max_full_cities
    remaining_limit = original_limit - already_full
    adjusted_limit = remaining_limit - cities_below_limit + 1
    if adjusted_limit <= 1:
        return None
    game.selected_map.max_full_cities = adjusted_limit
    try:
        return _prepare_completed_cities(game, pools, rng)
    finally:
        game.selected_map.max_full_cities = original_limit


@dataclass(frozen=True)
class BalancedGenerationRequest:
    seed: int
    map_num: int
    player_count: int
    ending_condition: EndingCondition
    score_range: tuple[int, int] = (16, 17)
    strategic_focus: StrategicFocus = StrategicFocus.NONE
    regional_focus: RegionalFocus | None = None
    use_mission_cards: bool = False
    use_emperors_favour: bool = False
    use_promo_markers: bool = False
    bonus_marker_setup: BonusMarkerSetup | None = None
    starting_position: StartingPosition = StartingPosition.ONE_ROUND_BEFORE
    bonus_markers_remaining: int = 0
    completed_cities_below_limit: int = 1
    prepared_routes_one_short: bool = False
    development_range: tuple[int, int] | None = None
    prepare_ending_condition: bool = True
    round_range: tuple[int, int] = (8, 20)


@dataclass(frozen=True)
class BalancedGeneratedState:
    game: object
    request: BalancedGenerationRequest
    attempt_seed: int
    development_range: tuple[int, int]
    prepared_player_index: int
    focus_variants: tuple[str, ...]
    starting_scores_by_seat: tuple[int, ...]
    starting_development_by_seat: tuple[int, ...]


@dataclass
class _PreparedFocus:
    required_pieces: list
    variants: list[str]
    east_west_players: list
    blocked_east_west_player: object | None = None


def _validate_request(request):
    if request.map_num not in (1, 2, 3) or request.player_count not in (3, 4, 5):
        raise ValueError("map_num and player_count must each be 1-3 and 3-5 respectively")
    if request.use_mission_cards and request.map_num != 1:
        raise ValueError("Mission cards are available only on Map 1")
    if request.regional_focus is not None and request.map_num != 3:
        raise ValueError("Britannia regional focus requires Map 3")
    if request.player_count == 3 and request.regional_focus in (
        RegionalFocus.SCOTLAND,
        RegionalFocus.ISLE_OF_MAN,
    ):
        raise ValueError("Three-player Britannia focus supports Wales only")
    if (
        request.strategic_focus
        in (
            StrategicFocus.SPECIAL_PRESTIGE,
            StrategicFocus.NETWORK_KEYS,
        )
        and request.regional_focus is not None
    ):
        raise ValueError("special_prestige cannot be combined with another strategic focus")
    minimum_score, maximum_score = request.score_range
    if minimum_score < 0 or maximum_score < minimum_score or maximum_score > 19:
        raise ValueError("score_range must be between 0 and 19")
    if request.bonus_markers_remaining < 0:
        raise ValueError("bonus_markers_remaining cannot be negative")
    if request.completed_cities_below_limit < 1:
        raise ValueError("completed_cities_below_limit must be positive")
    if request.development_range is not None:
        minimum, maximum = request.development_range
        if minimum < 0 or maximum < minimum or maximum > 20:
            raise ValueError("development_range must be between 0 and 20")
    minimum_round, maximum_round = request.round_range
    if minimum_round < 1 or maximum_round < minimum_round:
        raise ValueError("round_range must contain positive increasing rounds")


def _resolved_bonus_marker_setup(request):
    if request.bonus_marker_setup is not None:
        return BonusMarkerSetup(request.bonus_marker_setup)
    return BonusMarkerSetup.MIXED if request.use_promo_markers else BonusMarkerSetup.DEFAULT


def bonus_marker_configuration(setup, seed):
    setup = BonusMarkerSetup(setup)
    if setup is BonusMarkerSetup.DEFAULT:
        return False, "random", ()
    promos = [marker for marker, count in Map.PROMO_BONUS_MARKERS.items() for _ in range(count)]
    standards = [
        marker for marker, count in Map.STANDARD_BONUS_MARKER_SUPPLY.items() for _ in range(count)
    ]
    marker_rng = random.Random(seed + 7_919)
    if setup is BonusMarkerSetup.MIXED:
        promo_count = marker_rng.randint(1, len(promos) - 1)
        supply = tuple(
            marker_rng.sample(promos, promo_count) + marker_rng.sample(standards, 12 - promo_count)
        )
        return True, "manual", supply

    marker_rng.shuffle(standards)
    supply = tuple(promos + standards[: 12 - len(promos)])
    return True, "manual", supply


def _bonus_marker_configuration(request, attempt_seed):
    return bonus_marker_configuration(_resolved_bonus_marker_setup(request), attempt_seed)


def _prepare_east_west_focus(game, pools, rng, request, focus):
    path_length = None
    if request.regional_focus is RegionalFocus.ISLE_OF_MAN:
        path_length = (
            "medium"
            if request.ending_condition is EndingCondition.NEAR_COMPLETED_CITIES
            else "short"
        )
    if request.strategic_focus in DUAL_EAST_WEST_FOCUSES:
        prepared_players = _prepare_dual_east_west(game, pools, rng)
        if prepared_players is None:
            return False
        for index, (player, shape) in enumerate(prepared_players):
            focus.east_west_players.append(player)
            focus.required_pieces.append((player, shape))
            focus.variants.append("east_west" if index == 0 else "east_west_rival")
        return True

    prepared = _prepare_east_west(
        game,
        pools,
        rng,
        path_length,
        False
        if request.strategic_focus in BLOCKED_EAST_WEST_FOCUSES
        else not request.prepared_routes_one_short,
    )
    if prepared is None:
        return False
    player, shape, variant = prepared
    focus.east_west_players.append(player)
    focus.required_pieces.append((player, shape))
    focus.variants.append(f"east_west:{variant}")
    return True


def _block_east_west_route(game, pools, rng, focus):
    player = rng.choice(focus.east_west_players)
    routes = [
        route
        for route in game.selected_map.routes
        if sum(post.owner is None for post in route.posts) == 1
        and all(post.owner in (None, player) for post in route.posts)
    ]
    if not routes:
        return False
    route = rng.choice(routes)
    post = next(post for post in route.posts if post.owner is None)
    shapes = [post.required_shape] if post.required_shape else ["square", "circle"]
    blockers = [
        (candidate, shape)
        for candidate in game.players
        if candidate is not player
        for shape in shapes
        if pools[candidate][shape] > 0
    ]
    if not blockers:
        return False
    blocker, shape = rng.choice(blockers)
    post.owner = blocker
    post.owner_piece_shape = shape
    pools[blocker][shape] -= 1
    focus.blocked_east_west_player = player
    focus.variants.append("east_west_blocked")
    return True


def _prepare_strategic_focus(game, pools, rng, request, development_range):
    focus = _PreparedFocus([], [], [])
    if request.strategic_focus is StrategicFocus.NETWORK_KEYS:
        if _complete_balanced_development(game, pools, rng, development_range) is None:
            return None
        prepared = _prepare_network_keys(
            game,
            pools,
            rng,
            leave_one_open=request.prepared_routes_one_short,
        )
        if prepared is None:
            return None
        player, shape, variant = prepared
        focus.required_pieces.append((player, shape))
        focus.variants.append(variant)

    if request.strategic_focus in EAST_WEST_FOCUSES:
        if not _prepare_east_west_focus(game, pools, rng, request, focus):
            return None
        if request.strategic_focus in BLOCKED_EAST_WEST_FOCUSES and not _block_east_west_route(
            game, pools, rng, focus
        ):
            return None

    if request.regional_focus is not None:
        prepared = _prepare_britannia_region(
            game,
            pools,
            rng,
            REGIONAL_SCENARIOS[request.regional_focus],
            not request.prepared_routes_one_short,
        )
        if prepared is None:
            return None
        player, shape, variant = prepared
        focus.required_pieces.append((player, shape))
        focus.variants.append(f"{request.regional_focus.value}:{variant}")

    if request.strategic_focus is StrategicFocus.SPECIAL_PRESTIGE:
        prepared = _prepare_special_prestige(
            game, pools, rng, leave_one_open=request.prepared_routes_one_short
        )
        if prepared is None:
            return None
        player, shape, variant = prepared
        focus.required_pieces.append((player, shape))
        focus.variants.append(variant)
    return focus


def _prepare_ending_condition(game, pools, rng, request):
    if request.ending_condition is EndingCondition.NEAR_SCORE:
        return _prepare_score_route(game, pools, rng)
    if request.ending_condition is EndingCondition.NEAR_BONUS_MARKERS:
        return _prepare_bonus_marker_route(game, pools, rng)
    return _prepare_remaining_completed_cities(
        game, pools, rng, request.completed_cities_below_limit
    )


def _open_ending_route(game, pools, rng, request, player):
    if not request.prepared_routes_one_short:
        return None
    routes = [
        route
        for route in game.selected_map.routes
        if route.is_controlled_by(player)
        and (
            request.ending_condition is not EndingCondition.NEAR_BONUS_MARKERS
            or route.bonus_marker is not None
        )
    ]
    if not routes:
        return False
    route = rng.choice(routes)
    post = rng.choice([post for post in route.posts if post.owner is player])
    shape = post.owner_piece_shape
    post.owner = None
    post.owner_piece_shape = None
    pools[player][shape] += 1
    return player, shape


def _ensure_displacement_supply(player):
    while player.personal_supply_squares + player.personal_supply_circles < 3:
        shape = next(
            (
                candidate
                for candidate in ("square", "circle")
                if getattr(player, f"general_stock_{candidate}s") > 0
            ),
            None,
        )
        if shape is None:
            return False
        stock = f"general_stock_{shape}s"
        supply = f"personal_supply_{shape}s"
        setattr(player, stock, getattr(player, stock) - 1)
        setattr(player, supply, getattr(player, supply) + 1)
    return True


def _finish_supply_setup(game, pools, rng, focus):
    _divide_remaining_supply(game.players, pools, rng)
    try:
        for player, shape in focus.required_pieces:
            _ensure_personal_piece(player, shape)
    except StateGenerationError:
        return False
    return focus.blocked_east_west_player is None or _ensure_displacement_supply(
        focus.blocked_east_west_player
    )


def _set_balanced_scores(game, target):
    contributions = [
        total - player.score for total, player in zip(game.projected_scores(), game.players)
    ]
    scores = [target - contribution for contribution in contributions]
    if any(score < 0 or score > 19 for score in scores):
        return False
    for player, score in zip(game.players, scores):
        player.score = score
    return True


def _set_balanced_near_scores(game, request, prepared_player):
    """Balance projected scores while keeping ordinary scores in their requested range."""
    contributions = [
        total - player.score for total, player in zip(game.projected_scores(), game.players)
    ]
    minimum_score, maximum_score = request.score_range
    prepared_index = game.players.index(prepared_player)
    candidates = []
    minimum_target = min(contributions) + minimum_score
    maximum_target = max(contributions) + maximum_score
    for maximum_score_index in range(len(game.players)):
        for target in range(minimum_target, maximum_target + 1):
            scores = [
                max(minimum_score, min(maximum_score, target - contribution))
                for contribution in contributions
            ]
            scores[maximum_score_index] = maximum_score
            projected_scores = [
                contribution + score for contribution, score in zip(contributions, scores)
            ]
            spread = max(projected_scores) - min(projected_scores)
            distance = sum(abs(projected - target) for projected in projected_scores)
            candidates.append((spread, distance, -scores[prepared_index], tuple(scores)))
    spread, _distance, _prepared_score, scores = min(candidates)
    if spread > 3:
        return False
    for player, score in zip(game.players, scores):
        player.score = score
    return True


def _apply_starting_scores(game, rng, request, prepared_player, focus):
    if request.ending_condition is EndingCondition.NEAR_SCORE:
        if request.strategic_focus is not StrategicFocus.NONE or request.regional_focus:
            contributions = [
                total - player.score for total, player in zip(game.projected_scores(), game.players)
            ]
            target = contributions[game.players.index(prepared_player)] + request.score_range[1]
            if not _set_balanced_scores(game, target):
                return False
        else:
            if not _set_balanced_near_scores(game, request, prepared_player):
                return False
    elif request.strategic_focus is not StrategicFocus.NONE or request.regional_focus:
        if _balance_projected_scores(game, rng) is None:
            return False
    else:
        minimum_score, maximum_score = request.score_range
        base_score = rng.randint(minimum_score, maximum_score)
        for player in game.players:
            player.score = rng.randint(base_score, min(base_score + 1, maximum_score))

    if request.strategic_focus not in (DUAL_EAST_WEST_FOCUSES | BLOCKED_EAST_WEST_FOCUSES):
        return True
    contributions = [
        total - player.score for total, player in zip(game.projected_scores(), game.players)
    ]
    protected_players = (
        focus.east_west_players
        if request.strategic_focus in DUAL_EAST_WEST_FOCUSES
        else (focus.blocked_east_west_player,)
    )
    maximum_target = min(
        contributions[game.players.index(player)] + 8 for player in protected_players
    )
    target = min(max(contributions), maximum_target)
    return target >= max(contributions) and _set_balanced_scores(game, target)


def _configure_turn(game, rng, request, prepared_player):
    prepared_index = game.players.index(prepared_player)
    game.current_player_index = (
        prepared_index
        if request.starting_position is not StartingPosition.ONE_ROUND_BEFORE
        else (prepared_index + 1) % request.player_count
    )
    game.current_player = game.players[game.current_player_index]
    game.active_player = game.current_player_index
    game.round_number = rng.randint(*request.round_range)
    game.turn_number = (
        (game.round_number - 1) * request.player_count + game.current_player_index + 1
    )
    for player in game.players:
        player.actions_at_turn_start = player.actions
        player.actions_remaining = 0
        player.actions_granted_this_turn = 0
        player.ending_turn = False
    game.current_player.start_turn(extra_actions=int(game.OneActionOwner is game.current_player))
    if request.map_num == 3:
        game.current_player.refresh_map3_priv_actions(game)
    return prepared_index


def player_has_starting_score_source(game, player):
    """Return whether a player's existing points have an authoritative board source."""
    owns_city = any(city.determine_controller() is player for city in game.selected_map.cities)
    prestige = game.selected_map.specialprestigepoints
    spent_bonus_victory_circle = bool(
        prestige is not None and any(circle["owner"] is player for circle in prestige.circle_data)
    )
    return owns_city or spent_bonus_victory_circle


def starting_scores_have_valid_sources(game):
    return all(
        player.score == 0 or player_has_starting_score_source(game, player)
        for player in game.players
    )


def _ensure_player_controls_city(game, player, pools, rng):
    if player_has_starting_score_source(game, player):
        return True
    cities = list(game.selected_map.cities)
    rng.shuffle(cities)
    for city in cities:
        if len(city.offices) < 2 or any(office.controller is not None for office in city.offices):
            continue
        offices = list(city.offices)
        rng.shuffle(offices)
        for office in offices:
            if pools[player][office.shape] and _can_claim_office(player, office):
                return _place_office(office, player, pools)
    return False


def _is_valid_generated_state(game, request):
    if max(game.projected_scores()) - min(game.projected_scores()) > 3:
        return False
    if not starting_scores_have_valid_sources(game):
        return False
    if (
        request.ending_condition is EndingCondition.NEAR_BONUS_MARKERS
        or not request.prepare_ending_condition
    ) and (
        sum(route.bonus_marker is not None for route in game.selected_map.routes) != 3
        or game.replace_bonus_marker != 0
        or game.pending_bonus_markers
    ):
        return False
    try:
        validate_game(game)
        validate_loaded_game(game)
    except GameInvariantError:
        return False
    if game.game_end or not game.get_legal_actions():
        return False
    if (
        request.prepare_ending_condition
        and request.ending_condition is EndingCondition.NEAR_COMPLETED_CITIES
        and game.current_full_cities_count
        != game.selected_map.max_full_cities - request.completed_cities_below_limit
    ):
        return False
    try:
        validate_action_state(game, quiet=True)
    except ActionValidationError:
        return False
    return True


def _build_once(request, attempt_seed):
    rng = random.Random(attempt_seed)
    use_promos, promo_mode, promo_markers = _bonus_marker_configuration(request, attempt_seed)
    configuration = GameConfiguration(
        map_num=request.map_num,
        player_count=request.player_count,
        player_controls=human_players(request.player_count),
        use_mission_cards=request.use_mission_cards,
        use_emperors_favour=request.use_emperors_favour,
        use_promo_markers=use_promos,
        promo_marker_mode=promo_mode,
        promo_markers=promo_markers,
        seed=attempt_seed,
    )
    game = configuration.create_game()
    pools = {player: _configure_player(player) for player in game.players}
    development_range = request.development_range or rng.choice(DEVELOPMENT_RANGES)
    complicated_focus = (
        request.strategic_focus in DUAL_EAST_WEST_FOCUSES
        or request.strategic_focus in (StrategicFocus.SPECIAL_PRESTIGE, StrategicFocus.NETWORK_KEYS)
        or request.regional_focus is not None
    )
    if request.development_range is not None:
        initial_range = (0, min(1, development_range[1]))
    else:
        initial_range = (0, 1) if complicated_focus else (3, 5)
    if _complete_balanced_development(game, pools, rng, initial_range, allow_offices=True) is None:
        return None
    if request.strategic_focus is not StrategicFocus.NETWORK_KEYS and any(
        not _ensure_player_controls_city(game, player, pools, rng) for player in game.players
    ):
        return None

    focus = _prepare_strategic_focus(game, pools, rng, request, development_range)
    if focus is None:
        return None
    if request.strategic_focus is StrategicFocus.NETWORK_KEYS and any(
        not _ensure_player_controls_city(game, player, pools, rng) for player in game.players
    ):
        return None

    prepared_player = rng.choice(game.players)
    if request.prepare_ending_condition:
        prepared_player = _prepare_ending_condition(game, pools, rng, request)
        if prepared_player is None:
            return None

        opened_piece = _open_ending_route(game, pools, rng, request, prepared_player)
        if opened_piece is False:
            return None
        if opened_piece is not None:
            focus.required_pieces.append(opened_piece)

    if request.strategic_focus is not StrategicFocus.NETWORK_KEYS:
        if _complete_balanced_development(game, pools, rng, development_range) is None:
            return None
    if not _finish_supply_setup(game, pools, rng, focus):
        return None

    if request.prepare_ending_condition:
        if not _apply_starting_scores(game, rng, request, prepared_player, focus):
            return None
    else:
        base_score = rng.randint(*request.score_range)
        for player in game.players:
            player.score = rng.randint(base_score, min(base_score + 1, request.score_range[1]))

    if (
        request.ending_condition is EndingCondition.NEAR_BONUS_MARKERS
        or not request.prepare_ending_condition
    ):
        _configure_bonus_marker_scenario(
            game,
            rng,
            request.bonus_markers_remaining,
        )
    _assign_some_emperor_tiles(game, rng)
    game.current_full_cities_count = sum(city.city_is_full() for city in game.selected_map.cities)
    if not request.prepare_ending_condition and game.current_full_cities_count:
        return None
    prepared_index = _configure_turn(game, rng, request, prepared_player)
    if not _is_valid_generated_state(game, request):
        return None
    return BalancedGeneratedState(
        game,
        request,
        attempt_seed,
        development_range,
        prepared_index,
        tuple(focus.variants),
        tuple(player.score for player in game.players),
        tuple(_development_total(game, player) for player in game.players),
    )


def generate_balanced_state(request, *, max_attempts=2_000):
    _validate_request(request)
    if (
        request.strategic_focus in EAST_WEST_FOCUSES
        and request.regional_focus is RegionalFocus.ISLE_OF_MAN
    ):
        max_attempts = max(max_attempts, 10_000)
    for attempt in range(max_attempts):
        generated = _build_once(request, request.seed + attempt * 1_000_003)
        if generated is not None:
            return generated
    raise StateGenerationError(
        f"map {request.map_num}, {request.player_count} players, "
        f"{request.ending_condition.value}, {request.strategic_focus.value}, "
        f"regional={request.regional_focus.value if request.regional_focus else 'none'}"
    )


def save_balanced_state(generated, output_directory, *, scenario_directory=None):
    request = generated.request
    identity = {
        "generator_version": GENERATOR_VERSION,
        "seed": request.seed,
        "attempt_seed": generated.attempt_seed,
        "map_num": request.map_num,
        "player_count": request.player_count,
        "ending_condition": request.ending_condition.value,
        "score_range": request.score_range,
        "strategic_focus": request.strategic_focus.value,
        "bonus_markers_remaining": request.bonus_markers_remaining,
        "completed_cities_below_limit": request.completed_cities_below_limit,
        "starting_position": request.starting_position.value,
        "prepared_routes_one_short": request.prepared_routes_one_short,
        "development_range": request.development_range,
        "prepare_ending_condition": request.prepare_ending_condition,
        "round_range": request.round_range,
        "bonus_marker_setup": _resolved_bonus_marker_setup(request),
        "regional_focus": request.regional_focus,
        "options": (
            request.use_mission_cards,
            request.use_emperors_favour,
            _resolved_bonus_marker_setup(request),
        ),
    }
    state_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    directory = (
        Path(output_directory)
        / (scenario_directory or request.ending_condition.value)
        / f"map_{request.map_num}"
        / f"{request.player_count}_players"
    )
    save_path = save_game(generated.game, directory / f"state-{state_id}.hansa")
    validate_loaded_game(load_game(save_path))
    metadata = {
        **identity,
        "state_id": f"state-{state_id}",
        "strategic_focuses": generated.focus_variants,
        "development_range": generated.development_range,
        "prepared_player_index": generated.prepared_player_index,
        "starting_score_by_seat": generated.starting_scores_by_seat,
        "starting_development_by_seat": generated.starting_development_by_seat,
        "starting_position": request.starting_position.value,
        "save_file": save_path.name,
    }
    metadata_path = directory / f"state-{state_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return save_path, metadata_path
