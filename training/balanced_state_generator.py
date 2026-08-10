"""Experimental balanced generator with independent ending and strategic focuses."""

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
from training.targeted_state_generator import (
    DEVELOPMENT_RANGES,
    EndGameScenario,
    StateGenerationError,
    _assign_some_emperor_tiles,
    _balance_projected_scores,
    _complete_balanced_development,
    _configure_bonus_marker_scenario,
    _configure_player,
    _divide_remaining_supply,
    _ensure_personal_piece,
    _prepare_bonus_marker_route,
    _prepare_britannia_region,
    _prepare_completed_cities,
    _prepare_east_west,
    _prepare_score_route,
)


GENERATOR_VERSION = 1


class EndingCondition(str, Enum):
    NEAR_SCORE = "near_score"
    NEAR_BONUS_MARKERS = "near_bonus_markers"
    NEAR_COMPLETED_CITIES = "near_completed_cities"


class RegionalFocus(str, Enum):
    WALES = "britannia_wales"
    SCOTLAND = "britannia_scotland"
    ISLE_OF_MAN = "britannia_isle_of_man"


REGIONAL_SCENARIOS = {
    RegionalFocus.WALES: EndGameScenario.BRITANNIA_WALES,
    RegionalFocus.SCOTLAND: EndGameScenario.BRITANNIA_SCOTLAND,
    RegionalFocus.ISLE_OF_MAN: EndGameScenario.BRITANNIA_ISLE_OF_MAN,
}


def _prepare_remaining_completed_cities(game, pools, rng):
    """Prepare the remaining city limit after strategic focuses filled some cities."""
    already_full = sum(city.city_is_full() for city in game.selected_map.cities)
    original_limit = game.selected_map.max_full_cities
    remaining_limit = original_limit - already_full
    if remaining_limit <= 1:
        return None
    game.selected_map.max_full_cities = remaining_limit
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
    east_west: bool = False
    regional_focus: RegionalFocus | None = None
    use_mission_cards: bool = False
    use_emperors_favour: bool = False
    use_promo_markers: bool = False
    immediate_finish: bool = False


@dataclass(frozen=True)
class BalancedGeneratedState:
    game: object
    request: BalancedGenerationRequest
    attempt_seed: int
    development_range: tuple[int, int]
    prepared_player_index: int
    focus_variants: tuple[str, ...]


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


def _build_once(request, attempt_seed):
    rng = random.Random(attempt_seed)
    configuration = GameConfiguration(
        map_num=request.map_num,
        player_count=request.player_count,
        player_controls=human_players(request.player_count),
        use_mission_cards=request.use_mission_cards,
        use_emperors_favour=request.use_emperors_favour,
        use_promo_markers=request.use_promo_markers,
        seed=attempt_seed,
    )
    game = configuration.create_game()
    pools = {player: _configure_player(player) for player in game.players}
    development_range = rng.choice(DEVELOPMENT_RANGES)
    initial_range = (0, 1) if request.east_west and request.regional_focus else (3, 5)
    if _complete_balanced_development(game, pools, rng, initial_range, allow_offices=True) is None:
        return None

    required_pieces = []
    focus_variants = []
    if request.east_west:
        path_length = None
        if request.regional_focus is RegionalFocus.ISLE_OF_MAN:
            path_length = (
                "medium"
                if request.ending_condition is EndingCondition.NEAR_COMPLETED_CITIES
                else "short"
            )
        prepared = _prepare_east_west(game, pools, rng, path_length, None)
        if prepared is None:
            return None
        focus_player, shape, variant = prepared
        required_pieces.append((focus_player, shape))
        focus_variants.append(f"east_west:{variant}")
    if request.regional_focus is not None:
        prepared = _prepare_britannia_region(
            game, pools, rng, REGIONAL_SCENARIOS[request.regional_focus], None
        )
        if prepared is None:
            return None
        focus_player, shape, variant = prepared
        required_pieces.append((focus_player, shape))
        focus_variants.append(f"{request.regional_focus.value}:{variant}")

    if request.ending_condition is EndingCondition.NEAR_SCORE:
        prepared_player = _prepare_score_route(game, pools, rng)
    elif request.ending_condition is EndingCondition.NEAR_BONUS_MARKERS:
        prepared_player = _prepare_bonus_marker_route(game, pools, rng)
    else:
        prepared_player = _prepare_remaining_completed_cities(game, pools, rng)
    if prepared_player is None:
        return None

    if _complete_balanced_development(game, pools, rng, development_range) is None:
        return None
    _divide_remaining_supply(game.players, pools, rng)
    for player, shape in required_pieces:
        _ensure_personal_piece(player, shape)

    if request.ending_condition is EndingCondition.NEAR_SCORE:
        if request.east_west or request.regional_focus is not None:
            projected = game.projected_scores()
            contributions = [total - player.score for total, player in zip(projected, game.players)]
            target = contributions[game.players.index(prepared_player)] + 18
            required_scores = [target - contribution for contribution in contributions]
            if any(score < 0 or score > 19 for score in required_scores):
                return None
            for player, score in zip(game.players, required_scores):
                player.score = score
        else:
            for player in game.players:
                player.score = rng.choice((17, 18))
            prepared_player.score = 18
    elif request.east_west or request.regional_focus is not None:
        if _balance_projected_scores(game, rng) is None:
            return None
    else:
        base_score = rng.randint(6, 16)
        for player in game.players:
            player.score = rng.randint(base_score, min(base_score + 1, 16))

    if request.ending_condition is EndingCondition.NEAR_BONUS_MARKERS:
        _configure_bonus_marker_scenario(game, rng, prepared_player, request.immediate_finish)
    _assign_some_emperor_tiles(game, rng)
    game.current_full_cities_count = sum(city.city_is_full() for city in game.selected_map.cities)
    prepared_index = game.players.index(prepared_player)
    game.current_player_index = (
        prepared_index if request.immediate_finish else (prepared_index + 1) % request.player_count
    )
    game.current_player = game.players[game.current_player_index]
    game.active_player = game.current_player_index
    game.round_number = rng.randint(8, 20)
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

    if max(game.projected_scores()) - min(game.projected_scores()) > 3:
        return None
    try:
        validate_game(game)
        validate_loaded_game(game)
    except GameInvariantError:
        return None
    if game.game_end or not game.get_legal_actions():
        return None
    if (
        request.ending_condition is EndingCondition.NEAR_COMPLETED_CITIES
        and game.current_full_cities_count != game.selected_map.max_full_cities - 1
    ):
        return None
    try:
        validate_action_state(game, quiet=True)
    except ActionValidationError:
        return None
    return BalancedGeneratedState(
        game,
        request,
        attempt_seed,
        development_range,
        prepared_index,
        tuple(focus_variants),
    )


def generate_balanced_state(request, *, max_attempts=2_000):
    _validate_request(request)
    if request.east_west and request.regional_focus is RegionalFocus.ISLE_OF_MAN:
        max_attempts = max(max_attempts, 10_000)
    for attempt in range(max_attempts):
        generated = _build_once(request, request.seed + attempt * 1_000_003)
        if generated is not None:
            return generated
    raise StateGenerationError(f"Could not generate a balanced state after {max_attempts} attempts")


def save_balanced_state(generated, output_directory):
    request = generated.request
    identity = {
        "generator_version": GENERATOR_VERSION,
        "seed": request.seed,
        "attempt_seed": generated.attempt_seed,
        "map_num": request.map_num,
        "player_count": request.player_count,
        "ending_condition": request.ending_condition.value,
        "east_west": request.east_west,
        "regional_focus": request.regional_focus,
        "options": (
            request.use_mission_cards,
            request.use_emperors_favour,
            request.use_promo_markers,
        ),
    }
    state_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    directory = (
        Path(output_directory)
        / request.ending_condition.value
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
        "starting_position": "immediate_finish" if request.immediate_finish else "one_round_before",
        "save_file": save_path.name,
    }
    metadata_path = directory / f"state-{state_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return save_path, metadata_path
