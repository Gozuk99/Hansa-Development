"""Deterministic generators for playable, near-end-game Hansa positions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import random

from game.action_validation import ActionValidationError, validate_action_state
from game.game_config import GameConfiguration, human_players
from game.invariants import validate_game
from game.loaded_state_validation import validate_loaded_game
from game.persistence import load_game, save_game
from map_data.constants import (
    ACTIONS_MAX_VALUES,
    BANK_MAX_VALUES,
    BOOK_OF_KNOWLEDGE_MAX_VALUES,
    CITY_KEYS_MAX_VALUES,
    DARK_GREEN,
    PRIVILEGE_COLORS,
)
from map_data.map_attributes import BonusMarker


GENERATOR_VERSION = 1
DEFAULT_OUTPUT_DIRECTORY = Path("training_data/generated")


class EndGameScenario(str, Enum):
    NEAR_SCORE = "near_score"
    NEAR_BONUS_MARKERS = "near_bonus_markers"
    NEAR_COMPLETED_CITIES = "near_completed_cities"


@dataclass(frozen=True)
class GenerationRequest:
    seed: int
    scenario: EndGameScenario | None = None
    map_num: int | None = None
    player_count: int | None = None
    use_mission_cards: bool | None = None
    use_emperors_favour: bool | None = None
    use_promo_markers: bool | None = None


@dataclass(frozen=True)
class GeneratedState:
    game: object
    scenario: EndGameScenario
    seed: int
    attempt_seed: int


def _choose(value, choices, rng):
    return rng.choice(choices) if value is None else value


def _configure_player(player, rng):
    player.keys_index = rng.randrange(len(CITY_KEYS_MAX_VALUES))
    player.keys = CITY_KEYS_MAX_VALUES[player.keys_index]
    player.privilege = rng.choice(PRIVILEGE_COLORS)
    player.book = rng.choice(BOOK_OF_KNOWLEDGE_MAX_VALUES)
    player.actions_index = rng.randrange(len(ACTIONS_MAX_VALUES))
    player.actions = ACTIONS_MAX_VALUES[player.actions_index]
    player.bank = rng.choice(BANK_MAX_VALUES)

    player.general_stock_squares = 0
    player.personal_supply_squares = 0
    player.general_stock_circles = 0
    player.personal_supply_circles = 0
    return {
        "square": 26 - player.locked_ability_traders,
        "circle": 4 - player.locked_ability_merchants,
    }


def _can_claim_office(player, office):
    printed = office.printed_privilege or "WHITE"
    return PRIVILEGE_COLORS.index(player.privilege) >= PRIVILEGE_COLORS.index(printed)


def _place_office(office, player, pools):
    shape = office.shape
    if pools[player][shape] <= 0 or not _can_claim_office(player, office):
        return False
    office.controller = player
    office.owner_piece_shape = shape
    office.color = player.color
    pools[player][shape] -= 1
    return True


def _fill_city(city, players, pools, rng):
    for office in city.offices:
        candidates = [
            player
            for player in players
            if pools[player][office.shape] and _can_claim_office(player, office)
        ]
        if not candidates:
            return False
        _place_office(office, rng.choice(candidates), pools)
    return True


def _prepare_completed_cities(game, pools, rng):
    target = game.selected_map.max_full_cities - 1
    candidates = [
        city for city in game.selected_map.cities if city.color != DARK_GREEN and city.offices
    ]
    rng.shuffle(candidates)
    selected = candidates[:target]
    if len(selected) != target:
        return False
    return all(_fill_city(city, game.players, pools, rng) for city in selected)


def _board_candidates(game, player, pools):
    candidates = []
    for route in game.selected_map.routes:
        for post in route.posts:
            for shape in ("square", "circle"):
                if pools[player][shape] and post.can_be_claimed_by(shape):
                    candidates.append(("post", post, shape))

    for city in game.selected_map.cities:
        if city.color == DARK_GREEN:
            continue
        open_offices = [office for office in city.offices if office.controller is None]
        # Never accidentally complete another city while constructing random noise.
        if len(open_offices) > 1:
            office = open_offices[0]
            if pools[player][office.shape] and _can_claim_office(player, office):
                candidates.append(("office", office, office.shape))

    prestige = game.selected_map.specialprestigepoints
    if prestige is not None and pools[player]["circle"]:
        for circle in prestige.circle_data:
            if circle["owner"] is None and prestige.can_claim_prestige(player, circle["value"]):
                candidates.append(("prestige", circle, "circle"))
    return candidates


def _place_random_board_pieces(game, pools, rng):
    remaining = sum(sum(shapes.values()) for shapes in pools.values())
    placements = rng.randint(remaining // 3, max(remaining // 3, remaining * 2 // 3))
    players = list(game.players)
    for step in range(placements):
        player = players[step % len(players)]
        candidates = _board_candidates(game, player, pools)
        if not candidates:
            alternatives = [
                (other, candidate)
                for other in players
                for candidate in _board_candidates(game, other, pools)
            ]
            if not alternatives:
                break
            player, selected = rng.choice(alternatives)
            candidates = [selected]
        kind, target, shape = rng.choice(candidates)
        if kind == "post":
            target.claim(player, shape)
        elif kind == "office":
            _place_office(target, player, pools)
            continue
        else:
            target["owner"] = player
            target["color"] = player.color
        pools[player][shape] -= 1


def _divide_remaining_supply(players, pools, rng):
    for player in players:
        for shape in ("square", "circle"):
            count = pools[player][shape]
            personal = rng.randint(0, count)
            setattr(player, f"personal_supply_{shape}s", personal)
            setattr(player, f"general_stock_{shape}s", count - personal)


def _configure_bonus_marker_scenario(game, rng):
    keep = rng.choice((1, 2))
    while len(game.selected_map.bonus_marker_pool) > keep:
        marker = BonusMarker(game.selected_map.bonus_marker_pool.pop())
        owner = rng.choice(game.players)
        marker.owner = owner
        destination = rng.choice((owner.bonus_markers, owner.used_bonus_markers))
        destination.append(marker)


def _assign_some_emperor_tiles(game, rng):
    if not game.use_emperors_favour:
        return
    owner_fields = {
        "DisplaceAnywhere": "DisplaceAnywhereOwner",
        "+1Action": "OneActionOwner",
        "+1IncomeIfOthersIncome": "OneIncomeIfOthersIncomeOwner",
        "+1DisplacedPiece": "OneDisplacedPieceOwner",
        "+4PtsPerOwnedCity": "FourPtsPerOwnedCityOwner",
        "+7PtsPerCompletedAbility": "SevenPtsPerCompletedAbilityOwner",
    }
    rng.shuffle(game.tile_pool)
    for _ in range(rng.randint(0, len(game.tile_pool))):
        tile = game.tile_pool.pop()
        player = rng.choice(game.players)
        player.tiles.append(tile)
        setattr(game, owner_fields[tile], player)


def _build_once(request, attempt_seed):
    rng = random.Random(attempt_seed)
    scenario = _choose(request.scenario, tuple(EndGameScenario), rng)
    map_num = _choose(request.map_num, (1, 2, 3), rng)
    player_count = _choose(request.player_count, (3, 4, 5), rng)
    use_missions = False if map_num != 1 else _choose(request.use_mission_cards, (False, True), rng)
    use_favour = _choose(request.use_emperors_favour, (False, True), rng)
    use_promos = _choose(request.use_promo_markers, (False, True), rng)
    config = GameConfiguration(
        map_num=map_num,
        player_count=player_count,
        player_controls=human_players(player_count),
        use_mission_cards=use_missions,
        use_emperors_favour=use_favour,
        use_promo_markers=use_promos,
        seed=attempt_seed,
    )
    game = config.create_game()
    pools = {player: _configure_player(player, rng) for player in game.players}

    if scenario is EndGameScenario.NEAR_COMPLETED_CITIES:
        if not _prepare_completed_cities(game, pools, rng):
            return None
    _place_random_board_pieces(game, pools, rng)
    _divide_remaining_supply(game.players, pools, rng)

    if scenario is EndGameScenario.NEAR_SCORE:
        for player in game.players:
            player.score = rng.randint(17, 19)
    else:
        for player in game.players:
            player.score = rng.randint(6, 19)
    if scenario is EndGameScenario.NEAR_BONUS_MARKERS:
        _configure_bonus_marker_scenario(game, rng)

    _assign_some_emperor_tiles(game, rng)
    game.current_full_cities_count = sum(city.city_is_full() for city in game.selected_map.cities)
    game.current_player_index = rng.randrange(player_count)
    game.current_player = game.players[game.current_player_index]
    game.active_player = game.current_player_index
    game.round_number = rng.randint(8, 20)
    game.turn_number = (game.round_number - 1) * player_count + game.current_player_index + 1
    for player in game.players:
        player.actions_at_turn_start = player.actions
        player.actions_remaining = 0
        player.actions_granted_this_turn = 0
        player.ending_turn = False
    game.current_player.start_turn(extra_actions=int(game.OneActionOwner is game.current_player))
    if map_num == 3:
        game.current_player.refresh_map3_priv_actions(game)

    validate_game(game)
    validate_loaded_game(game)
    if game.game_end or not game.get_legal_actions():
        return None
    if scenario is EndGameScenario.NEAR_COMPLETED_CITIES:
        if game.current_full_cities_count != game.selected_map.max_full_cities - 1:
            return None
    try:
        validate_action_state(game, quiet=True)
    except ActionValidationError:
        return None
    return GeneratedState(game, scenario, request.seed, attempt_seed)


def generate_state(request: GenerationRequest, *, max_attempts: int = 100) -> GeneratedState:
    """Create one deterministic, validated state matching ``request``."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(max_attempts):
        generated = _build_once(request, request.seed + attempt * 1_000_003)
        if generated is not None:
            return generated
    raise RuntimeError(f"Could not generate a valid state after {max_attempts} attempts")


def _state_id(generated):
    identity = {
        "generator_version": GENERATOR_VERSION,
        "seed": generated.seed,
        "attempt_seed": generated.attempt_seed,
        "scenario": generated.scenario.value,
        "map_num": generated.game.map_num,
        "player_count": len(generated.game.players),
        "mission_cards": generated.game.use_mission_cards,
        "emperors_favour": generated.game.use_emperors_favour,
        "promo_bonus_markers": generated.game.configuration.use_promo_markers,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return f"state-{digest}"


def save_generated_state(generated: GeneratedState, output_directory=DEFAULT_OUTPUT_DIRECTORY):
    """Save a generated exact game and a small searchable metadata sidecar."""
    game = generated.game
    state_id = _state_id(generated)
    directory = (
        Path(output_directory)
        / generated.scenario.value
        / f"map_{game.map_num}"
        / f"{len(game.players)}_players"
    )
    save_path = save_game(game, directory / f"{state_id}.hansa")
    loaded = load_game(save_path)
    validate_loaded_game(loaded)
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "state_id": state_id,
        "scenario": generated.scenario.value,
        "seed": generated.seed,
        "attempt_seed": generated.attempt_seed,
        "map_num": game.map_num,
        "player_count": len(game.players),
        "scores": [player.score for player in game.players],
        "current_player_index": game.current_player_index,
        "bonus_markers_remaining": len(game.selected_map.bonus_marker_pool),
        "completed_cities": game.current_full_cities_count,
        "options": {
            "mission_cards": game.use_mission_cards,
            "emperors_favour": game.use_emperors_favour,
            "promo_bonus_markers": game.configuration.use_promo_markers,
        },
        "save_file": save_path.name,
    }
    metadata_path = directory / f"{state_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return save_path, metadata_path
