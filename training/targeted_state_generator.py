"""Deterministic generators for playable, near-end-game Hansa positions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from itertools import product
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


GENERATOR_VERSION = 7
DEFAULT_OUTPUT_DIRECTORY = Path("training_data/generated")


class StateGenerationError(RuntimeError):
    """Raised when a seed cannot produce a state satisfying every constraint."""


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
    score_range: tuple[int, int] | None = None
    immediate_finish: bool = False


@dataclass(frozen=True)
class GeneratedState:
    game: object
    scenario: EndGameScenario
    seed: int
    attempt_seed: int
    score_range: tuple[int, int] | None = None
    immediate_finish: bool = False


def _choose(value, choices, rng):
    return rng.choice(choices) if value is None else value


def _configure_player(player):
    player.keys_index = 0
    player.keys = CITY_KEYS_MAX_VALUES[0]
    player.privilege = PRIVILEGE_COLORS[0]
    player.book = BOOK_OF_KNOWLEDGE_MAX_VALUES[0]
    player.actions_index = 0
    player.actions = ACTIONS_MAX_VALUES[0]
    player.bank = BANK_MAX_VALUES[0]
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
    if (
        office.controller is not None
        or pools[player][shape] <= 0
        or not _can_claim_office(player, office)
    ):
        return False
    office.controller = player
    office.owner_piece_shape = shape
    office.color = player.color
    pools[player][shape] -= 1
    return True


def _upgrade_choices(player):
    choices = []
    if player.keys_index + 1 < len(CITY_KEYS_MAX_VALUES):
        choices.append(("keys", "square"))
    if PRIVILEGE_COLORS.index(player.privilege) + 1 < len(PRIVILEGE_COLORS):
        choices.append(("privilege", "square"))
    if BOOK_OF_KNOWLEDGE_MAX_VALUES.index(player.book) + 1 < len(BOOK_OF_KNOWLEDGE_MAX_VALUES):
        choices.append(("book", "circle"))
    if player.actions_index + 1 < len(ACTIONS_MAX_VALUES):
        choices.append(("actions", "square"))
    if BANK_MAX_VALUES.index(player.bank) + 1 < len(BANK_MAX_VALUES):
        choices.append(("bank", "square"))
    return choices


def _apply_upgrade(player, pools, choice):
    ability, shape = choice
    if ability == "keys":
        player.keys_index += 1
        player.keys = CITY_KEYS_MAX_VALUES[player.keys_index]
    elif ability == "privilege":
        index = PRIVILEGE_COLORS.index(player.privilege) + 1
        player.privilege = PRIVILEGE_COLORS[index]
    elif ability == "book":
        index = BOOK_OF_KNOWLEDGE_MAX_VALUES.index(player.book) + 1
        player.book = BOOK_OF_KNOWLEDGE_MAX_VALUES[index]
    elif ability == "actions":
        player.actions_index += 1
        player.actions = ACTIONS_MAX_VALUES[player.actions_index]
    else:
        index = BANK_MAX_VALUES.index(player.bank) + 1
        player.bank = BANK_MAX_VALUES[index]
    pools[player][shape] += 1


def _prepare_player_for_office(player, office, pools, rng):
    required_privilege = office.printed_privilege or "WHITE"
    while PRIVILEGE_COLORS.index(player.privilege) < PRIVILEGE_COLORS.index(required_privilege):
        _apply_upgrade(player, pools, ("privilege", "square"))
    while pools[player][office.shape] <= 0:
        choices = [choice for choice in _upgrade_choices(player) if choice[1] == office.shape]
        if not choices:
            return False
        _apply_upgrade(player, pools, rng.choice(choices))
    return True


def _office_choices(game, player, pools):
    choices = []
    for city in game.selected_map.cities:
        if city.color == DARK_GREEN:
            continue
        open_offices = [office for office in city.offices if office.controller is None]
        if len(open_offices) <= 1:
            continue
        office = open_offices[0]
        if pools[player][office.shape] and _can_claim_office(player, office):
            choices.append(office)
    return choices


def _prepare_balanced_development(game, pools, rng):
    target = rng.randint(7, 9)
    targets = {player: target for player in game.players}
    completed = {player: 0 for player in game.players}
    while any(completed[player] < targets[player] for player in game.players):
        order = list(game.players)
        rng.shuffle(order)
        for player in order:
            if completed[player] >= targets[player]:
                continue
            offices = _office_choices(game, player, pools)
            upgrades = _upgrade_choices(player)
            categories = []
            if offices:
                categories.append("office")
            if upgrades:
                categories.append("upgrade")
            if not categories:
                return None
            if rng.choice(categories) == "office":
                if not _place_office(rng.choice(offices), player, pools):
                    return None
            else:
                _apply_upgrade(player, pools, rng.choice(upgrades))
            completed[player] += 1
    return targets


def _fill_city_for_balanced_control(city, players, pools, control_counts, office_counts, rng):
    for office in city.offices:
        if office.controller is not None:
            continue
        candidates = list(players)
        rng.shuffle(candidates)
        candidates.sort(key=lambda player: (control_counts[player], office_counts[player]))
        owner = next(
            (
                player
                for player in candidates
                if _prepare_player_for_office(player, office, pools, rng)
            ),
            None,
        )
        if owner is None or not _place_office(office, owner, pools):
            return False
        office_counts[owner] += 1
    controller = city.determine_controller()
    if controller is not None:
        control_counts[controller] += 1
    return True


def _route_shapes(route, pool, required_shape=None):
    choices = [
        (post.required_shape,) if post.required_shape else ("square", "circle")
        for post in route.posts
    ]
    for shapes in product(*choices):
        if required_shape is not None and required_shape not in shapes:
            continue
        if all(shapes.count(shape) <= pool[shape] for shape in ("square", "circle")):
            return shapes
    return None


def _prepare_completed_cities(game, pools, rng):
    target = game.selected_map.max_full_cities - 1
    trigger_cities = [
        city
        for city in game.selected_map.cities
        if city.color != DARK_GREEN and city.offices and not city.city_is_full()
    ]
    rng.shuffle(trigger_cities)
    trigger_city = trigger_cities[0]
    candidates = [city for city in trigger_cities[1:] if city is not trigger_city]
    selected = candidates[:target]
    if len(selected) != target:
        return None
    control_counts = {
        player: sum(city.determine_controller() is player for city in game.selected_map.cities)
        for player in game.players
    }
    office_counts = {
        player: sum(
            office.controller is player
            for city in game.selected_map.cities
            for office in city.offices
        )
        for player in game.players
    }
    for city in selected:
        if not _fill_city_for_balanced_control(
            city, game.players, pools, control_counts, office_counts, rng
        ):
            return None
    open_offices = [office for office in trigger_city.offices if office.controller is None]
    for office in open_offices[:-1]:
        candidates = list(game.players)
        rng.shuffle(candidates)
        candidates.sort(key=lambda player: (control_counts[player], office_counts[player]))
        owner = next(
            (
                player
                for player in candidates
                if _prepare_player_for_office(player, office, pools, rng)
            ),
            None,
        )
        if owner is None or not _place_office(office, owner, pools):
            return None
        office_counts[owner] += 1

    office = open_offices[-1]
    routes = list(trigger_city.routes)
    rng.shuffle(routes)
    players = list(game.players)
    rng.shuffle(players)
    for player in players:
        if not _can_claim_office(player, office):
            continue
        for route in routes:
            shapes = _route_shapes(route, pools[player], office.shape)
            if shapes is None:
                continue
            for post, shape in zip(route.posts, shapes):
                post.owner = player
                post.owner_piece_shape = shape
                pools[player][shape] -= 1
            return player
    return None


def _prepare_bonus_marker_route(game, pools, rng):
    routes = [route for route in game.selected_map.routes if route.bonus_marker is not None]
    rng.shuffle(routes)
    players = list(game.players)
    rng.shuffle(players)
    for player in players:
        for route in routes:
            shapes = _route_shapes(route, pools[player])
            if shapes is None:
                continue
            for post, shape in zip(route.posts, shapes):
                post.owner = player
                post.owner_piece_shape = shape
                pools[player][shape] -= 1
            return player
    return None


def _prepare_score_route(game, pools, rng):
    routes = [
        route
        for route in game.selected_map.routes
        if route.bonus_marker is None
        and route.permanent_bonus_marker is None
        and all(city.color != DARK_GREEN and city.offices for city in route.cities)
        and all(all(office.controller is None for office in city.offices) for city in route.cities)
    ]
    rng.shuffle(routes)
    players = list(game.players)
    rng.shuffle(players)
    projected_scores = dict(zip(game.players, game.projected_scores()))
    players.sort(key=projected_scores.get)
    for player in players:
        for route in routes:
            endpoint_offices = [city.offices[0] for city in route.cities]
            if not all(_can_claim_office(player, office) for office in endpoint_offices):
                continue
            required = {
                shape: sum(office.shape == shape for office in endpoint_offices)
                for shape in ("square", "circle")
            }
            remaining = {
                shape: pools[player][shape] - required[shape] for shape in ("square", "circle")
            }
            if min(remaining.values()) < 0:
                continue
            shapes = _route_shapes(route, remaining)
            if shapes is None:
                continue
            for office in endpoint_offices:
                _place_office(office, player, pools)
            for post, shape in zip(route.posts, shapes):
                post.owner = player
                post.owner_piece_shape = shape
                pools[player][shape] -= 1

            for other in game.players:
                if other is player or any(
                    office.controller is other
                    for city in game.selected_map.cities
                    for office in city.offices
                ):
                    continue
                offices = [
                    city.offices[0]
                    for city in game.selected_map.cities
                    if city not in route.cities
                    and city.offices
                    and city.offices[0].controller is None
                    and pools[other][city.offices[0].shape]
                    and _can_claim_office(other, city.offices[0])
                ]
                rng.shuffle(offices)
                if not offices or not _place_office(offices[0], other, pools):
                    return None
            return player
    return None


def _divide_remaining_supply(players, pools, rng):
    for player in players:
        for shape in ("square", "circle"):
            count = pools[player][shape]
            personal = rng.randint(0, count)
            setattr(player, f"personal_supply_{shape}s", personal)
            setattr(player, f"general_stock_{shape}s", count - personal)


def _give_marker_to_player(game, marker, rng):
    owner = rng.choice(game.players)
    marker.owner = owner
    destination = rng.choice((owner.bonus_markers, owner.used_bonus_markers))
    destination.append(marker)


def _configure_bonus_marker_scenario(game, rng, prepared_player, immediate_finish):
    while game.selected_map.bonus_marker_pool:
        marker = BonusMarker(game.selected_map.bonus_marker_pool.pop())
        _give_marker_to_player(game, marker, rng)
    if not immediate_finish:
        for route in game.selected_map.routes:
            if route.bonus_marker is not None and not route.is_controlled_by(prepared_player):
                marker = route.bonus_marker
                route.bonus_marker = None
                _give_marker_to_player(game, marker, rng)


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
    pools = {player: _configure_player(player) for player in game.players}
    development_targets = _prepare_balanced_development(game, pools, rng)
    if development_targets is None:
        return None

    prepared_current_player = None
    if scenario is EndGameScenario.NEAR_SCORE:
        prepared_current_player = _prepare_score_route(game, pools, rng)
        if prepared_current_player is None:
            return None
    elif scenario is EndGameScenario.NEAR_COMPLETED_CITIES:
        prepared_current_player = _prepare_completed_cities(game, pools, rng)
        if prepared_current_player is None:
            return None
    elif scenario is EndGameScenario.NEAR_BONUS_MARKERS:
        prepared_current_player = _prepare_bonus_marker_route(game, pools, rng)
        if prepared_current_player is None:
            return None
    _divide_remaining_supply(game.players, pools, rng)

    applied_score_range = request.score_range
    if scenario is EndGameScenario.NEAR_SCORE:
        applied_score_range = (17, 18)
        for player in game.players:
            player.score = rng.choice(applied_score_range)
        prepared_current_player.score = 18
    elif request.score_range is not None:
        minimum_score, maximum_score = request.score_range
        if minimum_score < 0 or maximum_score < minimum_score or maximum_score > 19:
            raise ValueError("score range must be between 0 and 19")
        base_score = rng.randint(minimum_score, maximum_score)
        highest_score = min(base_score + 1, maximum_score)
        for player in game.players:
            player.score = rng.randint(base_score, highest_score)
    else:
        base_score = rng.randint(6, 18)
        for player in game.players:
            player.score = rng.randint(base_score, base_score + 1)
    if scenario is EndGameScenario.NEAR_BONUS_MARKERS:
        _configure_bonus_marker_scenario(
            game, rng, prepared_current_player, request.immediate_finish
        )

    _assign_some_emperor_tiles(game, rng)
    game.current_full_cities_count = sum(city.city_is_full() for city in game.selected_map.cities)
    if prepared_current_player is None:
        game.current_player_index = rng.randrange(player_count)
    else:
        prepared_index = game.players.index(prepared_current_player)
        game.current_player_index = (
            prepared_index if request.immediate_finish else (prepared_index + 1) % player_count
        )
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

    projected_scores = game.projected_scores()
    if max(projected_scores) - min(projected_scores) > 3:
        return None
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
    return GeneratedState(
        game,
        scenario,
        request.seed,
        attempt_seed,
        applied_score_range,
        request.immediate_finish,
    )


def generate_state(request: GenerationRequest, *, max_attempts: int = 2_000) -> GeneratedState:
    """Create one deterministic, validated state matching ``request``."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(max_attempts):
        generated = _build_once(request, request.seed + attempt * 1_000_003)
        if generated is not None:
            return generated
    raise StateGenerationError(f"Could not generate a valid state after {max_attempts} attempts")


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
        "score_range": generated.score_range,
        "immediate_finish": generated.immediate_finish,
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
        "score_range": generated.score_range,
        "starting_position": (
            "immediate_finish" if generated.immediate_finish else "one_round_before"
        ),
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
