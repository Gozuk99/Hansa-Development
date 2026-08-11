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
    COLOR_NAMES,
    DARK_GREEN,
    PRIVILEGE_COLORS,
)
from map_data.map_attributes import BonusMarker


GENERATOR_VERSION = 9
DEFAULT_OUTPUT_DIRECTORY = Path("training_data/generated")
DEVELOPMENT_RANGES = ((7, 9), (9, 11), (11, 13))


class StateGenerationError(RuntimeError):
    """Raised when a seed cannot produce a state satisfying every constraint."""


class EndGameScenario(str, Enum):
    NEAR_SCORE = "near_score"
    NEAR_BONUS_MARKERS = "near_bonus_markers"
    NEAR_COMPLETED_CITIES = "near_completed_cities"
    EAST_WEST = "east_west"
    BRITANNIA_WALES = "britannia_wales"
    BRITANNIA_SCOTLAND = "britannia_scotland"
    BRITANNIA_ISLE_OF_MAN = "britannia_isle_of_man"


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
    east_west_path_length: str | None = None
    prepared_route_full: bool | None = None


@dataclass(frozen=True)
class GeneratedState:
    game: object
    scenario: EndGameScenario
    seed: int
    attempt_seed: int
    score_range: tuple[int, int] | None = None
    immediate_finish: bool = False
    target_variant: str | None = None
    prepared_player_index: int | None = None
    prepared_route_full: bool | None = None
    development_range: tuple[int, int] | None = None


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


def _development_total(game, player):
    upgrades = (
        player.keys_index
        + PRIVILEGE_COLORS.index(player.privilege)
        + BOOK_OF_KNOWLEDGE_MAX_VALUES.index(player.book)
        + player.actions_index
        + BANK_MAX_VALUES.index(player.bank)
    )
    offices = sum(
        office.controller is player for city in game.selected_map.cities for office in city.offices
    )
    return upgrades + offices


def _complete_balanced_development(game, pools, rng, development_range, *, allow_offices=False):
    minimum, maximum = development_range
    completed = {player: _development_total(game, player) for player in game.players}
    if any(total > maximum for total in completed.values()):
        return None
    targets = {
        player: rng.randint(max(minimum, completed[player]), maximum) for player in game.players
    }
    while any(completed[player] < targets[player] for player in game.players):
        order = list(game.players)
        rng.shuffle(order)
        for player in order:
            if completed[player] >= targets[player]:
                continue
            offices = _office_choices(game, player, pools) if allow_offices else []
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


def _route_shapes(route, pool, required_shape=None, rng=None):
    choices = [
        (post.required_shape,) if post.required_shape else ("square", "circle")
        for post in route.posts
    ]
    legal_shapes = []
    for shapes in product(*choices):
        if required_shape is not None and required_shape not in shapes:
            continue
        if all(shapes.count(shape) <= pool[shape] for shape in ("square", "circle")):
            legal_shapes.append(shapes)
    if not legal_shapes:
        return None
    return rng.choice(legal_shapes) if rng is not None else legal_shapes[0]


def _fill_prepared_route(route, player, pool, required_shape, leave_one_open, rng):
    if any(post.is_owned() for post in route.posts):
        return False, None
    shapes = _route_shapes(route, pool, required_shape, rng)
    if shapes is None:
        return False, None
    open_index = rng.randrange(len(route.posts)) if leave_one_open else None
    for index, (post, shape) in enumerate(zip(route.posts, shapes)):
        if index == open_index:
            continue
        post.owner = player
        post.owner_piece_shape = shape
        pool[shape] -= 1
    return True, None if open_index is None else shapes[open_index]


def _can_prepare_route(route):
    return bool(
        route
        and len(route.posts) >= 3
        and route.bonus_marker is None
        and route.permanent_bonus_marker is None
        and not any(post.is_owned() for post in route.posts)
    )


def _next_open_office(city):
    return next((office for office in city.offices if office.controller is None), None)


def _place_city_office(city, player, pools, rng):
    if city.has_office_owned_by(player):
        return True
    office = _next_open_office(city)
    return bool(
        office
        and _prepare_player_for_office(player, office, pools, rng)
        and _place_office(office, player, pools)
    )


def _bounded_east_west_paths(game):
    cities = {city.name: city for city in game.selected_map.cities}
    start_name, end_name = game.selected_map.east_west_cities
    start, end = cities[start_name], cities[end_name]
    adjacency = {city: [] for city in game.selected_map.cities}
    for route in game.selected_map.routes:
        first, second = route.cities
        adjacency[first].append(second)
        adjacency[second].append(first)

    frontier = [(start, 0)]
    visited = {start}
    shortest = None
    while frontier:
        city, distance = frontier.pop(0)
        if city is end:
            shortest = distance
            break
        for adjacent in adjacency[city]:
            if adjacent not in visited:
                visited.add(adjacent)
                frontier.append((adjacent, distance + 1))
    if shortest is None:
        return {}

    paths = []

    def visit(city, path):
        if len(path) - 1 > shortest + 3:
            return
        if city is end:
            paths.append(tuple(path))
            return
        for adjacent in adjacency[city]:
            if adjacent not in path:
                visit(adjacent, (*path, adjacent))

    visit(start, (start,))
    lengths = sorted({len(path) - 1 for path in paths})
    groups = {"short": {lengths[0]}, "long": {lengths[-1]}}
    middle = set(lengths[1:-1])
    if middle:
        groups["medium"] = middle
    return {
        name: [path for path in paths if len(path) - 1 in selected_lengths]
        for name, selected_lengths in groups.items()
    }


def _route_between(game, first, second):
    return next(
        (route for route in game.selected_map.routes if set(route.cities) == {first, second}),
        None,
    )


def _prepare_east_west(game, pools, rng, requested_length, prepared_route_full):
    grouped_paths = _bounded_east_west_paths(game)
    if requested_length is not None and requested_length not in grouped_paths:
        return None
    categories = [requested_length] if requested_length else list(grouped_paths)
    rng.shuffle(categories)
    players = list(game.players)
    rng.shuffle(players)
    for category in categories:
        paths = list(grouped_paths[category])
        rng.shuffle(paths)
        for path in paths:
            gap_indices = list(range(len(path)))
            rng.shuffle(gap_indices)
            for player in players:
                for gap_index in gap_indices:
                    gap_city = path[gap_index]
                    if gap_city.color == DARK_GREEN or gap_city.has_office_owned_by(player):
                        continue
                    target_office = _next_open_office(gap_city)
                    if target_office is None:
                        continue
                    neighbor_index = 1 if gap_index == 0 else gap_index - 1
                    route = _route_between(game, gap_city, path[neighbor_index])
                    if not _can_prepare_route(route):
                        continue
                    if any(
                        not city.has_office_owned_by(player) and _next_open_office(city) is None
                        for index, city in enumerate(path)
                        if index != gap_index
                    ):
                        continue
                    for index, city in enumerate(path):
                        if index != gap_index and not _place_city_office(city, player, pools, rng):
                            return None
                    if not _prepare_player_for_office(player, target_office, pools, rng):
                        return None
                    start_name, end_name = game.selected_map.east_west_cities
                    if game.has_east_west_connection(start_name, end_name):
                        return None
                    leave_one_open = (
                        rng.choice((False, True))
                        if prepared_route_full is None
                        else not prepared_route_full
                    )
                    route_prepared, missing_shape = _fill_prepared_route(
                        route,
                        player,
                        pools[player],
                        target_office.shape,
                        leave_one_open,
                        rng,
                    )
                    if not route_prepared:
                        return None

                    original = (
                        target_office.controller,
                        target_office.owner_piece_shape,
                        target_office.color,
                    )
                    target_office.controller = player
                    target_office.owner_piece_shape = target_office.shape
                    target_office.color = player.color
                    completes_connection = game.has_east_west_connection(start_name, end_name)
                    (
                        target_office.controller,
                        target_office.owner_piece_shape,
                        target_office.color,
                    ) = original
                    if not completes_connection:
                        return None
                    return player, missing_shape, category
    return None


def _prepare_dual_east_west(game, pools, rng):
    paths = [path for group in _bounded_east_west_paths(game).values() for path in group]
    pairs = [
        (first, second)
        for first in paths
        for second in paths
        if first is not second
        and all(len(city.offices) >= 2 for city in set(first).intersection(second))
    ]
    rng.shuffle(pairs)
    players = list(game.players)
    rng.shuffle(players)
    for first_path, second_path in pairs:
        first_player, second_player = players[:2]
        first_gaps = list(range(len(first_path)))
        second_gaps = list(range(len(second_path)))
        rng.shuffle(first_gaps)
        rng.shuffle(second_gaps)
        for first_gap in first_gaps:
            for second_gap in second_gaps:
                gap_specs = []
                for path, gap_index in ((first_path, first_gap), (second_path, second_gap)):
                    gap_city = path[gap_index]
                    neighbor = path[1 if gap_index == 0 else gap_index - 1]
                    route = _route_between(game, gap_city, neighbor)
                    if gap_city.color == DARK_GREEN or not _can_prepare_route(route):
                        break
                    gap_specs.append((gap_city, route))
                if len(gap_specs) != 2 or gap_specs[0][1] is gap_specs[1][1]:
                    continue
                required_offices = {}
                for player, path, gap_index in (
                    (first_player, first_path, first_gap),
                    (second_player, second_path, second_gap),
                ):
                    for index, city in enumerate(path):
                        if index != gap_index:
                            required_offices.setdefault(city, []).append(player)
                if any(
                    len([office for office in city.offices if office.controller is None])
                    < len(owners)
                    for city, owners in required_offices.items()
                ):
                    continue
                for city, owners in required_offices.items():
                    for player in owners:
                        if not _place_city_office(city, player, pools, rng):
                            return None
                prepared = []
                for player, (gap_city, route) in zip((first_player, second_player), gap_specs):
                    target_office = _next_open_office(gap_city)
                    if target_office is None or not _prepare_player_for_office(
                        player, target_office, pools, rng
                    ):
                        return None
                    route_prepared, missing_shape = _fill_prepared_route(
                        route, player, pools[player], target_office.shape, False, rng
                    )
                    if not route_prepared:
                        return None
                    prepared.append((player, missing_shape))
                return tuple(prepared)
    return None


def _prepare_special_prestige(game, pools, rng, leave_one_open=False):
    prestige = game.selected_map.specialprestigepoints
    special_city = next(
        city
        for city in game.selected_map.cities
        if "SpecialPrestigePoints" in city.upgrade_city_type
    )
    players = list(game.players)
    rng.shuffle(players)
    for player in players:
        while pools[player]["circle"] <= 0 and player.book != BOOK_OF_KNOWLEDGE_MAX_VALUES[-1]:
            _apply_upgrade(player, pools, ("book", "circle"))
        if pools[player]["circle"] <= 0:
            continue
        target_privilege = rng.randrange(len(PRIVILEGE_COLORS))
        while PRIVILEGE_COLORS.index(player.privilege) < target_privilege:
            _apply_upgrade(player, pools, ("privilege", "square"))

        available_values = [
            circle["value"]
            for circle in prestige.circle_data
            if PRIVILEGE_COLORS.index(COLOR_NAMES[circle["color"]]) <= target_privilege
        ]
        claimed_values = rng.sample(available_values, rng.randrange(len(available_values)))
        other_players = [other for other in game.players if other is not player]
        for value in claimed_values:
            owner = rng.choice(other_players)
            circle = next(item for item in prestige.circle_data if item["value"] == value)
            required_privilege = COLOR_NAMES[circle["color"]]
            while PRIVILEGE_COLORS.index(owner.privilege) < PRIVILEGE_COLORS.index(
                required_privilege
            ):
                _apply_upgrade(owner, pools, ("privilege", "square"))
            while pools[owner]["circle"] <= 0 and owner.book != BOOK_OF_KNOWLEDGE_MAX_VALUES[-1]:
                _apply_upgrade(owner, pools, ("book", "circle"))
            if pools[owner]["circle"] <= 0:
                return None
            circle["owner"] = owner
            circle["color"] = owner.color
            pools[owner]["circle"] -= 1

        routes = list(special_city.routes)
        rng.shuffle(routes)
        for route in routes:
            if len(route.posts) < 3 or any(post.is_owned() for post in route.posts):
                continue
            if not prestige.can_claim_prestige(player):
                continue
            prepared, missing_shape = _fill_prepared_route(
                route, player, pools[player], "circle", leave_one_open, rng
            )
            if prepared:
                return player, missing_shape, "special_prestige"
    return None


def _prepare_network_keys(game, pools, rng):
    players = list(game.players)
    rng.shuffle(players)
    cities = [
        city for city in game.selected_map.cities if city.color != DARK_GREEN and city.offices
    ]
    rng.shuffle(cities)
    for player in players:
        if any(
            office.controller is player
            for city in game.selected_map.cities
            for office in city.offices
        ):
            continue
        target_keys = rng.choice(tuple(value for value in (2, 3, 4) if value >= player.keys))
        while player.keys < target_keys:
            _apply_upgrade(player, pools, ("keys", "square"))
        target_offices = rng.randint(3, 7)
        network_cities = set()
        offices_placed = 0
        while offices_placed < target_offices:
            candidates = [
                city
                for city in cities
                if _next_open_office(city) is not None
                and (
                    not network_cities
                    or city in network_cities
                    or any(
                        adjacent in network_cities
                        for route in city.routes
                        for adjacent in route.cities
                        if adjacent is not city
                    )
                )
            ]
            rng.shuffle(candidates)
            placed = False
            for city in candidates:
                office = _next_open_office(city)
                if office is None or not _prepare_player_for_office(player, office, pools, rng):
                    continue
                if _place_office(office, player, pools):
                    network_cities.add(city)
                    offices_placed += 1
                    placed = True
                    break
            if not placed:
                return None

        extension_routes = [
            (route, city)
            for route in game.selected_map.routes
            if _can_prepare_route(route)
            for city in route.cities
            if city.color != DARK_GREEN
            and _next_open_office(city) is not None
            and any(endpoint in network_cities for endpoint in route.cities)
        ]
        rng.shuffle(extension_routes)
        for route, city in extension_routes:
            office = _next_open_office(city)
            if office is None or not _prepare_player_for_office(player, office, pools, rng):
                continue
            prepared, missing_shape = _fill_prepared_route(
                route, player, pools[player], office.shape, True, rng
            )
            if prepared:
                return player, missing_shape, f"network_{target_offices}_keys_{target_keys}"
    return None


def _region_cities(game, region):
    names = {
        city.name
        for route in game.selected_map.routes
        if route.region == region
        for city in route.cities
    } | {"IsleOfMan"}
    return [city for city in game.selected_map.cities if city.name in names]


def _fill_city_for_player(city, player, pools, rng):
    for office in city.offices:
        if office.controller is not None:
            return False
        if not _prepare_player_for_office(player, office, pools, rng):
            return False
        if not _place_office(office, player, pools):
            return False
    return True


def _prepare_britannia_region(game, pools, rng, scenario, prepared_route_full):
    if game.map_num != 3:
        return None
    dual_region = scenario is EndGameScenario.BRITANNIA_ISLE_OF_MAN
    regions = (
        ("Wales", "Scotland")
        if dual_region
        else ("Scotland" if scenario is EndGameScenario.BRITANNIA_SCOTLAND else "Wales",)
    )
    if "Scotland" in regions and game.num_players == 3:
        return None

    target_candidates = (
        [city for city in game.selected_map.cities if city.name == "IsleOfMan"]
        if dual_region
        else [
            city
            for city in _region_cities(game, regions[0])
            if city.name != "IsleOfMan" and len(city.offices) >= 2
        ]
    )
    rng.shuffle(target_candidates)
    player_pairs = [
        (actor, rival) for actor in game.players for rival in game.players if actor is not rival
    ]
    rng.shuffle(player_pairs)
    for target_city in target_candidates:
        if any(office.controller is not None for office in target_city.offices):
            continue
        routes = [route for route in target_city.routes if _can_prepare_route(route)]
        rng.shuffle(routes)
        for actor, rival in player_pairs:
            support = {}
            used = {target_city}
            possible = True
            for region in regions:
                candidates = [
                    city
                    for city in _region_cities(game, region)
                    if city not in used
                    and city.name != "IsleOfMan"
                    and city.offices
                    and all(office.controller is None for office in city.offices)
                ]
                rng.shuffle(candidates)
                if len(candidates) < 2:
                    possible = False
                    break
                support[(region, actor)], support[(region, rival)] = candidates[:2]
                used.update(candidates[:2])
            if not possible:
                continue
            for route in routes:
                target_office = target_city.offices[1]
                if not _prepare_player_for_office(actor, target_office, pools, rng):
                    return None
                if not _prepare_player_for_office(rival, target_city.offices[0], pools, rng):
                    return None
                for region in regions:
                    if not _fill_city_for_player(support[(region, actor)], actor, pools, rng):
                        return None
                    if not _fill_city_for_player(support[(region, rival)], rival, pools, rng):
                        return None
                if not _place_office(target_city.offices[0], rival, pools):
                    return None

                before = game.calculate_britannia_region_points()
                original_color = target_office.color
                target_office.controller = actor
                target_office.owner_piece_shape = target_office.shape
                target_office.color = actor.color
                after = game.calculate_britannia_region_points()
                target_office.controller = None
                target_office.owner_piece_shape = None
                target_office.color = original_color
                if after[actor] <= before[actor]:
                    return None

                leave_one_open = (
                    rng.choice((False, True))
                    if prepared_route_full is None
                    else not prepared_route_full
                )
                route_prepared, missing_shape = _fill_prepared_route(
                    route,
                    actor,
                    pools[actor],
                    target_office.shape,
                    leave_one_open,
                    rng,
                )
                if not route_prepared:
                    return None
                return actor, missing_shape, "+".join(regions)
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
    routes = [route for route in trigger_city.routes if len(route.posts) >= 3]
    rng.shuffle(routes)
    players = list(game.players)
    rng.shuffle(players)
    for player in players:
        if not _can_claim_office(player, office):
            continue
        for route in routes:
            shapes = _route_shapes(route, pools[player], office.shape, rng)
            if shapes is None:
                continue
            for post, shape in zip(route.posts, shapes):
                post.owner = player
                post.owner_piece_shape = shape
                pools[player][shape] -= 1
            return player
    return None


def _prepare_bonus_marker_route(game, pools, rng):
    routes = [
        route
        for route in game.selected_map.routes
        if route.bonus_marker is not None and len(route.posts) >= 3
    ]
    rng.shuffle(routes)
    players = list(game.players)
    rng.shuffle(players)
    for player in players:
        for route in routes:
            shapes = _route_shapes(route, pools[player], rng=rng)
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
        if len(route.posts) >= 3
        and route.bonus_marker is None
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
            shapes = _route_shapes(route, remaining, rng=rng)
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


def _ensure_personal_piece(player, shape):
    if shape is None or getattr(player, f"personal_supply_{shape}s") > 0:
        return
    stock_field = f"general_stock_{shape}s"
    if getattr(player, stock_field) <= 0:
        raise StateGenerationError(f"Prepared {shape} is unavailable after supply division")
    setattr(player, stock_field, getattr(player, stock_field) - 1)
    supply_field = f"personal_supply_{shape}s"
    setattr(player, supply_field, getattr(player, supply_field) + 1)


def _give_marker_to_player(game, marker, rng):
    owner = rng.choice(game.players)
    marker.owner = owner
    destination = rng.choice((owner.bonus_markers, owner.used_bonus_markers))
    destination.append(marker)


def _configure_bonus_marker_scenario(
    game, rng, prepared_player, immediate_finish, markers_remaining=0
):
    while len(game.selected_map.bonus_marker_pool) > markers_remaining:
        marker = BonusMarker(game.selected_map.bonus_marker_pool.pop())
        _give_marker_to_player(game, marker, rng)
    if not immediate_finish:
        for route in game.selected_map.routes:
            prepared_route = any(post.owner is prepared_player for post in route.posts) and all(
                post.owner in (None, prepared_player) for post in route.posts
            )
            if (
                route.bonus_marker is not None
                and not route.is_controlled_by(prepared_player)
                and not prepared_route
            ):
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


def _balance_projected_scores(game, rng, minimum_score=0, maximum_score=19):
    contributions = game.projected_scores()
    lowest_target = max(value + minimum_score for value in contributions)
    highest_target = min(value + maximum_score for value in contributions)
    if lowest_target > highest_target:
        return None
    target = rng.randint(lowest_target, highest_target)
    for player, contribution in zip(game.players, contributions):
        player.score = target - contribution
    return (
        min(player.score for player in game.players),
        max(player.score for player in game.players),
    )


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
    development_range = rng.choice(DEVELOPMENT_RANGES)
    if _complete_balanced_development(game, pools, rng, (3, 5), allow_offices=True) is None:
        return None

    prepared_current_player = None
    required_personal_shape = None
    target_variant = None
    targeted_scoring = scenario in (
        EndGameScenario.EAST_WEST,
        EndGameScenario.BRITANNIA_WALES,
        EndGameScenario.BRITANNIA_SCOTLAND,
        EndGameScenario.BRITANNIA_ISLE_OF_MAN,
    )
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
    elif scenario is EndGameScenario.EAST_WEST:
        prepared = _prepare_east_west(
            game,
            pools,
            rng,
            request.east_west_path_length,
            request.prepared_route_full,
        )
        if prepared is None:
            return None
        prepared_current_player, required_personal_shape, target_variant = prepared
    elif scenario in (
        EndGameScenario.BRITANNIA_WALES,
        EndGameScenario.BRITANNIA_SCOTLAND,
        EndGameScenario.BRITANNIA_ISLE_OF_MAN,
    ):
        prepared = _prepare_britannia_region(
            game, pools, rng, scenario, request.prepared_route_full
        )
        if prepared is None:
            return None
        prepared_current_player, required_personal_shape, target_variant = prepared
    if _complete_balanced_development(game, pools, rng, development_range) is None:
        return None
    _divide_remaining_supply(game.players, pools, rng)
    _ensure_personal_piece(prepared_current_player, required_personal_shape)

    applied_score_range = request.score_range
    if scenario is EndGameScenario.NEAR_SCORE:
        applied_score_range = (17, 18)
        for player in game.players:
            player.score = rng.choice(applied_score_range)
        prepared_current_player.score = 18
    elif targeted_scoring:
        applied_score_range = _balance_projected_scores(game, rng)
        if applied_score_range is None:
            return None
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
        target_variant,
        (None if prepared_current_player is None else game.players.index(prepared_current_player)),
        (
            None
            if scenario
            not in (
                EndGameScenario.EAST_WEST,
                EndGameScenario.BRITANNIA_WALES,
                EndGameScenario.BRITANNIA_SCOTLAND,
                EndGameScenario.BRITANNIA_ISLE_OF_MAN,
            )
            else required_personal_shape is None
        ),
        development_range,
    )


def generate_state(request: GenerationRequest, *, max_attempts: int = 2_000) -> GeneratedState:
    """Create one deterministic, validated state matching ``request``."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if request.east_west_path_length not in (None, "short", "medium", "long"):
        raise ValueError("east_west_path_length must be short, medium, or long")
    if request.east_west_path_length and request.scenario is not EndGameScenario.EAST_WEST:
        raise ValueError("east_west_path_length requires the east_west scenario")
    britannia = {
        EndGameScenario.BRITANNIA_WALES,
        EndGameScenario.BRITANNIA_SCOTLAND,
        EndGameScenario.BRITANNIA_ISLE_OF_MAN,
    }
    targeted_scoring = britannia | {EndGameScenario.EAST_WEST}
    if request.prepared_route_full is not None and request.scenario not in targeted_scoring:
        raise ValueError("prepared_route_full requires a targeted scoring scenario")
    if request.scenario in britannia and request.map_num not in (None, 3):
        raise ValueError("Britannia scenarios require Map 3")
    if (
        request.scenario
        in (EndGameScenario.BRITANNIA_SCOTLAND, EndGameScenario.BRITANNIA_ISLE_OF_MAN)
        and request.player_count == 3
    ):
        raise ValueError("Scotland and dual-region scenarios require 4 or 5 players")
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
        "east_west_path_length": (
            generated.target_variant if generated.scenario is EndGameScenario.EAST_WEST else None
        ),
        "target_variant": generated.target_variant,
        "prepared_player_index": generated.prepared_player_index,
        "prepared_route_full": generated.prepared_route_full,
        "development_range": generated.development_range,
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
        "target_variant": generated.target_variant,
        "prepared_player_index": generated.prepared_player_index,
        "prepared_route_full": generated.prepared_route_full,
        "development_range": generated.development_range,
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
