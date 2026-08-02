"""Thorough validation for game graphs crossing the save-file boundary."""

from collections import Counter

from map_data.constants import (
    ACTIONS_MAX_VALUES,
    BANK_MAX_VALUES,
    BOOK_OF_KNOWLEDGE_MAX_VALUES,
    CITY_KEYS_MAX_VALUES,
    PRIVILEGE_COLORS,
)
from map_data.map_attributes import Map
from map_data.map1 import Map1
from map_data.map2 import Map2
from map_data.map3 import Map3

from game.action_schema import TILE_TYPES
from game.invariants import GameInvariantError, validate_game


def _require(condition, message):
    if not condition:
        raise GameInvariantError(message)


def validate_loaded_game(game):
    """Reject structurally impossible or internally contradictory saved games.

    This deliberately runs only when saving or loading.  The normal engine keeps
    using :func:`validate_game`, whose inexpensive checks are suitable after each
    action.
    """
    validate_game(game)
    players = game.players
    known_players = set(players)

    _require(game.num_players == len(players), "num_players does not match players")
    _require(game.map_num in (1, 2, 3), f"unsupported map number: {game.map_num}")
    expected_map_type = {1: Map1, 2: Map2, 3: Map3}[game.map_num]
    _require(
        isinstance(game.selected_map, expected_map_type),
        f"selected map does not match map {game.map_num}",
    )
    _require(
        [player.order for player in players] == list(range(1, len(players) + 1)),
        "player orders must be consecutive and match their seats",
    )
    _require(len({player.color for player in players}) == len(players), "player colors repeat")
    configuration = getattr(game, "configuration", None)
    if configuration is not None:
        configuration.validate()
        _require(configuration.map_num == game.map_num, "configuration map does not match game")
        _require(
            configuration.player_count == len(players),
            "configuration player count does not match game",
        )

    for player in players:
        _require(0 <= player.keys_index < len(CITY_KEYS_MAX_VALUES), "invalid Keys level")
        _require(
            player.keys == CITY_KEYS_MAX_VALUES[player.keys_index], "Keys value/index disagree"
        )
        _require(player.privilege in PRIVILEGE_COLORS, "invalid Privilege level")
        _require(player.book in BOOK_OF_KNOWLEDGE_MAX_VALUES, "invalid Book level")
        _require(0 <= player.actions_index < len(ACTIONS_MAX_VALUES), "invalid Actions level")
        _require(
            player.actions == ACTIONS_MAX_VALUES[player.actions_index],
            "Actions value/index disagree",
        )
        _require(player.bank in BANK_MAX_VALUES, "invalid Bank level")
        _require(player.board is not None and player.board.player is player, "invalid player board")
        _require(player.score >= 0 and player.final_score >= 0, "player score is negative")
        legal_turn_starts = set(ACTIONS_MAX_VALUES)
        if game.OneActionOwner is player:
            legal_turn_starts.update(value + 1 for value in ACTIONS_MAX_VALUES)
        _require(
            player.actions_at_turn_start in legal_turn_starts
            and player.actions_at_turn_start <= player.actions + int(game.OneActionOwner is player),
            f"player {player.order} has an invalid turn-start action count",
        )
        _require(
            player.actions_granted_this_turn >= 0,
            f"player {player.order} has a negative action grant",
        )
        _require(
            player is game.current_player or player.actions_granted_this_turn == 0,
            f"player {player.order} retains an action grant outside their turn",
        )
        _require(
            player.actions_remaining
            <= player.actions_at_turn_start + player.actions_granted_this_turn,
            f"player {player.order} has too many remaining actions",
        )

    _validate_map_graph(game, known_players)
    _validate_optional_modules(game, known_players)
    _validate_pending_state(game, known_players)
    _validate_terminal_state(game)
    return True


def _validate_map_graph(game, known_players):
    selected_map = game.selected_map
    _require(selected_map is not None, "game has no selected map")
    known_cities = set(selected_map.cities)
    known_routes = set(selected_map.routes)
    _require(len(known_cities) == len(selected_map.cities), "map contains a duplicate city object")
    _require(len(known_routes) == len(selected_map.routes), "map contains a duplicate route object")

    for route in selected_map.routes:
        _require(len(route.cities) == 2, "route must connect exactly two cities")
        _require(
            all(city in known_cities for city in route.cities), "route references an unknown city"
        )
        _require(all(route in city.routes for city in route.cities), "city/route links disagree")
        for post in route.posts:
            _require(
                post.required_shape in (None, "square", "circle"), "invalid required post shape"
            )
            if post.owner is not None:
                _require(
                    post.required_shape in (None, post.owner_piece_shape),
                    "piece violates a post's required shape",
                )
        _require(
            all(owner in known_players for owner in route.tribute_owners), "unknown Tribute owner"
        )
        _require(
            all(owner in known_players for owner in route.block_marker_owners),
            "unknown Block owner",
        )

    for city in selected_map.cities:
        _require(len(set(city.routes)) == len(city.routes), f"{city.name} repeats a route")
        _require(
            all(route in known_routes for route in city.routes), f"{city.name} has an unknown route"
        )
        for office in city.offices:
            _require(office.shape in ("square", "circle"), f"invalid office shape in {city.name}")
            _require(
                office.owner_piece_shape in (None, "square", "circle"),
                f"invalid occupied office shape in {city.name}",
            )
            _require(
                office.controller is not None or office.owner_piece_shape is None,
                f"empty office has an occupied piece shape in {city.name}",
            )

    _require(
        game.original_route_of_displacement is None
        or game.original_route_of_displacement in known_routes,
        "displacement references an unknown route",
    )
    _require(
        all(
            post in {p for route in known_routes for p in route.posts}
            for post in game.all_empty_posts
        ),
        "displacement references an unknown post",
    )


def _validate_optional_modules(game, known_players):
    if game.use_mission_cards:
        _require(game.map_num == 1, "Mission Cards are only valid on map 1")
        cards = [tuple(player.mission_card or ()) for player in game.players]
        city_names = {city.name for city in game.selected_map.cities}
        _require(
            all(len(card) == 3 for card in cards),
            "each player must have one three-city Mission Card",
        )
        _require(len(set(cards)) == len(cards), "Mission Cards assigned to players repeat")
        _require(
            all(set(card) <= city_names for card in cards),
            "Mission Card references an unknown city",
        )
    else:
        _require(
            all(player.mission_card is None for player in game.players),
            "Mission Card exists while disabled",
        )

    tile_owners = {
        "DisplaceAnywhere": game.DisplaceAnywhereOwner,
        "+1Action": game.OneActionOwner,
        "+1IncomeIfOthersIncome": game.OneIncomeIfOthersIncomeOwner,
        "+1DisplacedPiece": game.OneDisplacedPieceOwner,
        "+4PtsPerOwnedCity": game.FourPtsPerOwnedCityOwner,
        "+7PtsPerCompletedAbility": game.SevenPtsPerCompletedAbilityOwner,
    }
    owned_tiles = [tile for player in game.players for tile in player.tiles]
    if game.use_emperors_favour:
        _require(
            all(tile in TILE_TYPES for tile in game.tile_pool + owned_tiles),
            "unknown Emperor's Favour tile",
        )
        _require(
            len(game.tile_pool) + len(owned_tiles) == len(game.players),
            "Emperor's Favour tile count changed",
        )
        _require(
            len(set(game.tile_pool + owned_tiles)) == len(game.tile_pool + owned_tiles),
            "Emperor's Favour tile repeats",
        )
        for tile, owner in tile_owners.items():
            _require(
                owner is None or (owner in known_players and tile in owner.tiles),
                f"invalid owner for {tile}",
            )
        for player in game.players:
            for tile in player.tiles:
                _require(tile_owners[tile] is player, f"{tile} ownership fields disagree")
    else:
        _require(
            not game.tile_pool and not owned_tiles, "Emperor's Favour tile exists while disabled"
        )
        _require(
            all(owner is None for owner in tile_owners.values()),
            "Emperor's Favour owner exists while disabled",
        )
        _require(
            game.tile_to_buy is None and not game.waiting_for_buy_tile_with_bm,
            "Emperor's Favour purchase pending while disabled",
        )

    allowed_markers = set(Map.STANDARD_BONUS_MARKER_SUPPLY) | set(Map.PROMO_BONUS_MARKERS)
    markers = [
        marker.type
        for player in game.players
        for marker in player.bonus_markers + player.used_bonus_markers
    ]
    markers += [route.bonus_marker.type for route in game.selected_map.routes if route.bonus_marker]
    markers += list(game.selected_map.bonus_marker_pool) + list(game.pending_bonus_markers)
    if game.pending_exchange_marker is not None:
        markers.append(game.pending_exchange_marker.type)
    _require(all(marker in allowed_markers for marker in markers), "unknown bonus-marker type")
    _require(
        len(markers) == 15, f"bonus-marker supply must contain 15 markers, found {len(markers)}"
    )
    marker_counts = Counter(markers)
    allowed_counts = {**Map.STANDARD_BONUS_MARKER_SUPPLY, **Map.PROMO_BONUS_MARKERS}
    for starting_marker in ("Move3", "SwapOffice", "PlaceAdjacent"):
        allowed_counts[starting_marker] += 1
    _require(
        all(count <= allowed_counts[marker] for marker, count in marker_counts.items()),
        "bonus-marker type exceeds its available supply",
    )
    configuration = getattr(game, "configuration", None)
    if configuration is not None and not configuration.use_promo_markers:
        _require(
            not (set(markers) & set(Map.PROMO_BONUS_MARKERS)),
            "promotional bonus marker exists while disabled",
        )


def _validate_pending_state(game, known_players):
    player_refs = (
        game.pending_income_favour_owner,
        game.exchange_target_player,
        game.cardiff_priv,
        game.carlisle_priv,
        game.london_priv,
    )
    _require(
        all(player is None or player in known_players for player in player_refs),
        "pending state references an unknown player",
    )
    _require(
        all(player in known_players for player in game.pending_tribute_income_owners),
        "pending Income references an unknown player",
    )
    _require(game.replace_bonus_marker >= 0, "negative replacement bonus-marker count")
    _require(
        len(game.pending_bonus_markers) <= game.replace_bonus_marker,
        "replacement bonus-marker state is inconsistent",
    )
    held_by = [player for player in game.players if player.holding_pieces]
    allowed_holder = (
        game.displaced_player.player if game.waiting_for_displaced_player else game.current_player
    )
    _require(
        all(player is allowed_holder for player in held_by),
        "pieces are held by a player outside the active workflow",
    )
    for holder in held_by:
        for shape, owner, _region in holder.holding_pieces:
            _require(shape in ("square", "circle"), "held piece has an invalid shape")
            _require(owner in known_players, "held piece has an unknown owner")

    if game.waiting_for_bm_exchange_bm:
        _require(game.pending_exchange_marker is not None, "Exchange Bonus Marker has no marker")
    else:
        _require(game.pending_exchange_marker is None, "stale Exchange Bonus Marker")
        _require(game.exchange_target_player is None, "stale Exchange Bonus Marker target")

    if game.waiting_for_buy_tile_with_bm:
        _require(game.use_emperors_favour, "tile purchase is pending while its module is disabled")
        _require(game.tile_to_buy in game.tile_pool, "pending tile purchase is unavailable")
    else:
        _require(game.tile_to_buy is None, "stale pending tile purchase")
        _require(game.first_bm_to_spend_on_tile is None, "stale tile payment marker")
    if game.waiting_for_displaced_player:
        displaced = game.displaced_player
        initial_piece_count = 3 if displaced.displaced_shape == "circle" else 2
        if (
            displaced.displaced_shape == "square"
            and game.OneDisplacedPieceOwner is displaced.player
        ):
            initial_piece_count += 1
        maximum_remaining = initial_piece_count - int(displaced.played_displaced_shape)
        _require(
            1 <= displaced.total_pieces_to_place <= maximum_remaining,
            "active displacement has an impossible remaining-piece count",
        )
        _require(
            not displaced.use_optional_displaced_shape
            or (not displaced.played_displaced_shape and displaced.total_pieces_to_place > 1),
            "optional displaced piece selection is inconsistent",
        )
    else:
        displaced = game.displaced_player
        _require(displaced.displaced_shape is None, "stale displaced shape")
        _require(not displaced.played_displaced_shape, "stale displaced-piece progress")
        _require(not displaced.use_optional_displaced_shape, "stale optional displaced piece")
        _require(displaced.total_pieces_to_place == 0, "stale displacement piece count")


def _validate_terminal_state(game):
    full_cities = sum(city.city_is_full() for city in game.selected_map.cities)
    _require(
        game.current_full_cities_count == full_cities,
        "completed-city count does not match the board",
    )
    ending_condition = (
        game.bonus_pool_exhausted_during_claim
        or any(player.score >= 20 for player in game.players)
        or full_cities >= game.selected_map.max_full_cities
    )
    if game.game_end:
        _require(ending_condition, "completed game has no valid ending condition")
        _require(not game.pending_workflows, "completed game has a pending workflow")
        _require(
            not game.game_end_pending_immediate_resolution,
            "completed game still has deferred ending",
        )
        _require(game.current_player.actions_remaining == 0, "completed game has actions remaining")
        _require(
            all(player.final_score_breakdown for player in game.players),
            "completed game has no final scoring breakdown",
        )
    elif game.game_end_pending_immediate_resolution:
        _require(ending_condition, "deferred game end has no valid ending condition")
        _require(bool(game.pending_workflows), "deferred game end has no pending workflow")
    else:
        _require(not ending_condition, "active game has passed a mandatory ending condition")
        _require(
            all(
                player.final_score == 0 and not player.final_score_breakdown
                for player in game.players
            ),
            "active game contains final scoring",
        )
