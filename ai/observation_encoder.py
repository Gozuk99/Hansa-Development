"""Headless, player-visible neural-network observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from game.turn_state import TurnPhase
from map_data.constants import (
    ACTIONS_MAX_VALUES,
    BANK_MAX_VALUES,
    BLACK,
    BLACKISH_BROWN,
    BOOK_OF_KNOWLEDGE_MAX_VALUES,
    DARK_BLUE,
    DARK_GREEN,
    DARK_RED,
    GREY,
    ORANGE,
    PINK,
    PRIVILEGE_COLORS,
    WHITE,
)


@dataclass(frozen=True)
class AIObservation:
    """One player's visible features and legal choices for the same decision."""

    features: torch.Tensor
    legal_action_mask: torch.Tensor
    observer_index: int


class ObservationEncoder:
    """Encode the documented observation schema from an acting player's view."""

    MAX_PLAYERS = 5
    MAX_CITIES = 30
    MAX_OFFICES = 10
    MAX_ROUTES = 40
    MAX_POSTS_PER_ROUTE = 5
    MAX_MARKERS_PER_COLLECTION = 15
    MAX_HELD_PIECES = 5

    GAME_SIZE = 25
    PLAYER_SIZE = 56
    CITY_SIZE = 78
    ROUTE_SIZE = 38
    OPTIONAL_COMPONENTS_SIZE = 33
    WORKFLOW_SIZE = 43
    FEATURE_SIZE = (
        GAME_SIZE
        + MAX_PLAYERS * PLAYER_SIZE
        + MAX_CITIES * CITY_SIZE
        + MAX_ROUTES * ROUTE_SIZE
        + OPTIONAL_COMPONENTS_SIZE
        + WORKFLOW_SIZE
    )

    PIECE_TYPE_TO_ID = {None: 0, "square": 1, "circle": 2}
    REQUIRED_SHAPE_TO_ID = {None: 0, "square": 1, "circle": 2}
    PRIVILEGE_TO_ID = {None: 0, "WHITE": 1, "ORANGE": 2, "PINK": 3, "BLACK": 4}
    REGION_TO_ID = {None: 0, "Scotland": 1, "Wales": 2}
    PHASE_TO_ID = {phase: index for index, phase in enumerate(TurnPhase)}
    BONUS_MARKER_TYPE_TO_ID = {
        "PlaceAdjacent": 1,
        "SwapOffice": 2,
        "Move3": 3,
        "UpgradeAbility": 4,
        "3Actions": 5,
        "4Actions": 6,
        "ExchangeBonusMarker": 7,
        "Tribute4EstablishingTP": 8,
        "BlockTradeRoute": 9,
    }
    TILE_TYPE_TO_ID = {
        "DisplaceAnywhere": 1,
        "+1Action": 2,
        "+1IncomeIfOthersIncome": 3,
        "+1DisplacedPiece": 4,
        "+4PtsPerOwnedCity": 5,
        "+7PtsPerCompletedAbility": 6,
    }
    PERMANENT_MARKER_TYPE_TO_ID = {
        None: 0,
        "MoveAny2": 1,
        "+1Priv": 2,
        "ClaimGreenCity": 3,
        "Place2TradesmenFromRoute": 4,
        "Place2ScotlandOrWales": 5,
    }
    UPGRADE_TYPE_TO_ID = {
        "Keys": 1,
        "Privilege": 2,
        "Book": 3,
        "Actions": 4,
        "Bank": 5,
        "SpecialPrestigePoints": 6,
    }
    CITY_TYPE_TO_ID = {
        tuple(GREY): 1,
        tuple(BLACKISH_BROWN): 2,
        tuple(DARK_RED): 3,
        tuple(DARK_GREEN): 4,
        tuple(DARK_BLUE): 5,
        (65, 103, 114): 6,
    }
    ROUTE_TYPE_TO_ID = {tuple(WHITE): 1, tuple(BLACKISH_BROWN): 2}
    BONUS_WORKFLOW_TO_ID = {
        "waiting_for_bm_swap_office": 1,
        "waiting_for_bm_place_adjacent": 2,
        "waiting_for_bm_upgrade_ability": 3,
        "waiting_for_bm_move_any_2": 4,
        "waiting_for_bm_move3": 5,
        "waiting_for_bm_exchange_bm": 6,
        "waiting_for_bm_tribute_trading_post": 7,
        "waiting_for_bm_block_trade_route": 8,
        "waiting_for_bm_green_city": 9,
    }

    def __init__(self):
        self.game_tensor_size = self.GAME_SIZE
        self.city_tensor_size = self.MAX_CITIES * self.CITY_SIZE
        self.route_tensor_size = self.MAX_ROUTES * self.ROUTE_SIZE
        self.player_tensor_size = self.MAX_PLAYERS * self.PLAYER_SIZE
        self.all_game_state_size = self.FEATURE_SIZE

    def build(self, game) -> AIObservation:
        observer_index = self._observer_index(game)
        features = self.get_game_state(game, observer_index=observer_index)
        mask = torch.tensor(game.ai_action_mask(), dtype=torch.uint8)
        return AIObservation(features, mask, observer_index)

    def get_game_state(self, game, observer_index=None):
        """Return a flat, fixed-size feature tensor for one acting player."""
        if observer_index is None:
            observer_index = self._observer_index(game)
        self._validate_observer(game, observer_index)

        relative_players = self._relative_players(game, observer_index)
        owner_ids = {player: index + 1 for index, player in enumerate(relative_players)}
        groups = (
            self._game_features(game, owner_ids),
            self._player_features(game, relative_players),
            self._city_features(game, owner_ids),
            self._route_features(game, owner_ids),
            self._optional_features(game, owner_ids),
            self._workflow_features(game, relative_players[0], owner_ids),
        )
        features = torch.tensor([value for group in groups for value in group], dtype=torch.int16)
        if features.numel() != self.FEATURE_SIZE:
            raise RuntimeError(
                f"Observation has {features.numel()} values; expected {self.FEATURE_SIZE}"
            )
        return features

    @staticmethod
    def _observer_index(game):
        index = game.active_player
        if type(index) is not int or not 0 <= index < len(game.players):
            raise ValueError(f"Invalid active player index: {index!r}")
        return index

    @staticmethod
    def _validate_observer(game, observer_index):
        if type(observer_index) is not int or not 0 <= observer_index < len(game.players):
            raise ValueError(f"Invalid observer index: {observer_index!r}")

    @staticmethod
    def _relative_players(game, observer_index):
        return game.players[observer_index:] + game.players[:observer_index]

    @staticmethod
    def _pad(values, capacity, label):
        values = list(values)
        if len(values) > capacity:
            raise ValueError(f"{label} exceeds observation capacity {capacity}: {len(values)}")
        return values + [0] * (capacity - len(values))

    @staticmethod
    def _position(value, track, label):
        try:
            return track.index(value)
        except ValueError as error:
            raise ValueError(f"Unknown {label} value: {value!r}") from error

    def _game_features(self, game, owner_ids):
        east_west_players = [
            int(player in game.players_who_completed_east_west)
            for player in sorted(owner_ids, key=owner_ids.get)
        ]
        east_west_players = self._pad(east_west_players, 5, "East-West players")
        tribute_queue = self._pad(
            (owner_ids[player] for player in game.pending_tribute_income_owners),
            5,
            "tribute response queue",
        )
        return [
            game.map_num,
            game.num_players,
            game.turn_number,
            game.round_number,
            owner_ids[game.current_player],
            self.PHASE_TO_ID[game.turn_phase],
            int(game.use_mission_cards),
            int(game.use_emperors_favour),
            20,
            game.current_full_cities_count,
            game.selected_map.max_full_cities,
            len(game.selected_map.bonus_marker_pool),
            game.east_west_completed_count,
            *east_west_players,
            int(game.bonus_pool_exhausted_during_claim),
            *tribute_queue,
            int(game.game_end_pending_immediate_resolution),
        ]

    def _player_features(self, game, relative_players):
        features = []
        for relative_index in range(self.MAX_PLAYERS):
            if relative_index >= len(relative_players):
                features.extend([0] * self.PLAYER_SIZE)
                continue
            player = relative_players[relative_index]
            unused = self._marker_ids(
                player.bonus_markers, self.MAX_MARKERS_PER_COLLECTION, "unused bonus markers"
            )
            used = self._marker_ids(
                player.used_bonus_markers, self.MAX_MARKERS_PER_COLLECTION, "used bonus markers"
            )
            owned_tiles = [int(tile in player.tiles) for tile in self.TILE_TYPE_TO_ID]
            mission = self.assign_mission_card_mapping(game, player, observer=relative_players[0])
            features.extend(
                [
                    1,
                    player.score,
                    player.final_score,
                    player.general_stock_squares,
                    player.general_stock_circles,
                    player.personal_supply_squares,
                    player.personal_supply_circles,
                    player.keys_index,
                    self._position(player.privilege, PRIVILEGE_COLORS, "privilege"),
                    self._position(player.book, BOOK_OF_KNOWLEDGE_MAX_VALUES, "book"),
                    player.actions_index,
                    self._position(player.bank, BANK_MAX_VALUES, "bank"),
                    player.actions_remaining,
                    *unused,
                    *used,
                    *owned_tiles,
                    player.brown_priv_count,
                    player.blue_priv_count,
                    player.london_priv_count,
                    *mission,
                    int(
                        player is relative_players[0]
                        and game.use_mission_cards
                        and bool(player.mission_card)
                    ),
                ]
            )
        return features

    def _marker_ids(self, markers, capacity, label):
        ids = [self.BONUS_MARKER_TYPE_TO_ID[marker.type] for marker in markers]
        return self._pad(ids, capacity, label)

    def _city_features(self, game, owner_ids):
        features = []
        for city_index in range(self.MAX_CITIES):
            if city_index >= len(game.selected_map.cities):
                features.extend([0] * self.CITY_SIZE)
                continue
            city = game.selected_map.cities[city_index]
            if len(city.offices) > self.MAX_OFFICES:
                raise ValueError(
                    f"City {city.name} has {len(city.offices)} offices; capacity is {self.MAX_OFFICES}"
                )
            upgrades = self._pad(
                (self.UPGRADE_TYPE_TO_ID[value] for value in city.upgrade_city_type),
                2,
                f"{city.name} upgrades",
            )
            tributes = self._pad(
                (owner_ids.get(player, 0) for player in city.tributed_players),
                4,
                f"{city.name} tribute owners",
            )
            features.extend([1, self.CITY_TYPE_TO_ID[tuple(city.color)], *upgrades, *tributes])
            for office_index in range(self.MAX_OFFICES):
                if office_index >= len(city.offices):
                    features.extend([0] * 7)
                    continue
                office = city.offices[office_index]
                features.extend(
                    [
                        1,
                        int(office.place_adjacent_office),
                        self.PIECE_TYPE_TO_ID[office.shape],
                        self.PRIVILEGE_TO_ID.get(
                            getattr(office, "printed_privilege", self._color_name(office.color)),
                            0,
                        ),
                        office.awards_points,
                        owner_ids.get(office.controller, 0),
                        self.PIECE_TYPE_TO_ID[office.owner_piece_shape],
                    ]
                )
        return features

    def _route_features(self, game, owner_ids):
        city_ids = {city: index + 1 for index, city in enumerate(game.selected_map.cities)}
        features = []
        for route_index in range(self.MAX_ROUTES):
            if route_index >= len(game.selected_map.routes):
                features.extend([0] * self.ROUTE_SIZE)
                continue
            route = game.selected_map.routes[route_index]
            if len(route.posts) > self.MAX_POSTS_PER_ROUTE:
                raise ValueError(
                    f"Route {route_index} has {len(route.posts)} posts; capacity is {self.MAX_POSTS_PER_ROUTE}"
                )
            tribute_ids = self._pad(
                (owner_ids.get(owner, 0) for owner in route.tribute_owners),
                5,
                f"route {route_index} tribute owners",
            )
            block_ids = self._pad(
                (owner_ids.get(owner, 0) for owner in route.block_marker_owners),
                5,
                f"route {route_index} block owners",
            )
            features.extend(
                [
                    1,
                    city_ids[route.cities[0]],
                    city_ids[route.cities[1]],
                    self.REGION_TO_ID[route.region],
                    self.ROUTE_TYPE_TO_ID.get(tuple(route.color), 0),
                    route.required_circles,
                    self.BONUS_MARKER_TYPE_TO_ID.get(
                        route.bonus_marker.type if route.bonus_marker else None, 0
                    ),
                    self.PERMANENT_MARKER_TYPE_TO_ID.get(route.has_permanent_bm_type, 0),
                    *tribute_ids,
                    *block_ids,
                ]
            )
            for post_index in range(self.MAX_POSTS_PER_ROUTE):
                if post_index >= len(route.posts):
                    features.extend([0] * 4)
                    continue
                post = route.posts[post_index]
                features.extend(
                    [
                        1,
                        self.REQUIRED_SHAPE_TO_ID[post.required_shape],
                        owner_ids.get(post.owner, 0),
                        self.PIECE_TYPE_TO_ID[post.owner_piece_shape],
                    ]
                )
        return features

    def _optional_features(self, game, owner_ids):
        available_tiles = [int(tile in game.tile_pool) for tile in self.TILE_TYPE_TO_ID]
        prestige = []
        special = getattr(game.selected_map, "specialprestigepoints", None)
        circles = special.circle_data if special is not None else []
        for index in range(4):
            if index >= len(circles):
                prestige.extend([0, 0, 0])
                continue
            circle = circles[index]
            prestige.extend(
                [
                    circle["value"],
                    self.PRIVILEGE_TO_ID.get(self._color_name(circle["color"]), 0),
                    owner_ids.get(circle["owner"], 0),
                ]
            )
        visible_pending_markers = (
            game.pending_bonus_markers if game.turn_phase == TurnPhase.REPLACE_BONUS_MARKERS else ()
        )
        pending = self._pad(
            (self.BONUS_MARKER_TYPE_TO_ID[value] for value in visible_pending_markers),
            15,
            "pending replacement markers",
        )
        return [*available_tiles, *prestige, *pending]

    def _workflow_features(self, game, observer, owner_ids):
        held = []
        if len(observer.holding_pieces) > self.MAX_HELD_PIECES:
            raise ValueError(
                f"Held pieces exceed observation capacity {self.MAX_HELD_PIECES}: "
                f"{len(observer.holding_pieces)}"
            )
        for shape, owner, region in observer.holding_pieces:
            held.extend(
                [self.PIECE_TYPE_TO_ID[shape], owner_ids.get(owner, 0), self.REGION_TO_ID[region]]
            )
        held.extend([0] * (self.MAX_HELD_PIECES * 3 - len(held)))

        displaced = game.displaced_player
        displacement_active = game.waiting_for_displaced_player
        workflow_id = next(
            (
                workflow_id
                for attribute, workflow_id in self.BONUS_WORKFLOW_TO_ID.items()
                if getattr(game, attribute)
            ),
            0,
        )
        original_route = (
            game.selected_map.routes.index(game.original_route_of_displacement) + 1
            if game.original_route_of_displacement in game.selected_map.routes
            else 0
        )
        selected_target = owner_ids.get(game.exchange_target_player, 0)
        permanent_workflow = (
            2
            if game.pending_britannia_place2 or game.waiting_for_place2_in_scotland_or_wales
            else int(bool(game.pending_route_piece_choices or game.waiting_for_place2_from_route))
        )
        pending_pieces = []
        if len(game.pending_route_piece_choices) > self.MAX_HELD_PIECES:
            raise ValueError("Pending route pieces exceed observation capacity 5")
        for shape, owner, region in game.pending_route_piece_choices:
            pending_pieces.extend(
                [self.PIECE_TYPE_TO_ID[shape], owner_ids.get(owner, 0), self.REGION_TO_ID[region]]
            )
        pending_pieces.extend([0] * (15 - len(pending_pieces)))

        return [
            *held,
            observer.pieces_to_pickup,
            observer.pieces_to_place,
            self.PIECE_TYPE_TO_ID[displaced.displaced_shape] if displacement_active else 0,
            int(displacement_active and displaced.played_displaced_shape),
            int(displacement_active and displaced.use_optional_displaced_shape),
            displaced.total_pieces_to_place if displacement_active else 0,
            original_route,
            workflow_id,
            selected_target,
            self.TILE_TYPE_TO_ID.get(game.tile_to_buy, 0),
            self.BONUS_MARKER_TYPE_TO_ID.get(game.first_bm_to_spend_on_tile, 0),
            permanent_workflow,
            *pending_pieces,
            game.replace_bonus_marker,
        ]

    def assign_mission_card_mapping(self, game, player, observer=None):
        """Compatibility helper: expose only the observer's own Mission Card."""
        if observer is None:
            observer = game.players[self._observer_index(game)]
        if not game.use_mission_cards or player is not observer or not player.mission_card:
            return (0, 0, 0)
        city_ids = {city.name: index + 1 for index, city in enumerate(game.selected_map.cities)}
        return tuple(city_ids[name] for name in player.mission_card)

    @staticmethod
    def _color_name(color):
        mapping = {
            tuple(WHITE): "WHITE",
            tuple(ORANGE): "ORANGE",
            tuple(PINK): "PINK",
            tuple(BLACK): "BLACK",
        }
        return mapping.get(tuple(color))
