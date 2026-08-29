"""Headless, player-visible neural-network observations."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch

from ai.observation_schema import OBSERVATION_SIZE
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
    MOVE_SNAPSHOT_SIZE = MAX_ROUTES * MAX_POSTS_PER_ROUTE * 2
    PAID_ACTION_HISTORY_SIZE = 3
    ROUTE_REWARD_HISTORY_SIZE = MAX_ROUTES * 2
    FEATURE_SIZE = (
        GAME_SIZE
        + MAX_PLAYERS * PLAYER_SIZE
        + MAX_CITIES * CITY_SIZE
        + MAX_ROUTES * ROUTE_SIZE
        + OPTIONAL_COMPONENTS_SIZE
        + WORKFLOW_SIZE
        + MOVE_SNAPSHOT_SIZE
        + PAID_ACTION_HISTORY_SIZE
        + ROUTE_REWARD_HISTORY_SIZE
    )
    ROUTE_REWARD_HISTORY_START = FEATURE_SIZE - ROUTE_REWARD_HISTORY_SIZE
    PAID_ACTION_HISTORY_START = ROUTE_REWARD_HISTORY_START - PAID_ACTION_HISTORY_SIZE
    MOVE_SNAPSHOT_START = PAID_ACTION_HISTORY_START - MOVE_SNAPSHOT_SIZE
    if FEATURE_SIZE != OBSERVATION_SIZE:
        raise RuntimeError(
            f"Observation layout has {FEATURE_SIZE} values; schema declares {OBSERVATION_SIZE}"
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
    HIDDEN_USED_BONUS_MARKER_ID = 10
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
        self._structural_templates = WeakKeyDictionary()

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
        features = list(self._structural_template(game))
        features[: self.GAME_SIZE] = self._game_features(game, owner_ids)
        player_start = self.GAME_SIZE
        city_start = player_start + self.MAX_PLAYERS * self.PLAYER_SIZE
        route_start = city_start + self.MAX_CITIES * self.CITY_SIZE
        optional_start = route_start + self.MAX_ROUTES * self.ROUTE_SIZE
        workflow_start = optional_start + self.OPTIONAL_COMPONENTS_SIZE
        move_snapshot_start = workflow_start + self.WORKFLOW_SIZE
        paid_action_history_start = move_snapshot_start + self.MOVE_SNAPSHOT_SIZE
        route_reward_history_start = paid_action_history_start + self.PAID_ACTION_HISTORY_SIZE
        features[player_start:city_start] = self._player_features(game, relative_players)
        self._write_dynamic_city_features(features, city_start, game, owner_ids)
        self._write_dynamic_route_features(features, route_start, game, owner_ids)
        self._write_dynamic_optional_features(features, optional_start, game, owner_ids)
        features[workflow_start:move_snapshot_start] = self._workflow_features(
            game, relative_players[0], owner_ids
        )
        features[move_snapshot_start:paid_action_history_start] = (
            self._normal_move_snapshot_features(game, owner_ids)
        )
        features[paid_action_history_start:route_reward_history_start] = (
            self._paid_action_history_features(game)
        )
        features[route_reward_history_start:] = self._route_reward_history_features(game)
        features = torch.frombuffer(array("h", features), dtype=torch.int16)
        if features.numel() != self.FEATURE_SIZE:
            raise RuntimeError(
                f"Observation has {features.numel()} values; expected {self.FEATURE_SIZE}"
            )
        return features

    def _structural_template(self, game):
        revision = getattr(game, "_observation_structure_revision", 0)
        cached = self._structural_templates.get(game)
        if cached is not None and cached[0] == revision:
            return cached[2]

        signature = self._structural_signature(game)

        template = [0] * self.FEATURE_SIZE
        template[0] = game.map_num
        template[1] = game.num_players
        template[6] = int(game.use_mission_cards)
        template[7] = int(game.use_emperors_favour)
        template[8] = 20
        template[10] = game.selected_map.max_full_cities

        city_start = self.GAME_SIZE + self.MAX_PLAYERS * self.PLAYER_SIZE
        for city_index, city in enumerate(game.selected_map.cities):
            base = city_start + city_index * self.CITY_SIZE
            template[base] = 1
            template[base + 1] = self.CITY_TYPE_TO_ID[tuple(city.color)]
            upgrades = self._pad(
                (self.UPGRADE_TYPE_TO_ID[value] for value in city.upgrade_city_type),
                2,
                f"{city.name} upgrades",
            )
            template[base + 2 : base + 4] = upgrades

        route_start = city_start + self.MAX_CITIES * self.CITY_SIZE
        city_ids = {city: index + 1 for index, city in enumerate(game.selected_map.cities)}
        for route_index, route in enumerate(game.selected_map.routes):
            base = route_start + route_index * self.ROUTE_SIZE
            template[base : base + 6] = (
                1,
                city_ids[route.cities[0]],
                city_ids[route.cities[1]],
                self.REGION_TO_ID[route.region],
                self.ROUTE_TYPE_TO_ID.get(tuple(route.color), 0),
                route.required_circles,
            )
            template[base + 7] = self.PERMANENT_MARKER_TYPE_TO_ID.get(
                route.has_permanent_bm_type, 0
            )
            for post_index, post in enumerate(route.posts):
                post_base = base + 18 + post_index * 4
                template[post_base] = 1
                template[post_base + 1] = self.REQUIRED_SHAPE_TO_ID[post.required_shape]

        optional_start = route_start + self.MAX_ROUTES * self.ROUTE_SIZE
        special = getattr(game.selected_map, "specialprestigepoints", None)
        circles = special.circle_data if special is not None else []
        for index, circle in enumerate(circles[:4]):
            base = optional_start + 6 + index * 3
            template[base] = circle["value"]
            template[base + 1] = self.PRIVILEGE_TO_ID.get(self._color_name(circle["color"]), 0)

        template = tuple(template)
        self._structural_templates[game] = (revision, signature, template)
        return template

    @staticmethod
    def _structural_signature(game):
        special = getattr(game.selected_map, "specialprestigepoints", None)
        circles = special.circle_data if special is not None else []
        return (
            game.map_num,
            game.num_players,
            game.use_mission_cards,
            game.use_emperors_favour,
            game.selected_map.max_full_cities,
            tuple(
                (tuple(city.color), tuple(city.upgrade_city_type))
                for city in game.selected_map.cities
            ),
            tuple(
                (
                    tuple(route.color),
                    route.region,
                    route.required_circles,
                    route.has_permanent_bm_type,
                    tuple(post.required_shape for post in route.posts),
                )
                for route in game.selected_map.routes
            ),
            tuple((circle["value"], tuple(circle["color"])) for circle in circles[:4]),
        )

    def _write_dynamic_city_features(self, features, start, game, owner_ids):
        for city_index, city in enumerate(game.selected_map.cities):
            if len(city.offices) > self.MAX_OFFICES:
                raise ValueError(
                    f"City {city.name} has {len(city.offices)} offices; "
                    f"capacity is {self.MAX_OFFICES}"
                )
            base = start + city_index * self.CITY_SIZE
            tributes = self._pad(
                (owner_ids.get(player, 0) for player in city.tributed_players),
                4,
                f"{city.name} tribute owners",
            )
            features[base + 4 : base + 8] = tributes
            for office_index, office in enumerate(city.offices):
                office_base = base + 8 + office_index * 7
                features[office_base : office_base + 7] = (
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
                )

    def _write_dynamic_route_features(self, features, start, game, owner_ids):
        for route_index, route in enumerate(game.selected_map.routes):
            if len(route.posts) > self.MAX_POSTS_PER_ROUTE:
                raise ValueError(
                    f"Route {route_index} has {len(route.posts)} posts; "
                    f"capacity is {self.MAX_POSTS_PER_ROUTE}"
                )
            base = start + route_index * self.ROUTE_SIZE
            features[base + 6] = self.BONUS_MARKER_TYPE_TO_ID.get(
                route.bonus_marker.type if route.bonus_marker else None, 0
            )
            features[base + 8 : base + 13] = self._pad(
                (owner_ids.get(owner, 0) for owner in route.tribute_owners),
                5,
                f"route {route_index} tribute owners",
            )
            features[base + 13 : base + 18] = self._pad(
                (owner_ids.get(owner, 0) for owner in route.block_marker_owners),
                5,
                f"route {route_index} block owners",
            )
            for post_index, post in enumerate(route.posts):
                post_base = base + 18 + post_index * 4
                features[post_base + 2] = owner_ids.get(post.owner, 0)
                features[post_base + 3] = self.PIECE_TYPE_TO_ID[post.owner_piece_shape]

    def _write_dynamic_optional_features(self, features, start, game, owner_ids):
        features[start : start + 6] = [int(tile in game.tile_pool) for tile in self.TILE_TYPE_TO_ID]
        special = getattr(game.selected_map, "specialprestigepoints", None)
        circles = special.circle_data if special is not None else []
        for index, circle in enumerate(circles[:4]):
            features[start + 8 + index * 3] = owner_ids.get(circle["owner"], 0)
        visible_pending_markers = (
            game.pending_bonus_markers if game.turn_phase == TurnPhase.REPLACE_BONUS_MARKERS else ()
        )
        features[start + 18 : start + 33] = self._pad(
            (self.BONUS_MARKER_TYPE_TO_ID[value] for value in visible_pending_markers),
            15,
            "pending replacement markers",
        )

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
            used = self._used_marker_ids(game, player, relative_players[0])
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

    def _used_marker_ids(self, game, player, observer):
        marker_count = len(player.used_bonus_markers)
        if marker_count > self.MAX_MARKERS_PER_COLLECTION:
            raise ValueError(
                "used bonus markers exceed observation capacity "
                f"{self.MAX_MARKERS_PER_COLLECTION}: {marker_count}"
            )
        types_are_visible = player is observer or (
            game.waiting_for_bm_exchange_bm and game.exchange_target_player is player
        )
        if types_are_visible:
            return self._marker_ids(
                player.used_bonus_markers,
                self.MAX_MARKERS_PER_COLLECTION,
                "used bonus markers",
            )
        return self._pad(
            [self.HIDDEN_USED_BONUS_MARKER_ID] * marker_count,
            self.MAX_MARKERS_PER_COLLECTION,
            "hidden used bonus markers",
        )

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

    def _normal_move_snapshot_features(self, game, owner_ids):
        """Encode the immutable board occupancy from before the paid Move began."""
        snapshot = getattr(game, "normal_move_pre_board_snapshot", None)
        if snapshot is None:
            return [0] * self.MOVE_SNAPSHOT_SIZE

        features = [0] * self.MOVE_SNAPSHOT_SIZE
        for route_index, route_posts in enumerate(snapshot):
            if route_index >= self.MAX_ROUTES:
                raise ValueError("Move snapshot exceeds route observation capacity")
            if len(route_posts) > self.MAX_POSTS_PER_ROUTE:
                raise ValueError("Move snapshot exceeds post observation capacity")
            for post_index, (owner, shape) in enumerate(route_posts):
                base = (route_index * self.MAX_POSTS_PER_ROUTE + post_index) * 2
                features[base] = owner_ids.get(owner, 0)
                features[base + 1] = self.PIECE_TYPE_TO_ID[shape]
        return features

    @staticmethod
    def _paid_action_history_features(game):
        """Expose paid-action types already completed in the current turn."""
        player = game.current_player
        return [
            player.consecutive_paid_move_actions,
            player.paid_actions_spent_this_turn,
            player.paid_move_actions_spent_this_turn,
        ]

    def _route_reward_history_features(self, game):
        """Expose pending Move->Claim and consumed Move-focus state by route slot."""
        player = game.current_player
        pending = player.pending_move_claim_route_slots
        rewarded = player.rewarded_move_focus_route_slots
        features = [0] * self.ROUTE_REWARD_HISTORY_SIZE
        for route_slot in range(len(game.selected_map.routes)):
            base = route_slot * 2
            features[base] = int(route_slot in pending)
            features[base + 1] = int(route_slot in rewarded)
        return features

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
