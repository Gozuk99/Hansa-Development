"""Reusable, validated configuration for starting an interactive game."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
import random
from typing import Iterable, Sequence

from game.action_schema import TILE_TYPES
from game.game_info import Game
from game.setup import MAX_PLAYERS, MIN_PLAYERS, SUPPORTED_MAPS
from map_data.constants import INPUT_SIZE, OUTPUT_SIZE
from map_data.map_attributes import Map


EMPERORS_FAVOUR_TILES = TILE_TYPES


class PlayerControl(str, Enum):
    HUMAN = "Human"
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    MAGNUS = 'Impossible ("Magnus")'

    @property
    def is_human(self) -> bool:
        return self is PlayerControl.HUMAN


AI_DIFFICULTY_TOP_K = {
    PlayerControl.EASY: 15,
    PlayerControl.MEDIUM: 10,
    PlayerControl.HARD: 5,
    PlayerControl.MAGNUS: 1,
}

SELECTION_MODES = ("random", "manual")


def human_players(count: int) -> tuple[PlayerControl, ...]:
    return (PlayerControl.HUMAN,) * count


@dataclass(frozen=True)
class GameConfiguration:
    """Every choice needed to construct a game, independent of the GUI."""

    map_num: int = 1
    player_count: int = MIN_PLAYERS
    player_controls: tuple[PlayerControl, ...] = field(
        default_factory=lambda: human_players(MIN_PLAYERS)
    )
    use_mission_cards: bool = False
    use_emperors_favour: bool = False
    emperor_tile_mode: str = "random"
    emperor_tiles: tuple[str, ...] = ()
    use_promo_markers: bool = False
    promo_marker_mode: str = "random"
    promo_markers: tuple[str, ...] = ()
    seed: int | None = None
    difficulty_top_k: tuple[tuple[PlayerControl, int], ...] = field(
        default_factory=lambda: tuple(AI_DIFFICULTY_TOP_K.items())
    )

    def __post_init__(self) -> None:
        controls = tuple(PlayerControl(value) for value in self.player_controls)
        object.__setattr__(self, "player_controls", controls)
        object.__setattr__(self, "emperor_tiles", tuple(self.emperor_tiles))
        object.__setattr__(self, "promo_markers", tuple(self.promo_markers))
        object.__setattr__(self, "difficulty_top_k", tuple(self.difficulty_top_k))
        self.validate()

    def validate(self) -> None:
        if self.map_num not in SUPPORTED_MAPS:
            raise ValueError(f"Unsupported map number: {self.map_num}")
        if not MIN_PLAYERS <= self.player_count <= MAX_PLAYERS:
            raise ValueError(f"Player count must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
        if len(self.player_controls) != self.player_count:
            raise ValueError("Player controls must contain one entry per player")
        if self.use_mission_cards and self.map_num != 1:
            raise ValueError("Mission Cards can only be enabled on map 1")

        self._validate_optional_selection(
            enabled=self.use_emperors_favour,
            mode=self.emperor_tile_mode,
            selected=self.emperor_tiles,
            allowed=EMPERORS_FAVOUR_TILES,
            exact_count=self.player_count,
            label="Emperor's Favour tiles",
            unique=True,
        )
        self._validate_optional_selection(
            enabled=self.use_promo_markers,
            mode=self.promo_marker_mode,
            selected=self.promo_markers,
            allowed={
                **Map.STANDARD_BONUS_MARKER_SUPPLY,
                **Map.PROMO_BONUS_MARKERS,
            },
            exact_count=12,
            label="bonus-marker supply",
            unique=False,
        )

        thresholds = dict(self.difficulty_top_k)
        if set(thresholds) != set(AI_DIFFICULTY_TOP_K):
            raise ValueError("Difficulty thresholds must define every AI difficulty")
        if any(not isinstance(value, int) or value < 1 for value in thresholds.values()):
            raise ValueError("Difficulty thresholds must be positive integers")

    @staticmethod
    def _validate_optional_selection(
        *,
        enabled: bool,
        mode: str,
        selected: Sequence[str],
        allowed: Iterable[str],
        exact_count: int | None,
        label: str,
        unique: bool,
    ) -> None:
        if mode not in SELECTION_MODES:
            raise ValueError(f"Unknown {label} selection mode: {mode}")
        if not enabled and selected:
            raise ValueError(f"{label} cannot be selected while the module is disabled")
        if enabled and mode == "random" and selected:
            raise ValueError(f"Random {label} selection cannot include manual choices")
        if not enabled or mode != "manual":
            return

        allowed_counts = (
            dict(allowed) if isinstance(allowed, dict) else {value: 1 for value in allowed}
        )
        if not selected:
            raise ValueError(f"Manual {label} selection cannot be empty")
        if exact_count is not None and len(selected) != exact_count:
            raise ValueError(f"Select exactly {exact_count} {label}")
        if unique and len(set(selected)) != len(selected):
            raise ValueError(f"Manual {label} selections must be unique")
        for value in set(selected):
            if value not in allowed_counts:
                raise ValueError(f"Unknown {label} choice: {value}")
            if selected.count(value) > allowed_counts[value]:
                raise ValueError(f"Too many {value} selections")

    @property
    def has_ai_players(self) -> bool:
        return any(not control.is_human for control in self.player_controls)

    def top_k_for(self, control: PlayerControl) -> int:
        if control.is_human:
            raise ValueError("Human players do not have an AI difficulty")
        return dict(self.difficulty_top_k)[control]

    def with_player_count(self, count: int) -> GameConfiguration:
        controls = self.player_controls[:count]
        controls += human_players(max(0, count - len(controls)))
        tiles = self.emperor_tiles
        tile_mode = self.emperor_tile_mode
        if self.emperor_tile_mode == "manual" and len(tiles) != count:
            tiles = ()
            tile_mode = "random"
        return replace(
            self,
            player_count=count,
            player_controls=controls,
            emperor_tile_mode=tile_mode,
            emperor_tiles=tiles,
        )

    def resolved_bonus_marker_supply(self) -> tuple[str, ...] | None:
        if not self.use_promo_markers:
            return None

        rng = random.Random(self.seed)
        available_promos = [
            marker for marker, count in Map.PROMO_BONUS_MARKERS.items() for _ in range(count)
        ]
        if self.promo_marker_mode == "manual":
            return tuple(self.promo_markers)
        else:
            promos = rng.sample(
                available_promos,
                rng.randint(1, len(available_promos)),
            )

        available_standard = [
            marker
            for marker, count in Map.STANDARD_BONUS_MARKER_SUPPLY.items()
            for _ in range(count)
        ]
        standards = rng.sample(available_standard, 12 - len(promos))
        return tuple(standards + promos)

    def create_game(self):
        """Construct the engine and attach controller metadata to each player."""
        game = Game(
            map_num=self.map_num,
            num_players=self.player_count,
            load_models=False,
            seed=self.seed,
            use_mission_cards=self.use_mission_cards,
            use_emperors_favour=self.use_emperors_favour,
            bonus_marker_supply=self.resolved_bonus_marker_supply(),
        )
        if self.use_emperors_favour and self.emperor_tile_mode == "manual":
            game.tile_pool = list(self.emperor_tiles)

        game.configuration = self
        for player, control in zip(game.players, self.player_controls):
            player.control = control
            player.ai_top_k = None if control.is_human else self.top_k_for(control)
            if not control.is_human:
                player.hansa_nn = self._load_ai_model(player.order)
        return game

    @staticmethod
    def _load_ai_model(player_order: int):
        # AI models are optional; human-only games must not import PyTorch.
        from ai.ai_model import HansaNN

        model_file = f"hansa_nn_model{player_order}.pth"
        return HansaNN(INPUT_SIZE, OUTPUT_SIZE, model_file=model_file)


def choose_ranked_ai_action(
    ranked_actions: Sequence[tuple[int, float]],
    control: PlayerControl,
    rng: random.Random,
    thresholds: dict[PlayerControl, int] | None = None,
) -> int:
    """Choose from the configured top-k actions, weighted by model score."""
    control = PlayerControl(control)
    if control.is_human:
        raise ValueError("Cannot choose an AI action for a Human player")
    if not ranked_actions:
        raise ValueError("At least one ranked legal action is required")

    top_k = (thresholds or AI_DIFFICULTY_TOP_K)[control]
    candidates = sorted(ranked_actions, key=lambda item: item[1], reverse=True)[:top_k]
    if control is PlayerControl.MAGNUS:
        return candidates[0][0]

    maximum = max(score for _action, score in candidates)
    weights = [math.exp(score - maximum) for _action, score in candidates]
    return rng.choices([action for action, _score in candidates], weights=weights, k=1)[0]
