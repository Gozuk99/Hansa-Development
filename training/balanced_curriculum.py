"""Curriculum runner with balanced configurations and composable focuses."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.game_config import GameConfiguration, human_players
from game.persistence import save_game
from training.balanced_state_generator import (
    BalancedGenerationRequest,
    BonusMarkerSetup,
    EAST_WEST_FOCUSES,
    EndingCondition,
    RegionalFocus,
    StartingPosition,
    StrategicFocus,
    bonus_marker_configuration,
    generate_balanced_state,
    save_balanced_state,
)
from training.curriculum import CurriculumRunner, StateDescriptor


CONFIGURATIONS = tuple(
    (map_num, player_count) for map_num in (1, 2, 3) for player_count in (3, 4, 5)
)
ENDING_CONDITIONS = tuple(EndingCondition)
EARLY_ROUTE_SCAFFOLD_RATE = 0.70
EARLY_ROUTE_SCAFFOLD_SEED_OFFSET = 71_003


@dataclass(frozen=True)
class MaturityProfile:
    name: str
    weight: int
    score_range: tuple[int, int] | None
    development_range: tuple[int, int] | None
    bonus_markers_remaining: int | None
    completed_cities_below_limit: int | None
    starting_position: StartingPosition | None
    round_range: tuple[int, int]

    @property
    def fresh(self):
        return self.score_range is None

    @property
    def starting_position_label(self):
        if self.name == "end":
            return self.starting_position.value
        return ""


MATURITY_PROFILES = (
    # Fresh remains excluded until early-game performance is stable.
    # MaturityProfile("fresh", 1, None, None, None, None, None, (1, 1)),
    MaturityProfile("early", 24, (0, 5), (2, 4), 9, 7, StartingPosition.ONE_ROUND_BEFORE, (2, 5)),
    MaturityProfile(
        "early_mixed",
        48,
        (2, 6),
        (3, 5),
        7,
        6,
        StartingPosition.ONE_ROUND_BEFORE,
        (3, 6),
    ),
    MaturityProfile("mid", 9, (6, 11), (5, 7), 3, 5, StartingPosition.ONE_ROUND_BEFORE, (6, 10)),
    MaturityProfile("late", 9, (12, 15), (7, 9), 2, 3, StartingPosition.ONE_ROUND_BEFORE, (11, 15)),
    MaturityProfile(
        "end", 6, (16, 17), (9, 11), 1, 2, StartingPosition.TWO_DECISIONS_BEFORE, (16, 20)
    ),
    MaturityProfile(
        "mixed",
        64,
        (0, 12),
        (2, 8),
        5,
        5,
        StartingPosition.ONE_ROUND_BEFORE,
        (6, 12),
    ),
)
MATURITY_CYCLE = tuple(profile for profile in MATURITY_PROFILES for _ in range(profile.weight))


def _select_focus(rng, map_num, player_count, ending_condition):
    focus_roll = rng.random()
    focus = StrategicFocus.NONE
    if focus_roll < 0.10:
        focus = StrategicFocus.SPECIAL_PRESTIGE
    elif focus_roll < 0.25:
        focus = StrategicFocus.NETWORK_KEYS
    elif focus_roll < 0.50:
        focus = StrategicFocus.DUAL_EAST_WEST if rng.random() < 0.5 else StrategicFocus.EAST_WEST
        if ending_condition is EndingCondition.NEAR_COMPLETED_CITIES:
            focus = StrategicFocus.EAST_WEST
    regional = None
    if (
        focus
        not in (
            StrategicFocus.SPECIAL_PRESTIGE,
            StrategicFocus.NETWORK_KEYS,
        )
        and map_num == 3
        and rng.random() < 0.25
    ):
        choices = [RegionalFocus.WALES]
        if player_count > 3:
            choices.extend((RegionalFocus.SCOTLAND, RegionalFocus.ISLE_OF_MAN))
        if ending_condition is EndingCondition.NEAR_COMPLETED_CITIES and focus in EAST_WEST_FOCUSES:
            choices = [choice for choice in choices if choice is not RegionalFocus.ISLE_OF_MAN]
        elif ending_condition is EndingCondition.NEAR_BONUS_MARKERS:
            choices = [choice for choice in choices if choice is not RegionalFocus.ISLE_OF_MAN]
        regional = rng.choice(choices)
    return focus, regional


def _focus_labels(focus, regional):
    labels = {
        StrategicFocus.NONE: [],
        StrategicFocus.EAST_WEST: ["east_west"],
        StrategicFocus.DUAL_EAST_WEST: ["dual_east_west"],
        StrategicFocus.BLOCKED_EAST_WEST: ["east_west", "blocked"],
        StrategicFocus.BLOCKED_DUAL_EAST_WEST: ["dual_east_west", "blocked"],
        StrategicFocus.SPECIAL_PRESTIGE: ["special_prestige"],
        StrategicFocus.NETWORK_KEYS: ["network_keys"],
    }[focus]
    if regional is not None:
        labels.append(regional.value)
    return tuple(labels)


def _scenario_condition_label(maturity, ending_condition):
    if maturity.name in {"early", "early_mixed", "mixed"}:
        return None
    if maturity.name == "mid":
        return {
            EndingCondition.NEAR_SCORE: "score_focus",
            EndingCondition.NEAR_BONUS_MARKERS: "bonus_marker_focus",
            EndingCondition.NEAR_COMPLETED_CITIES: "completed_city_focus",
        }[ending_condition]
    return ending_condition.value


def _select_optional_modules(rng, map_num):
    use_missions = map_num == 1 and rng.random() < 0.40
    use_favour = rng.choice((False, True))
    marker_roll = rng.random()
    if marker_roll < 0.50:
        marker_setup = BonusMarkerSetup.DEFAULT
    elif marker_roll < 0.75:
        marker_setup = BonusMarkerSetup.ALL_PROMOS
    else:
        marker_setup = BonusMarkerSetup.MIXED
    return use_missions, use_favour, marker_setup


def _uses_early_route_scaffold(seed):
    """Choose the early variant without consuming the normal generator RNG stream."""
    return random.Random(seed + EARLY_ROUTE_SCAFFOLD_SEED_OFFSET).random() < (
        EARLY_ROUTE_SCAFFOLD_RATE
    )


class BalancedCurriculumRunner(CurriculumRunner):
    """Keep every nine generated training positions balanced by map and player count."""

    def __init__(self, trainer, *args, **kwargs):
        super().__init__(trainer, *args, **kwargs)
        saved = trainer.curriculum_state or {}
        self.training_generation_number = saved.get("training_generation_number", 0)

    def _curriculum_state(self):
        state = super()._curriculum_state()
        state["training_generation_number"] = self.training_generation_number
        return state

    def _configuration_for_game(self):
        block, index = divmod(self.training_generation_number, len(CONFIGURATIONS))
        configurations = list(CONFIGURATIONS)
        random.Random(self.config.seed + block * 1_000_003).shuffle(configurations)
        return configurations[index]

    def _maturity_for_game(self, generation_number=None):
        generation_number = (
            self.training_generation_number if generation_number is None else generation_number
        )
        block, index = divmod(generation_number, len(MATURITY_CYCLE))
        profiles = list(MATURITY_CYCLE)
        random.Random(self.config.seed + 500_009 + block * 1_000_003).shuffle(profiles)
        return profiles[index]

    @staticmethod
    def _stage_label(_stage):
        return "early_early_mixed_mid_late_end_mixed_game"

    @staticmethod
    def _stage_action_limit(_stage):
        return 10_000

    def _training_action_limit(self, stage, descriptor):
        maturity = (descriptor.scenario or stage.name).partition("+")[0]
        return 15_000 if maturity in {"early", "early_mixed"} else self._stage_action_limit(stage)

    def _generate_state(self, stage, seed, directory, *, map_num=None, player_count=None):
        rng = random.Random(seed)
        is_training_generation = map_num is None and player_count is None
        if is_training_generation:
            map_num, player_count = self._configuration_for_game()
        else:
            map_num = rng.choice((1, 2, 3)) if map_num is None else map_num
            player_count = rng.choice((3, 4, 5)) if player_count is None else player_count

        pending = StateDescriptor(
            directory / f"pending-{seed}.hansa", None, map_num, player_count, seed
        )
        self._latest_descriptor = pending
        use_missions, use_favour, marker_setup = _select_optional_modules(rng, map_num)
        maturity = self._maturity_for_game(
            self.training_generation_number if is_training_generation else seed
        )
        early_route_scaffold = (
            is_training_generation and maturity.name == "early" and _uses_early_route_scaffold(seed)
        )
        if maturity.fresh:
            use_promos, promo_mode, promo_markers = bonus_marker_configuration(marker_setup, seed)
            configuration = GameConfiguration(
                map_num=map_num,
                player_count=player_count,
                player_controls=human_players(player_count),
                use_mission_cards=use_missions,
                use_emperors_favour=use_favour,
                use_promo_markers=use_promos,
                promo_marker_mode=promo_mode,
                promo_markers=promo_markers,
                seed=seed,
            )
            path = save_game(configuration.create_game(), directory / f"full-{seed}.hansa")
            descriptor = StateDescriptor(path, None, map_num, player_count, seed, "fresh", "fresh")
            self._latest_descriptor = descriptor
            if is_training_generation:
                self.training_generation_number += 1
            return descriptor

        ending_condition = ENDING_CONDITIONS[
            self.training_generation_number % len(ENDING_CONDITIONS)
        ]
        focus, regional = _select_focus(rng, map_num, player_count, ending_condition)
        if maturity.name in {"early", "early_mixed", "mixed"}:
            focus, regional = StrategicFocus.NONE, None
        elif maturity.name == "mid":
            focus = {
                StrategicFocus.DUAL_EAST_WEST: StrategicFocus.EAST_WEST,
                StrategicFocus.BLOCKED_DUAL_EAST_WEST: StrategicFocus.BLOCKED_EAST_WEST,
                StrategicFocus.NETWORK_KEYS: StrategicFocus.NONE,
            }.get(focus, focus)
            regional = None
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=seed,
                map_num=map_num,
                player_count=player_count,
                ending_condition=ending_condition,
                score_range=maturity.score_range,
                strategic_focus=focus,
                regional_focus=regional,
                use_mission_cards=use_missions,
                use_emperors_favour=use_favour,
                use_promo_markers=marker_setup is not BonusMarkerSetup.DEFAULT,
                bonus_marker_setup=marker_setup,
                bonus_markers_remaining=maturity.bonus_markers_remaining,
                completed_cities_below_limit=maturity.completed_cities_below_limit,
                starting_position=maturity.starting_position,
                prepared_routes_one_short=maturity.name not in {"early", "early_mixed", "mixed"},
                development_range=maturity.development_range,
                prepare_ending_condition=maturity.name not in {"early", "early_mixed", "mixed"},
                round_range=maturity.round_range,
                mixed_development=maturity.name == "mixed",
                early_mixed_development=maturity.name == "early_mixed",
                early_route_scaffold=early_route_scaffold,
            )
        )
        path, metadata_path = save_balanced_state(
            generated,
            directory,
            scenario_directory=(
                maturity.name if maturity.name in {"early", "early_mixed", "mixed"} else None
            ),
        )
        scenario = "+".join(
            filter(
                None,
                (
                    maturity.name,
                    _scenario_condition_label(maturity, ending_condition),
                    *_focus_labels(focus, regional),
                ),
            )
        )
        descriptor = StateDescriptor(
            path,
            metadata_path,
            map_num,
            player_count,
            seed,
            scenario,
            maturity.starting_position_label,
            None,
            getattr(generated, "starting_scores_by_seat", ()),
            getattr(generated, "starting_development_by_seat", ()),
            getattr(generated, "development_roles_by_seat", ()),
            early_route_scaffold if maturity.name == "early" else None,
            getattr(generated, "scaffolded_route_ids_by_seat", ()),
            getattr(generated, "scaffolded_route_lengths_by_seat", ()),
        )
        self._latest_descriptor = descriptor
        if is_training_generation:
            self.training_generation_number += 1
        return descriptor
