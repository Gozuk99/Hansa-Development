"""Curriculum runner with balanced configurations and composable focuses."""

from __future__ import annotations

import random

from game.game_config import GameConfiguration, human_players
from game.persistence import save_game
from training.balanced_state_generator import (
    BalancedGenerationRequest,
    EAST_WEST_FOCUSES,
    EndingCondition,
    RegionalFocus,
    StartingPosition,
    StrategicFocus,
    generate_balanced_state,
    save_balanced_state,
)
from training.curriculum import CurriculumRunner, StateDescriptor


CONFIGURATIONS = tuple(
    (map_num, player_count) for map_num in (1, 2, 3) for player_count in (3, 4, 5)
)
ENDING_CONDITIONS = tuple(EndingCondition)
CLOSE_FINISH_INTERVAL = 20
NEAR_SCORE_RANGES = ((12, 14), (14, 16), (16, 18), (12, 16))


def _select_focus(rng, map_num, player_count, ending_condition):
    focus_roll = rng.random()
    focus = StrategicFocus.NONE
    if focus_roll < 0.10:
        focus = StrategicFocus.SPECIAL_PRESTIGE
    elif focus_roll < 0.25:
        focus = StrategicFocus.NETWORK_KEYS
    elif focus_roll < 0.50:
        dual = rng.random() < 0.5
        blocked = rng.random() < 0.5
        focus = {
            (False, False): StrategicFocus.EAST_WEST,
            (True, False): StrategicFocus.DUAL_EAST_WEST,
            (False, True): StrategicFocus.BLOCKED_EAST_WEST,
            (True, True): StrategicFocus.BLOCKED_DUAL_EAST_WEST,
        }[dual, blocked]
        if ending_condition is EndingCondition.NEAR_COMPLETED_CITIES:
            focus = {
                StrategicFocus.DUAL_EAST_WEST: StrategicFocus.EAST_WEST,
                StrategicFocus.BLOCKED_DUAL_EAST_WEST: StrategicFocus.BLOCKED_EAST_WEST,
            }.get(focus, focus)
        elif (
            ending_condition is EndingCondition.NEAR_BONUS_MARKERS
            and focus is StrategicFocus.BLOCKED_DUAL_EAST_WEST
        ):
            focus = StrategicFocus.BLOCKED_EAST_WEST
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

    def _score_range_for_game(self):
        block, index = divmod(self.training_generation_number, len(NEAR_SCORE_RANGES))
        score_ranges = list(NEAR_SCORE_RANGES)
        random.Random(self.config.seed + 500_009 + block * 1_000_003).shuffle(score_ranges)
        return score_ranges[index]

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
        use_missions = map_num == 1 and rng.choice((False, True))
        use_favour = rng.choice((False, True))
        use_promos = rng.choice((False, True))
        if stage.full_game:
            configuration = GameConfiguration(
                map_num=map_num,
                player_count=player_count,
                player_controls=human_players(player_count),
                use_mission_cards=use_missions,
                use_emperors_favour=use_favour,
                use_promo_markers=use_promos,
                seed=seed,
            )
            path = save_game(configuration.create_game(), directory / f"full-{seed}.hansa")
            descriptor = StateDescriptor(path, None, map_num, player_count, seed, "full_game")
            self._latest_descriptor = descriptor
            if is_training_generation:
                self.training_generation_number += 1
            return descriptor

        ending_condition = ENDING_CONDITIONS[
            self.training_generation_number % len(ENDING_CONDITIONS)
        ]
        focus, regional = _select_focus(rng, map_num, player_count, ending_condition)
        two_decision_finish = self.training_generation_number % CLOSE_FINISH_INTERVAL == 0
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=seed,
                map_num=map_num,
                player_count=player_count,
                ending_condition=ending_condition,
                score_range=self._score_range_for_game(),
                strategic_focus=focus,
                regional_focus=regional,
                use_mission_cards=use_missions,
                use_emperors_favour=use_favour,
                use_promo_markers=use_promos,
                bonus_markers_remaining=0 if two_decision_finish else 1,
                completed_cities_below_limit=1 if two_decision_finish else 2,
                starting_position=(
                    StartingPosition.TWO_DECISIONS_BEFORE
                    if two_decision_finish
                    else StartingPosition.ONE_ROUND_BEFORE
                ),
                prepared_routes_one_short=True,
            )
        )
        path, metadata_path = save_balanced_state(generated, directory)
        scenario = "+".join((ending_condition.value, *_focus_labels(focus, regional)))
        descriptor = StateDescriptor(
            path,
            metadata_path,
            map_num,
            player_count,
            seed,
            scenario,
            "two_decisions_before" if two_decision_finish else "one_round_before",
        )
        self._latest_descriptor = descriptor
        if is_training_generation:
            self.training_generation_number += 1
        return descriptor
