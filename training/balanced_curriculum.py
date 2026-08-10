"""Experimental curriculum runner with balanced configurations and composable focuses."""

from __future__ import annotations

import random

from game.game_config import GameConfiguration, human_players
from game.persistence import save_game
from training.balanced_state_generator import (
    BalancedGenerationRequest,
    EndingCondition,
    RegionalFocus,
    generate_balanced_state,
    save_balanced_state,
)
from training.curriculum import CurriculumRunner, StateDescriptor


CONFIGURATIONS = tuple(
    (map_num, player_count) for map_num in (1, 2, 3) for player_count in (3, 4, 5)
)
ENDING_CONDITIONS = tuple(EndingCondition)


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
        east_west = rng.random() < 0.25
        regional_focus = None
        if map_num == 3 and rng.random() < 0.25:
            choices = [RegionalFocus.WALES]
            if player_count > 3:
                choices.extend((RegionalFocus.SCOTLAND, RegionalFocus.ISLE_OF_MAN))
            if ending_condition is EndingCondition.NEAR_COMPLETED_CITIES and east_west:
                choices = [choice for choice in choices if choice is not RegionalFocus.ISLE_OF_MAN]
            regional_focus = rng.choice(choices)
        immediate_finish = self.training_generation_number % 10 == 0
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=seed,
                map_num=map_num,
                player_count=player_count,
                ending_condition=ending_condition,
                east_west=east_west,
                regional_focus=regional_focus,
                use_mission_cards=use_missions,
                use_emperors_favour=use_favour,
                use_promo_markers=use_promos,
                immediate_finish=immediate_finish,
            )
        )
        path, metadata_path = save_balanced_state(generated, directory)
        focuses = []
        if east_west:
            focuses.append("east_west")
        if regional_focus is not None:
            focuses.append(regional_focus.value)
        scenario = "+".join((ending_condition.value, *focuses))
        descriptor = StateDescriptor(
            path,
            metadata_path,
            map_num,
            player_count,
            seed,
            scenario,
            "immediate_finish" if immediate_finish else "one_round_before",
        )
        self._latest_descriptor = descriptor
        if is_training_generation:
            self.training_generation_number += 1
        return descriptor
