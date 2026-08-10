import random
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from training.balanced_curriculum import BalancedCurriculumRunner, CONFIGURATIONS
from training.balanced_state_generator import (
    BalancedGenerationRequest,
    EndingCondition,
    RegionalFocus,
    generate_balanced_state,
)
from training.curriculum import CurriculumConfig


class BalancedCurriculumTests(unittest.TestCase):
    @staticmethod
    def _runner(training_generation_number=0):
        runner = object.__new__(BalancedCurriculumRunner)
        runner.config = SimpleNamespace(seed=124)
        runner.training_generation_number = training_generation_number
        return runner

    def test_each_nine_game_block_contains_every_map_player_combination(self):
        runner = self._runner()

        for block in range(3):
            observed = set()
            for offset in range(9):
                runner.training_generation_number = block * 9 + offset
                observed.add(runner._configuration_for_game())
            self.assertEqual(observed, set(CONFIGURATIONS))

    def test_three_player_regional_options_exclude_scotland_and_isle_of_man(self):
        request = BalancedGenerationRequest(
            seed=1,
            map_num=3,
            player_count=3,
            ending_condition=EndingCondition.NEAR_SCORE,
            regional_focus=RegionalFocus.SCOTLAND,
        )
        with self.assertRaisesRegex(ValueError, "Three-player"):
            generate_balanced_state(request, max_attempts=1)

    def test_east_west_and_isle_of_man_can_be_generated_together(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=124,
                map_num=3,
                player_count=4,
                ending_condition=EndingCondition.NEAR_SCORE,
                east_west=True,
                regional_focus=RegionalFocus.ISLE_OF_MAN,
                use_emperors_favour=True,
                use_promo_markers=True,
            )
        )

        self.assertEqual(generated.game.map_num, 3)
        self.assertTrue(any(value.startswith("east_west:") for value in generated.focus_variants))
        self.assertTrue(
            any(value.startswith("britannia_isle_of_man:") for value in generated.focus_variants)
        )

    def test_configuration_shuffle_is_reproducible(self):
        expected = list(CONFIGURATIONS)
        random.Random(124).shuffle(expected)
        runner = self._runner()
        self.assertEqual(runner._configuration_for_game(), expected[0])

    def test_evaluation_generation_does_not_advance_training_sequence(self):
        runner = self._runner(7)
        runner._latest_descriptor = None
        stage = SimpleNamespace(full_game=True)

        with tempfile.TemporaryDirectory() as directory:
            runner._generate_state(stage, 55, Path(directory), map_num=2, player_count=3)

        self.assertEqual(runner.training_generation_number, 7)

    def test_training_generation_counter_is_saved_in_curriculum_state(self):
        runner = self._runner(12)
        runner.stage_index = 0
        runner.batch_number = 0
        runner.game_number = 99
        runner.rolling_losses = []
        runner.config = CurriculumConfig()

        state = runner._curriculum_state()

        self.assertEqual(state["training_generation_number"], 12)


if __name__ == "__main__":
    unittest.main()
