import unittest

from training.results_schema import interpret_results_row


class ResultsSchemaTests(unittest.TestCase):
    def test_current_and_historical_training_rows(self):
        cases = (
            ({"run": "training_fresh", "scenario": ""}, "training_fresh", ""),
            (
                {"run_type": "training", "training_stage": "mid"},
                "training_mid",
                "",
            ),
            (
                {"run_type": "training", "curriculum_stage": "late+network_keys"},
                "training_late",
                "network_keys",
            ),
            (
                {"run_type": "training", "curriculum_stage": "near_score"},
                "training_end",
                "near_score",
            ),
        )
        for row, expected_run, expected_scenario in cases:
            with self.subTest(row=row):
                interpreted = interpret_results_row(row)
                self.assertEqual(interpreted.run, expected_run)
                self.assertEqual(interpreted.scenario, expected_scenario)
                self.assertEqual(interpreted.run_type, "training")

    def test_zero_epsilon_rows(self):
        for row in (
            {"run": "training_fresh_zero_epsilon"},
            {"training_stage": "fresh", "run_mode": "training_zero_epsilon"},
            {"training_stage": "fresh", "training_exploration_mode": "zero_epsilon"},
        ):
            with self.subTest(row=row):
                self.assertEqual(
                    interpret_results_row(row).run,
                    "training_fresh_zero_epsilon",
                )

    def test_legacy_timeout_and_decision_aliases(self):
        interpreted = interpret_results_row(
            {
                "run_type": "training_timeout",
                "training_stage": "end",
                "trajectory_decision_count": "2048",
                "sampled_training_decision_count": "1024",
            }
        )

        self.assertEqual(interpreted.run, "training_end")
        self.assertEqual(interpreted.completion_reason, "action_limit")
        self.assertEqual(interpreted.total_training_decisions, "2048")
        self.assertEqual(interpreted.sampled_training_decisions, "1024")

    def test_current_decision_fields_take_precedence_over_legacy_aliases(self):
        interpreted = interpret_results_row(
            {
                "total_training_decisions": "4096",
                "sampled_training_decisions": "3500",
                "trajectory_decision_count": "2048",
                "sampled_training_decision_count": "1024",
            }
        )

        self.assertEqual(interpreted.total_training_decisions, "4096")
        self.assertEqual(interpreted.sampled_training_decisions, "3500")

    def test_evaluation_sets(self):
        cases = (
            ({"run": "evaluation_early"}, "early"),
            (
                {"run_type": "evaluation", "evaluation_set": "mid_late_end"},
                "mid_late_end",
            ),
            ({"run_type": "evaluation", "evaluation_set": "fresh"}, "fresh"),
            ({"run_type": "evaluation"}, "mid_late_end"),
        )
        for row, expected_set in cases:
            with self.subTest(row=row):
                interpreted = interpret_results_row(row)
                self.assertEqual(interpreted.run, f"evaluation_{expected_set}")
                self.assertEqual(interpreted.evaluation_set, expected_set)
                self.assertEqual(interpreted.run_type, "evaluation")

    def test_missing_optional_fields_have_existing_defaults(self):
        interpreted = interpret_results_row({})

        self.assertEqual(interpreted.run, "training_end")
        self.assertEqual(interpreted.scenario, "")
        self.assertIsNone(interpreted.evaluation_set)
        self.assertEqual(interpreted.completion_reason, "")
        self.assertEqual(interpreted.total_training_decisions, "")
        self.assertEqual(interpreted.sampled_training_decisions, "")


if __name__ == "__main__":
    unittest.main()
