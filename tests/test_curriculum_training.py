import csv
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from ai.ai_model import MODEL_CHECKPOINT_FORMAT, MODEL_CHECKPOINT_VERSION
from training.curriculum import (
    CSV_FIELDS,
    DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS,
    DETAILED_PROFILING_FIELDS,
    SHADOW_FILTER_CSV_FIELDS,
    CurriculumConfig,
    CurriculumRunner,
    CurriculumStage,
    PromotionCriteria,
    StateDescriptor,
    ZERO_EPSILON_EXPLORATION_MODE,
    _format_game_numbers,
    csv_fields,
)
from training.self_play import (
    ActionLimitExceeded,
    IncompleteGameError,
    TrainingConfig,
    TRAINING_CHECKPOINT_FORMAT,
    TRAINING_CHECKPOINT_VERSION,
    TrainingProgress,
)
from training.targeted_state_generator import StateGenerationError
from tests.action_helpers import evaluation_state
from tools.run_curriculum_training import parse_args as parse_curriculum_args


VALIDATION_STATE = evaluation_state("map1_3p_near_score")


def completed_trajectory():
    decision = SimpleNamespace(policy_tier=1)
    return SimpleNamespace(
        decisions=(decision,),
        winner_indices=(0,),
        seat_tiers=(1, 3, 5),
        final_scores=(30, 20, 10),
        action_trace=(285,),
        play_seconds=1.0,
        inference_seconds=0.2,
        scoring_seconds=0.3,
        execution_seconds=0.1,
        validation_seconds=0.05,
        observation_seconds=0.1,
        legality_seconds=0.05,
        selection_seconds=0.05,
        context_seconds=0.01,
        reward_seconds=0.05,
        move_action_count=2,
        spent_action_count=5,
        move_ratio=0.4,
        pointless_move_workflows=1,
        repeated_move_penalties=2,
        all_move_turn_penalties=1,
        moves_creating_claimable_route=2,
        move_claim_conversions=1,
        move_claim_conversion_rate=0.5,
    )


def deadlocked_trajectory():
    trajectory = completed_trajectory()
    trajectory.completion_reason = "no_replacement_route"
    trajectory.winner_indices = ()
    return trajectory


def action_limited_trajectory():
    trajectory = completed_trajectory()
    trajectory.completion_reason = "action_limit"
    trajectory.winner_indices = ()
    return trajectory


class FakeTrainer:
    def __init__(self, *, detailed_profiling=False):
        self.model = FakeModel()
        self.config = TrainingConfig(
            max_actions=10,
            detailed_profiling=detailed_profiling,
        )
        self.progress = TrainingProgress()
        self.progress.last_loss = None
        self.progress.mean_loss = None
        self.curriculum_state = None
        self.rng = random.Random(124)
        self.collect_calls = 0
        self.update_calls = 0
        self.saved_curriculum_state = None
        self.measured_losses = 0
        self.checkpoint_calls = 0

    def collect_game(self, _path, *, failure_callback=None, evaluation=False, **_kwargs):
        self.collect_calls += 1
        self.evaluation_calls = getattr(self, "evaluation_calls", []) + [evaluation]
        if evaluation:
            self.evaluation_rotations = getattr(self, "evaluation_rotations", []) + [
                _kwargs.get("evaluation_tier_rotation")
            ]
            self.capture_action_limits = getattr(self, "capture_action_limits", []) + [
                _kwargs.get("capture_action_limit")
            ]
        self.rng.random()
        self.progress.completed_games += 1
        self.zero_epsilon_calls = getattr(self, "zero_epsilon_calls", []) + [
            _kwargs.get("zero_epsilon", False)
        ]
        trajectory = completed_trajectory()
        trajectory.training_exploration_mode = (
            ZERO_EPSILON_EXPLORATION_MODE
            if _kwargs.get("zero_epsilon", False) and not evaluation
            else "normal"
        )
        return trajectory

    def update_model(self, trajectories, *, curriculum_maturities=None):
        self.update_calls += 1
        self.curriculum_maturities = getattr(self, "curriculum_maturities", []) + list(
            curriculum_maturities or ()
        )
        decision_count = len(trajectories[0].decisions)
        self.last_training_sample_coverage = (
            SimpleNamespace(
                total_decisions=decision_count,
                sampled_decisions=decision_count,
                sampled_fraction=1.0,
            ),
        )
        self.progress.training_updates += 1
        self.progress.last_loss = 12.5
        self.progress.mean_loss = 12.5
        return 12.5

    def trajectory_loss(self, _trajectory):
        self.measured_losses += 1
        return float(self.measured_losses * 10)

    def save_checkpoint(self, path, _states, *, curriculum_state=None):
        self.checkpoint_calls += 1
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("checkpoint", encoding="utf-8")
        self.curriculum_state = curriculum_state
        self.saved_curriculum_state = curriculum_state


class FakeModel:
    def __init__(self):
        self.save_calls = 0

    def save_model(self, path):
        self.save_calls += 1
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("playable model", encoding="utf-8")
        return target


class TestRunner(CurriculumRunner):
    __test__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_seeds = []
        self.generated_configurations = []

    def _generate_state(self, _stage, seed, _directory, *, map_num=None, player_count=None):
        self.generated_seeds.append(seed)
        self.generated_configurations.append((map_num, player_count))
        return StateDescriptor(
            VALIDATION_STATE,
            None,
            2 if map_num is None else map_num,
            5 if player_count is None else player_count,
            seed,
        )


class CurriculumTrainingTests(unittest.TestCase):
    def test_default_csv_schema_is_compact_and_keeps_raw_counters(self):
        for field in (
            "early_training_action_limit_outcome",
            "sampled_training_fraction",
            *(f"sampled_octile_{index}" for index in range(1, 9)),
            "move_ratio",
            "move_claim_conversion_rate",
            "rolling_mean_loss",
            "starting_position",
            "training_exploration_mode",
            "evaluation_set",
        ):
            self.assertNotIn(field, CSV_FIELDS)
        for field in (
            "curriculum_stage",
            "training_stage",
            "run_type",
            "run_mode",
            "trajectory_decision_count",
            "sampled_training_decision_count",
        ):
            self.assertNotIn(field, CSV_FIELDS)
        self.assertIn("run", CSV_FIELDS)
        self.assertIn("scenario", CSV_FIELDS)
        for field in (
            "total_training_decisions",
            "sampled_training_decisions",
            "move_action_count",
            "spent_action_count",
            "moves_creating_claimable_route",
            "move_claim_conversions",
            "shadow_filter_selected_count",
            "shadow_filter_epsilon_selected_count",
            "generation_seconds",
            "play_seconds",
            "learning_seconds",
            "q_loss",
            "policy_loss",
            "total_loss",
            "policy_q_top1_agreement",
            "policy_top1_q_rank",
            "policy_entropy",
            "policy_top2_mass",
            "policy_top5_mass",
            "policy_top10_mass",
            "policy_trunk_gradient_scale",
        ):
            self.assertIn(field, CSV_FIELDS)
        for field in DETAILED_PROFILING_FIELDS:
            self.assertNotIn(field, CSV_FIELDS)
            self.assertIn(field, csv_fields(detailed_profiling=True))
        for field in (
            "mission_cards_enabled",
            "emperors_favour_enabled",
            "emperors_favour_tiles",
            "bonus_marker_supply_mode",
            "bonus_marker_draw_supply",
            "starting_bonus_marker_routes",
        ):
            self.assertIn(field, CSV_FIELDS)

    def test_shadow_filter_audit_csv_contains_only_flagged_training_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = self.runner(root)
            record = SimpleNamespace(
                decision_index=6,
                action_index=321,
                action_type="PostInteraction",
                semantic_action_indices=(321, 322, 323),
                semantic_q_rank=24,
                q_value=-4.5,
                q_gap_from_best=7.25,
                semantic_policy_rank=13,
                policy_probability=0.01,
                immediate_reward=-5.0,
                local_training_target=-1000.0,
                local_training_adjustment=-200.0,
                reward_to_go=-900.0,
                final_training_target=-1200.0,
                receives_terminal_credit=False,
                terminal_credit_value=0.0,
                acting_player_index=1,
                acting_player_final_score=12,
                acting_player_won=False,
                policy_tier=2,
                used_epsilon=True,
            )
            trajectory = completed_trajectory()
            trajectory.shadow_filter_records = (record,)
            trajectory.shadow_filter_selected_count = 1
            trajectory.shadow_filter_epsilon_selected_count = 1
            trajectory.training_exploration_mode = "normal"
            trajectory.completion_reason = "20_points"
            descriptor = StateDescriptor(
                VALIDATION_STATE,
                None,
                2,
                5,
                123,
                scenario="late+near_score",
            )

            rows = runner._shadow_filter_rows(
                trajectory,
                descriptor,
                runner.config.stages[0],
                retry_count=2,
                game_number=17,
            )
            runner._append_shadow_filter_csv(rows)

            with (root / "filter_shadow.csv").open(newline="", encoding="utf-8") as source:
                written = list(csv.DictReader(source))
            self.assertEqual(tuple(written[0]), SHADOW_FILTER_CSV_FIELDS)
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["game#"], "17")
            self.assertEqual(written[0]["run"], "training_late")
            self.assertEqual(written[0]["game_maturity"], "late")
            self.assertEqual(written[0]["decision_number"], "7")
            self.assertEqual(json.loads(written[0]["semantic_action_indices"]), [321, 322, 323])
            self.assertEqual(written[0]["would_filter"], "True")
            self.assertEqual(written[0]["final_training_target"], "-1200.0")
            self.assertEqual(written[0]["used_epsilon"], "True")

    def test_zero_epsilon_training_schedule_is_configurable_and_validated(self):
        self.assertEqual(
            dict(CurriculumConfig().zero_epsilon_training_fractions),
            {"fresh": 0.05, "early": 0.05, "mid": 0.05, "late": 0.50, "end": 1.00},
        )
        self.assertEqual(
            parse_curriculum_args(
                ["--zero-epsilon-training-percentage", "12.5"]
            ).zero_epsilon_training_percentage,
            12.5,
        )
        maturity_args = parse_curriculum_args(
            [
                "--fresh-zero-epsilon-training-percentage",
                "6",
                "--early-zero-epsilon-training-percentage",
                "7",
                "--mid-zero-epsilon-training-percentage",
                "8",
                "--late-zero-epsilon-training-percentage",
                "40",
                "--end-zero-epsilon-training-percentage",
                "100",
            ]
        )
        self.assertEqual(
            (
                maturity_args.fresh_zero_epsilon_training_percentage,
                maturity_args.early_zero_epsilon_training_percentage,
                maturity_args.mid_zero_epsilon_training_percentage,
                maturity_args.late_zero_epsilon_training_percentage,
                maturity_args.end_zero_epsilon_training_percentage,
            ),
            (6.0, 7.0, 8.0, 40.0, 100.0),
        )
        with self.assertRaises(ValueError):
            CurriculumConfig(
                zero_epsilon_training_fractions=(("fresh", -0.01),)
                + DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS[1:]
            )
        with self.assertRaises(ValueError):
            CurriculumConfig(
                zero_epsilon_training_fractions=(("fresh", 1.01),)
                + DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS[1:]
            )

    def test_detailed_profiling_cli_is_disabled_by_default_and_opt_in(self):
        self.assertFalse(parse_curriculum_args([]).detailed_profiling)
        self.assertTrue(parse_curriculum_args(["--detailed-profiling"]).detailed_profiling)

    def test_shadow_filter_audit_cli_is_disabled_by_default_and_opt_in(self):
        self.assertFalse(parse_curriculum_args([]).shadow_filter_audit)
        self.assertTrue(parse_curriculum_args(["--shadow-filter-audit"]).shadow_filter_audit)

    def test_saved_game_numbers_are_compact_but_preserve_gaps(self):
        self.assertEqual(_format_game_numbers([16, 17, 18, 19, 20]), "16-20")
        self.assertEqual(_format_game_numbers([2, 3, 5]), "2, 3, and 5")

    def test_evaluation_tier_rotation_advances_each_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = FakeTrainer()
            runner = self.runner(Path(directory), trainer, iterations=2)

            runner.run()

            self.assertEqual(trainer.evaluation_rotations, [0, 1])

    def test_fresh_fixed_evaluation_is_distinct_and_never_updates_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            manifest = [
                {
                    "save_file": "standard.hansa",
                    "metadata_file": "standard.json",
                    "map_num": 1,
                    "player_count": 3,
                    "seed": 100,
                    "scenario": "near_score",
                    "suite_version": 5,
                    "evaluation_set": "mid_late_end",
                },
                {
                    "save_file": "fresh.hansa",
                    "metadata_file": "fresh.json",
                    "map_num": 2,
                    "player_count": 5,
                    "seed": 200,
                    "scenario": "fresh_game",
                    "suite_version": 10,
                    "evaluation_set": "fresh",
                },
                {
                    "save_file": "removed-mixed.hansa",
                    "metadata_file": "removed-mixed.json",
                    "map_num": 3,
                    "player_count": 4,
                    "seed": 300,
                    "scenario": "mixed_development",
                    "suite_version": 8,
                    "evaluation_set": "mixed_development",
                },
            ]
            (evaluation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            trainer = FakeTrainer()
            runner = self.runner(root, trainer)

            runner.run()

            self.assertEqual(trainer.update_calls, 1)
            self.assertEqual(trainer.evaluation_calls, [False, True, True])
            self.assertEqual(trainer.capture_action_limits, [False, True])
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            evaluation_rows = [row for row in rows if row["run"].startswith("evaluation_")]
            self.assertEqual(
                [row["run"] for row in evaluation_rows],
                ["evaluation_mid_late_end", "evaluation_fresh"],
            )
            self.assertEqual([row["evaluation_suite_size"] for row in evaluation_rows], ["1", "1"])

    def test_fresh_evaluation_timeout_is_recorded_without_training(self):
        class FreshTimeoutTrainer(FakeTrainer):
            def collect_game(self, path, *, failure_callback=None, evaluation=False, **kwargs):
                trajectory = super().collect_game(
                    path,
                    failure_callback=failure_callback,
                    evaluation=evaluation,
                    **kwargs,
                )
                return action_limited_trajectory() if evaluation else trajectory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            manifest = [
                {
                    "save_file": "fresh.hansa",
                    "metadata_file": "fresh.json",
                    "map_num": 3,
                    "player_count": 4,
                    "seed": 300,
                    "scenario": "fresh_game",
                    "suite_version": 10,
                    "evaluation_set": "fresh",
                }
            ]
            (evaluation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            trainer = FreshTimeoutTrainer()
            self.runner(root, trainer).run()

            self.assertEqual(trainer.update_calls, 1)
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            fresh = next(row for row in rows if row["run"] == "evaluation_fresh")
            self.assertEqual(fresh["completion_reason"], "action_limit")
            self.assertNotIn("timeout", fresh)

    def test_existing_windows_utf8_csv_header_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "results.csv"
            csv_path.write_text("game#,batch#\n7,3\n", encoding="utf-8-sig")

            runner = self.runner(root)

            self.assertEqual(runner.report_game_number, 7)
            self.assertEqual(runner.report_batch_number, 3)

    def test_default_near_end_action_limit_is_ten_thousand(self):
        self.assertEqual(CurriculumConfig().stages[0].action_limit, 10_000)
        self.assertEqual(CurriculumConfig().stages[1].action_limit, 10_000)
        self.assertEqual(CurriculumConfig().stages[0].score_range, (10, 17))

    def test_fresh_training_option_is_not_available(self):
        self.assertFalse(hasattr(parse_curriculum_args([]), "fresh"))
        with self.assertRaises(SystemExit):
            parse_curriculum_args(["--fresh"])

    def test_existing_csv_is_migrated_to_compact_schema_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "results.csv"
            runner = self.runner(root)
            csv_path.write_text(
                "game#,curriculum_stage,training_stage,run_type,run_mode,"
                "training_exploration_mode,evaluation_set,trajectory_decision_count,"
                "sampled_training_decision_count\n"
                "1,early+score_focus,early,training,training_zero_epsilon,"
                "zero_epsilon,,4096,1024\n"
                "2,near_score,end,training_timeout,training_normal,,,2048,1024\n",
                encoding="utf-8",
            )

            runner._append_csv(({"game#": 3, "run": "training_end"},))

            with csv_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["game#"] for row in rows], ["1", "2", "3"])
            self.assertIn("play_seconds", rows[0])
            self.assertEqual(rows[0]["run"], "training_early_zero_epsilon")
            self.assertEqual(rows[0]["scenario"], "score_focus")
            self.assertEqual(rows[0]["total_training_decisions"], "4096")
            self.assertEqual(rows[0]["sampled_training_decisions"], "1024")
            self.assertEqual(rows[1]["run"], "training_end")
            self.assertEqual(rows[1]["completion_reason"], "action_limit")
            self.assertNotIn("training_exploration_mode", rows[0])

    def test_csv_timing_values_are_rounded_to_hundredths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = self.runner(root)
            row = runner._trajectory_row(
                completed_trajectory(),
                StateDescriptor(VALIDATION_STATE, None, 2, 5, 1, "near_bonus_markers"),
                runner.config.stages[0],
                "training",
                0,
                None,
                1,
                generation_seconds=0.123456,
                learning_seconds=0.987654,
            )

            self.assertEqual(row["generation_seconds"], 0.12)
            self.assertEqual(row["learning_seconds"], 0.99)
            self.assertEqual(row["run"], "training_end")
            self.assertEqual(row["scenario"], "near_bonus_markers")
            self.assertEqual(row["move_action_count"], 2)
            self.assertEqual(row["spent_action_count"], 5)
            self.assertEqual(row["pointless_move_workflows"], 1)
            self.assertEqual(row["repeated_move_penalties"], 2)
            self.assertEqual(row["all_move_turn_penalties"], 1)
            self.assertEqual(row["moves_creating_claimable_route"], 2)
            self.assertEqual(row["move_claim_conversions"], 1)

    def test_run_labels_cover_training_evaluation_and_timeout_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(Path(directory))
            expected = {
                "fresh": "training_fresh",
                "early": "training_early",
                "mid+score_focus": "training_mid",
                "late+near_score": "training_late",
                "end+near_bonus_markers": "training_end",
            }
            for scenario, expected_run in expected.items():
                row = runner._trajectory_row(
                    completed_trajectory(),
                    StateDescriptor(VALIDATION_STATE, None, 1, 3, 1, scenario),
                    runner.config.stages[0],
                    "training",
                    0,
                    None,
                    1,
                )
                self.assertEqual(row["run"], expected_run)

            zero_epsilon = completed_trajectory()
            zero_epsilon.training_exploration_mode = ZERO_EPSILON_EXPLORATION_MODE
            fresh = runner._trajectory_row(
                zero_epsilon,
                StateDescriptor(VALIDATION_STATE, None, 1, 3, 1, "fresh"),
                runner.config.stages[0],
                "training",
                0,
                None,
                1,
            )
            self.assertEqual(fresh["run"], "training_fresh_zero_epsilon")

            for evaluation_set in ("fresh", "mid_late_end"):
                evaluation = runner._trajectory_row(
                    completed_trajectory(),
                    StateDescriptor(
                        VALIDATION_STATE,
                        None,
                        1,
                        3,
                        1,
                        "fresh" if evaluation_set == "fresh" else "near_score",
                        evaluation_set=evaluation_set,
                    ),
                    runner.config.stages[0],
                    "evaluation",
                    0,
                    None,
                    1,
                )
                self.assertEqual(evaluation["run"], f"evaluation_{evaluation_set}")

            timeout = runner._trajectory_row(
                action_limited_trajectory(),
                StateDescriptor(VALIDATION_STATE, None, 1, 3, 1, "early"),
                runner.config.stages[0],
                "training_timeout",
                0,
                None,
                1,
            )
            self.assertEqual(timeout["run"], "training_early")
            self.assertEqual(timeout["completion_reason"], "action_limit")

    def test_detailed_profiling_fields_are_written_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(
                Path(directory),
                FakeTrainer(detailed_profiling=True),
            )
            row = runner._trajectory_row(
                completed_trajectory(),
                StateDescriptor(VALIDATION_STATE, None, 2, 5, 1, "late"),
                runner.config.stages[0],
                "training",
                0,
                None,
                1,
            )

            self.assertEqual(row["inference_seconds"], 0.2)
            self.assertEqual(row["reward_seconds"], 0.05)
            self.assertEqual(
                tuple(field for field in row if field in DETAILED_PROFILING_FIELDS),
                DETAILED_PROFILING_FIELDS,
            )

    def test_loop_control_timing_is_printed_only_with_detailed_profiling(self):
        class ZeroSubtimingTrainer(FakeTrainer):
            def collect_game(self, *args, **kwargs):
                trajectory = super().collect_game(*args, **kwargs)
                trajectory.play_seconds = 2.0
                for field in DETAILED_PROFILING_FIELDS:
                    setattr(trajectory, field, 0.0)
                return trajectory

        messages_by_mode = {}
        for detailed_profiling in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                runner = self.runner(
                    Path(directory),
                    ZeroSubtimingTrainer(detailed_profiling=detailed_profiling),
                )
                messages = []
                runner.progress_callback = messages.append
                runner.run()
                messages_by_mode[detailed_profiling] = messages

        self.assertFalse(any("loop/control" in message for message in messages_by_mode[False]))
        self.assertTrue(any("loop/control 2.00s" in message for message in messages_by_mode[True]))

    def config(self, **changes):
        values = {
            "iterations": 1,
            "training_games_per_batch": 1,
            "evaluation_games_per_batch": 1,
            "update_batch_size": 1,
            "retry_limit": 1,
            "stages": (CurriculumStage("test", 10, (18, 19)),),
            "promotion": PromotionCriteria(require_tier_one_advantage=True),
        }
        values.update(changes)
        return CurriculumConfig(**values)

    def test_early_training_keeps_only_total_sample_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(Path(directory))
            coverage = SimpleNamespace(
                total_decisions=5_000,
                sampled_decisions=4_096,
                sampled_fraction=4_096 / 5_000,
                sampled_octiles=(512,) * 8,
            )

            row = runner._trajectory_row(
                completed_trajectory(),
                StateDescriptor(VALIDATION_STATE, None, 1, 3, 1, "early", ""),
                runner.config.stages[0],
                "training",
                0,
                None,
                1,
                training_sample_coverage=coverage,
            )

            self.assertEqual(row["sampled_training_decisions"], 4_096)
            self.assertEqual(row["total_training_decisions"], 5_000)
            self.assertEqual(row["run"], "training_early")
            self.assertNotIn("sampled_training_fraction", row)
            self.assertFalse(any(field.startswith("sampled_octile_") for field in row))

    def test_fresh_setup_diagnostics_are_written_compactly(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(Path(directory))
            descriptor = StateDescriptor(
                path=VALIDATION_STATE,
                metadata_path=None,
                map_num=1,
                player_count=3,
                seed=1,
                scenario="fresh",
                starting_position="",
                mission_cards_enabled=True,
                emperors_favour_enabled=True,
                emperors_favour_tiles=("+1Action", "DisplaceAnywhere", "+1DisplacedPiece"),
                bonus_marker_supply_mode="random_mix",
                bonus_marker_draw_supply=("Move3", "BlockTradeRoute"),
                starting_bonus_marker_routes=(
                    ("Move3", "A", "B"),
                    ("SwapOffice", "C", "D"),
                    ("PlaceAdjacent", "E", "F"),
                ),
            )

            row = runner._trajectory_row(
                completed_trajectory(),
                descriptor,
                runner.config.stages[0],
                "training",
                0,
                None,
                1,
            )

            self.assertEqual(row["run"], "training_fresh")
            self.assertTrue(row["mission_cards_enabled"])
            self.assertTrue(row["emperors_favour_enabled"])
            self.assertEqual(row["bonus_marker_supply_mode"], "random_mix")
            self.assertEqual(json.loads(row["emperors_favour_tiles"])[0], "+1Action")
            self.assertEqual(
                json.loads(row["bonus_marker_draw_supply"]), ["Move3", "BlockTradeRoute"]
            )
            self.assertEqual(len(json.loads(row["starting_bonus_marker_routes"])), 3)

    def runner(self, root, trainer=None, **config_changes):
        return TestRunner(
            trainer or FakeTrainer(),
            self.config(**config_changes),
            checkpoint_path=root / "model.pth",
            playable_model_path=root / "playable.pth",
            csv_path=root / "results.csv",
            temporary_directory=root / "states",
            failure_directory=root / "failures",
            evaluation_suite_directory=root / "evaluation",
            backup_directory=root / "backups",
        )

    @staticmethod
    def write_backup_sources(runner):
        runner.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "training_checkpoint_format": TRAINING_CHECKPOINT_FORMAT,
                "training_checkpoint_version": TRAINING_CHECKPOINT_VERSION,
            },
            runner.checkpoint_path,
        )
        torch.save(
            {
                "model_checkpoint_format": MODEL_CHECKPOINT_FORMAT,
                "model_checkpoint_version": MODEL_CHECKPOINT_VERSION,
            },
            runner.playable_model_path,
        )

    def test_rolling_backup_is_created_at_exact_hundred_game_milestones(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            runner = self.runner(root, trainer)
            self.write_backup_sources(runner)

            trainer.progress.completed_games = 99
            self.assertIsNone(runner._create_rolling_backup())
            trainer.progress.completed_games = 100
            first = runner._create_rolling_backup()
            trainer.progress.completed_games = 199
            self.assertIsNone(runner._create_rolling_backup())
            trainer.progress.completed_games = 200
            second = runner._create_rolling_backup()

            self.assertEqual(first.name, "game_0100")
            self.assertEqual(second.name, "game_0200")
            for backup in (first, second):
                self.assertTrue((backup / "hansa_nn_model.pth").is_file())
                self.assertTrue((backup / "training_checkpoint.pth").is_file())
                self.assertTrue((backup / "metadata.json").is_file())

    def test_rolling_backup_metadata_records_training_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(Path(directory), FakeTrainer())
            self.write_backup_sources(runner)
            runner.batch_number = 7
            runner.trainer.progress.completed_games = 300
            runner.trainer.progress.training_updates = 1_234

            backup = runner._create_rolling_backup()
            metadata = json.loads((backup / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(metadata["completed_training_games"], 300)
            self.assertEqual(metadata["batch_number"], 7)
            self.assertEqual(metadata["training_updates"], 1_234)
            self.assertEqual(metadata["training_checkpoint_format"], TRAINING_CHECKPOINT_FORMAT)
            self.assertEqual(metadata["training_checkpoint_version"], TRAINING_CHECKPOINT_VERSION)
            self.assertTrue(metadata["timestamp"].endswith("+00:00"))

    def test_rolling_backup_retains_only_ten_after_new_snapshot_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(Path(directory), FakeTrainer())
            self.write_backup_sources(runner)
            runner.backup_directory.mkdir(parents=True)
            for completed_games in range(100, 1_001, 100):
                (runner.backup_directory / f"game_{completed_games:04d}").mkdir()
            runner.trainer.progress.completed_games = 1_100

            runner._create_rolling_backup()

            backups = sorted(path.name for path in runner.backup_directory.glob("game_*"))
            self.assertEqual(len(backups), 10)
            self.assertNotIn("game_0100", backups)
            self.assertIn("game_1100", backups)

    def test_failed_rolling_backup_preserves_every_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(Path(directory), FakeTrainer())
            self.write_backup_sources(runner)
            runner.backup_directory.mkdir(parents=True)
            expected = {f"game_{completed_games:04d}" for completed_games in range(100, 1_001, 100)}
            for name in expected:
                (runner.backup_directory / name).mkdir()
            runner.trainer.progress.completed_games = 1_100

            with mock.patch(
                "training.curriculum.shutil.copy2",
                side_effect=(None, OSError("simulated interrupted backup")),
            ):
                with self.assertRaisesRegex(OSError, "simulated interrupted backup"):
                    runner._create_rolling_backup()

            remaining = {
                path.name
                for path in runner.backup_directory.iterdir()
                if path.name.startswith("game_")
            }
            self.assertEqual(remaining, expected)
            self.assertFalse((runner.backup_directory / "game_1100").exists())

    def test_training_milestone_publishes_backup_after_normal_active_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            trainer.progress.completed_games = 99
            runner = self.runner(root, trainer)

            with mock.patch("training.curriculum._validate_backup_artifacts"):
                runner.run()

            backup = root / "backups" / "game_0100"
            self.assertTrue((backup / "hansa_nn_model.pth").is_file())
            self.assertTrue((backup / "training_checkpoint.pth").is_file())
            self.assertEqual(trainer.checkpoint_calls, 2)

    def test_obsolete_mixed_evaluation_entries_are_not_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            (evaluation / "manifest.json").write_text(
                json.dumps(
                    [
                        {"evaluation_set": "mid_late_end"},
                        {"evaluation_set": "fresh"},
                        {"evaluation_set": "mixed_development"},
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(self.runner(root)._evaluation_game_count(), 2)

    def test_one_iteration_trains_evaluates_checkpoints_and_logs_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            runner = self.runner(root, trainer)
            messages = []
            runner.progress_callback = messages.append

            state = runner.run()

            self.assertEqual(trainer.collect_calls, 2)
            self.assertEqual(trainer.update_calls, 1)
            self.assertEqual(trainer.measured_losses, 1)
            self.assertEqual(trainer.progress.completed_games, 1)
            self.assertEqual(trainer.evaluation_calls, [False, True])
            self.assertEqual(runner.generated_seeds, [124, 10000])
            self.assertTrue((root / "model.pth").is_file())
            self.assertEqual((root / "playable.pth").read_text(encoding="utf-8"), "playable model")
            self.assertEqual(list((root / "states").iterdir()), [])
            self.assertEqual(state, trainer.saved_curriculum_state)
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(
                [row["run"] for row in rows],
                ["training_end_zero_epsilon", "evaluation_mid_late_end"],
            )
            self.assertEqual([row["game#"] for row in rows], ["1", "2"])
            self.assertEqual(rows[0]["latest_loss"], "12.5")
            self.assertEqual(rows[0]["total_training_decisions"], "1")
            self.assertEqual(rows[0]["sampled_training_decisions"], "1")
            self.assertEqual(trainer.curriculum_maturities, ["test"])
            self.assertEqual(rows[1]["latest_loss"], "10.0")
            self.assertEqual(rows[1]["evaluation_suite_size"], "1")
            self.assertEqual(rows[0]["state_seed"], "124")
            self.assertEqual(rows[0]["action_seed"], "1000000131")
            self.assertEqual(rows[0]["play_seconds"], "1.0")
            self.assertNotIn("inference_seconds", rows[0])
            self.assertNotEqual(rows[0]["generation_seconds"], "")
            self.assertNotEqual(rows[0]["learning_seconds"], "")
            prefix = "[Batch 1/1] "
            self.assertIn(f"{prefix}Training game 1/1...", messages)
            self.assertIn(
                f"{prefix}Starting stage 'mixed_end_game': 1 training games and 1 evaluation game(s)",
                messages,
            )
            self.assertTrue(
                any(
                    message.startswith(f"{prefix}Evaluation game 1/1: normal;")
                    for message in messages
                )
            )
            self.assertIn(f"{prefix}Saved training games 1", messages)

    def test_zero_epsilon_and_evaluation_run_modes_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            runner = self.runner(
                root,
                trainer,
                zero_epsilon_training_fractions=tuple(
                    (maturity, 1.0)
                    for maturity, _fraction in DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS
                ),
            )

            runner.run()

            self.assertEqual(trainer.update_calls, 1)
            self.assertEqual(trainer.zero_epsilon_calls, [True, False])
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            training, evaluation = rows
            self.assertEqual(training["run"], "training_end_zero_epsilon")
            self.assertEqual(evaluation["run"], "evaluation_mid_late_end")

    def test_fresh_zero_epsilon_game_uses_the_normal_training_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            runner = self.runner(
                root,
                trainer,
                zero_epsilon_training_fractions=tuple(
                    (maturity, 1.0)
                    for maturity, _fraction in DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS
                ),
            )
            runner._generate_state = lambda *_args, **_kwargs: StateDescriptor(
                VALIDATION_STATE,
                None,
                1,
                3,
                124,
                "fresh",
                "",
            )

            runner.run()

            self.assertEqual(trainer.update_calls, 1)
            self.assertEqual(trainer.curriculum_maturities, ["fresh"])
            self.assertEqual(trainer.zero_epsilon_calls, [True, False])
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                training = next(csv.DictReader(source))
            self.assertEqual(training["run"], "training_fresh_zero_epsilon")

    def test_evaluation_retries_replacement_route_deadlock(self):
        class RetryEvaluationTrainer(FakeTrainer):
            def __init__(self):
                super().__init__()
                self.evaluation_attempts = 0

            def collect_game(self, path, *, failure_callback=None, evaluation=False, **_kwargs):
                if evaluation:
                    self.evaluation_attempts += 1
                    if self.evaluation_attempts == 1:
                        self.collect_calls += 1
                        return deadlocked_trajectory()
                return super().collect_game(
                    path, failure_callback=failure_callback, evaluation=evaluation
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = RetryEvaluationTrainer()
            runner = self.runner(root, trainer)
            messages = []
            runner.progress_callback = messages.append

            runner.run()

            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            evaluation = rows[1]
            self.assertEqual(trainer.evaluation_attempts, 2)
            self.assertEqual(evaluation["completion_reason"], "normal")
            self.assertEqual(evaluation["retry_count"], "1")
            self.assertEqual(evaluation["action_seed"], "1000010008")
            self.assertIn(
                "[Batch 1/1] Evaluation game 1/1: no_replacement_route; retrying",
                messages,
            )

    def test_evaluation_retries_action_limit_without_recording_failure(self):
        class RetryEvaluationTrainer(FakeTrainer):
            def __init__(self):
                super().__init__()
                self.evaluation_attempts = 0

            def collect_game(self, path, *, failure_callback=None, evaluation=False, **_kwargs):
                if evaluation:
                    self.evaluation_attempts += 1
                    if self.evaluation_attempts == 1:
                        self.collect_calls += 1
                        raise ActionLimitExceeded("limit")
                return super().collect_game(
                    path, failure_callback=failure_callback, evaluation=evaluation
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = RetryEvaluationTrainer()
            runner = self.runner(root, trainer)

            runner.run()

            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(trainer.evaluation_attempts, 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["run"], "evaluation_mid_late_end")
            self.assertEqual(rows[1]["completion_reason"], "normal")
            self.assertEqual(rows[1]["retry_count"], "1")

    def test_exhausted_evaluation_is_not_recorded_and_does_not_stop_training(self):
        class DeadlockedEvaluationTrainer(FakeTrainer):
            def collect_game(self, path, *, failure_callback=None, evaluation=False, **_kwargs):
                if evaluation:
                    self.collect_calls += 1
                    return deadlocked_trajectory()
                return super().collect_game(
                    path, failure_callback=failure_callback, evaluation=evaluation
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = DeadlockedEvaluationTrainer()
            runner = self.runner(root, trainer, retry_limit=5)
            messages = []
            runner.progress_callback = messages.append

            runner.run()

            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["run"] for row in rows], ["training_end"])
            self.assertEqual(trainer.collect_calls, 4)
            self.assertIn("keeping the fixed state for the next batch", messages[-2])

    def test_multiple_iterations_append_rows_instead_of_replacing_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = self.runner(root, iterations=2)

            runner.run()

            self.assertEqual(runner.generated_seeds, [124, 10000, 20138, 10001])
            self.assertEqual(
                runner.generated_configurations,
                [(None, None), (1, 3), (None, None), (1, 4)],
            )
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["batch#"] for row in rows], ["1", "1", "2", "2"])

    def test_long_batch_saves_after_each_model_update_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            runner = self.runner(
                root,
                trainer,
                training_games_per_batch=5,
                update_batch_size=2,
            )

            runner.run()

            self.assertEqual(trainer.update_calls, 5)
            self.assertEqual(trainer.checkpoint_calls, 4)
            self.assertEqual(trainer.model.save_calls, 3)
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 6)

    def test_new_csv_starts_reporting_at_one_without_resetting_training_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            first = self.runner(root, trainer)
            trainer.curriculum_state = {
                "configuration_version": 4,
                "configuration_signature": first._configuration_signature(),
                "stage_index": 0,
                "batch_number": 7,
                "game_number": 70,
            }

            self.runner(root, trainer).run()

            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["game#"] for row in rows], ["1", "2"])
            self.assertEqual([row["batch#"] for row in rows], ["1", "1"])

    def test_action_limit_trains_without_retry_or_completed_csv_row(self):
        class LimitedTrainer(FakeTrainer):
            def collect_game(self, path, *, failure_callback=None, evaluation=False, **_kwargs):
                if not evaluation:
                    self.collect_calls += 1
                    return action_limited_trajectory()
                return super().collect_game(
                    path, failure_callback=failure_callback, evaluation=evaluation
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = LimitedTrainer()
            runner = self.runner(root, trainer)

            runner.run()

            self.assertEqual(trainer.update_calls, 1)
            self.assertEqual(trainer.collect_calls, 2)
            self.assertEqual(runner.generated_seeds, [124, 10000])
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [row["run"] for row in rows],
                ["training_end", "evaluation_mid_late_end"],
            )
            self.assertEqual(rows[0]["completion_reason"], "action_limit")

    def test_dead_end_saves_failure_and_retries_with_new_seed(self):
        class RetryTrainer(FakeTrainer):
            def __init__(self):
                super().__init__()
                self.exploration_modes = []

            def collect_game(self, path, *, failure_callback=None, evaluation=False, **_kwargs):
                self.exploration_modes.append(_kwargs.get("zero_epsilon", False))
                if self.collect_calls == 0:
                    self.collect_calls += 1
                    error = IncompleteGameError("no legal interaction")
                    failure_callback(None, (7, 8), (), error)
                    raise error
                return super().collect_game(
                    path, failure_callback=failure_callback, evaluation=evaluation
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = RetryTrainer()
            runner = self.runner(
                root,
                trainer,
                zero_epsilon_training_fractions=tuple(
                    (maturity, 1.0)
                    for maturity, _fraction in DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS
                ),
            )

            runner.run()

            self.assertEqual(len(set(runner.generated_seeds)), 3)
            self.assertEqual(len(list((root / "failures").iterdir())), 1)
            self.assertEqual(trainer.exploration_modes, [True, True, False])

    def test_replacement_deadlock_is_trained_marked_unfinished_and_retried(self):
        class RetryTrainer(FakeTrainer):
            def collect_game(self, path, *, failure_callback=None, evaluation=False, **_kwargs):
                if not evaluation and self.collect_calls == 0:
                    self.collect_calls += 1
                    return deadlocked_trajectory()
                return super().collect_game(
                    path, failure_callback=failure_callback, evaluation=evaluation
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = RetryTrainer()
            runner = self.runner(root, trainer)

            runner.run()

            self.assertEqual(trainer.update_calls, 2)
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertNotIn("completed", rows[0])
            self.assertEqual(
                [row["completion_reason"] for row in rows],
                ["normal", "normal"],
            )
            self.assertEqual([row["retry_count"] for row in rows], ["1", "0"])

    def test_exhausted_retries_discard_state_and_continue_until_game_completes(self):
        class RetryTrainer(FakeTrainer):
            def collect_game(self, path, *, failure_callback=None, evaluation=False, **_kwargs):
                if not evaluation and self.collect_calls < 2:
                    self.collect_calls += 1
                    return deadlocked_trajectory()
                return super().collect_game(
                    path, failure_callback=failure_callback, evaluation=evaluation
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = RetryTrainer()
            runner = self.runner(root, trainer, retry_limit=1)

            runner.run()

            self.assertEqual(trainer.update_calls, 3)
            self.assertEqual(runner.generated_seeds, [124, 125, 10131, 10000])
            with (root / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(
                [row["run"] for row in rows],
                ["training_end", "evaluation_mid_late_end"],
            )

    def test_generation_failure_retries_with_a_new_seed(self):
        class GenerationRetryRunner(TestRunner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.generation_attempts = 0

            def _generate_state(self, *args, **kwargs):
                self.generation_attempts += 1
                if self.generation_attempts == 1:
                    raise StateGenerationError("constraints not satisfied")
                return super()._generate_state(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = GenerationRetryRunner(
                FakeTrainer(),
                self.config(),
                checkpoint_path=root / "model.pth",
                playable_model_path=root / "playable.pth",
                csv_path=root / "results.csv",
                temporary_directory=root / "states",
                failure_directory=root / "failures",
                evaluation_suite_directory=root / "evaluation",
            )
            messages = []
            runner.progress_callback = messages.append

            runner.run()

            self.assertEqual(runner.generation_attempts, 3)
            self.assertEqual(runner.generated_seeds, [125, 10000])
            self.assertTrue(
                any(
                    "Training game 1/1 (retry 1: generation constraints:" in message
                    for message in messages
                )
            )

    def test_checkpoint_rejects_changed_curriculum_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            first = self.runner(root, trainer)
            trainer.curriculum_state = {
                "configuration_version": 4,
                "configuration_signature": first._configuration_signature(),
                "stage_index": 0,
            }

            changed_stage = (CurriculumStage("different", 10, (18, 19)),)
            with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
                self.runner(root, trainer, stages=changed_stage)

    def test_checkpoint_allows_changing_evaluation_game_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            first = self.runner(root, trainer)
            trainer.curriculum_state = {
                "configuration_version": 4,
                "configuration_signature": first._configuration_signature(),
                "stage_index": 0,
            }

            self.runner(root, trainer, evaluation_games_per_batch=2)

    def test_checkpoint_with_legacy_global_zero_epsilon_setting_preserves_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            trainer.optimizer = object()
            trainer.progress.completed_games = 12_345
            trainer.progress.training_updates = 6_789
            trainer.progress.policy_training_updates = 432
            first = self.runner(root, trainer)
            trainer.curriculum_state = {
                "configuration_version": 4,
                "configuration_signature": first._configuration_signature(
                    legacy_zero_epsilon_training_fraction=0.05
                ),
                "stage_index": 0,
                "batch_number": 27,
                "game_number": 99,
                "rolling_losses": [10.0, 9.0],
            }
            model = trainer.model
            optimizer = trainer.optimizer
            progress = deepcopy(trainer.progress)

            resumed = self.runner(root, trainer)

            self.assertIs(trainer.model, model)
            self.assertIs(trainer.optimizer, optimizer)
            self.assertEqual(trainer.progress, progress)
            self.assertEqual(resumed.batch_number, 27)
            self.assertEqual(resumed.game_number, 99)
            self.assertEqual(resumed.rolling_losses, [10.0, 9.0])

    def test_near_end_action_limit_upgrade_accepts_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = FakeTrainer()
            stages = (CurriculumStage("near_end_18_19", 3_000, (18, 19)),)
            runner = self.runner(root, trainer, stages=stages)
            trainer.curriculum_state = {
                "configuration_version": 4,
                "configuration_signature": runner._configuration_signature(
                    near_end_action_limit=2_000, retry_limit=2
                ),
            }

            resumed = self.runner(root, trainer, stages=stages)

            self.assertEqual(resumed.config.stages[0].action_limit, 3_000)

    def test_unexpected_error_stops_and_saves_one_reproducible_bundle(self):
        class ErrorTrainer(FakeTrainer):
            def collect_game(self, _path, *, failure_callback=None, evaluation=False, **_kwargs):
                error = ValueError("unexpected")
                failure_callback(None, (9,), (), error)
                raise error

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = self.runner(root, ErrorTrainer())

            with self.assertRaisesRegex(ValueError, "unexpected"):
                runner.run()

            failures = list((root / "failures").iterdir())
            self.assertEqual(len(failures), 1)
            self.assertTrue((failures[0] / "source_state.hansa").is_file())
            self.assertFalse((root / "playable.pth").exists())

    def test_promotion_requires_tier_one_to_outperform_lower_tiers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = self.runner(root)
            rows = [{}]
            tier_one_wins = completed_trajectory()
            tier_three_wins = SimpleNamespace(seat_tiers=(1, 3, 5), winner_indices=(1,))

            self.assertTrue(runner._should_promote(rows, rows, (tier_one_wins,), 0))
            self.assertFalse(runner._should_promote(rows, rows, (tier_one_wins,), 0, 1))
            self.assertFalse(runner._should_promote(rows, rows, (tier_three_wins,), 0))
            deadlock = [{"completion_reason": "no_replacement_route"}]
            self.assertFalse(runner._should_promote(deadlock, rows, (tier_one_wins,), 0))


if __name__ == "__main__":
    unittest.main()
