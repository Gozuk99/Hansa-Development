import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from ai.ai_model import HansaNN
from tests.action_helpers import evaluation_state
from tools.evaluate_model_lineage import (
    DISCOVERY_FIELDS,
    GreedyHeadToHeadTrainer,
    ModelArtifact,
    T0_SELECTION_MODE,
    _write_csv,
    discover_models,
    evaluate_head_to_head_pair,
    mirrored_seat_ownership,
)
from training.self_play import IncompleteGameError, SelfPlayTrainer, TrainingConfig


STATE = evaluation_state("map1_3p_near_score")


class ModelLineageEvaluationTests(unittest.TestCase):
    def test_discovery_reports_ready_missing_and_ambiguous_archives_without_guessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            (archive / "v1").mkdir(parents=True)
            (archive / "v1" / "hansa_nn_model_v1.pth").write_bytes(b"v1")
            (archive / "v2").mkdir()
            (archive / "v3").mkdir()
            (archive / "v3" / "hansa_nn_model.pth").write_bytes(b"first")
            (archive / "v3" / "hansa_nn_model_old.pth").write_bytes(b"second")
            current = root / "hansa_nn_model.pth"
            current.write_bytes(b"current")

            artifacts, rows = discover_models(
                archive,
                current,
                model_loader=lambda _path: object(),
            )

            self.assertEqual([artifact.name for artifact in artifacts], ["v1", "current"])
            self.assertEqual(
                {row["model"]: row["status"] for row in rows},
                {"v1": "ready", "v2": "missing", "v3": "ambiguous", "current": "ready"},
            )

    def test_discovery_reports_a_model_that_fails_compatibility_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            model_path = archive / "broken" / "hansa_nn_model.pth"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"broken")
            current = root / "hansa_nn_model.pth"
            current.write_bytes(b"current")

            def loader(path):
                if path.name == "hansa_nn_model.pth" and path.parent.name == "broken":
                    raise ValueError("incompatible")
                return object()

            artifacts, rows = discover_models(archive, current, model_loader=loader)

            self.assertEqual([artifact.name for artifact in artifacts], ["current"])
            broken = next(row for row in rows if row["model"] == "broken")
            self.assertEqual(broken["status"], "invalid")
            self.assertIn("incompatible", broken["detail"])

    def test_mirrored_ownership_balances_every_supported_player_count(self):
        for player_count in (3, 4, 5):
            with self.subTest(player_count=player_count):
                pair = mirrored_seat_ownership(player_count, "current", "archive")
                self.assertEqual(
                    sum(owner == "current" for row in pair for owner in row), player_count
                )
                self.assertEqual(
                    sum(owner == "archive" for row in pair for owner in row), player_count
                )
                self.assertEqual(
                    pair[1],
                    tuple("archive" if owner == "current" else "current" for owner in pair[0]),
                )

    def test_per_seat_models_are_evaluation_only(self):
        trainer = SelfPlayTrainer(
            model=HansaNN(),
            config=TrainingConfig(max_actions=1, disable_move_action=False),
        )
        with self.assertRaisesRegex(ValueError, "restricted to evaluation"):
            trainer.collect_game(STATE, evaluation_models_by_seat=(trainer.model,) * 3)

    def test_per_seat_evaluation_does_not_change_parameters_or_action_trace(self):
        model = HansaNN()
        config = TrainingConfig(max_actions=20, disable_move_action=False, seed=991)
        baseline = SelfPlayTrainer(model=model, config=config)
        routed = SelfPlayTrainer(model=HansaNN(), config=config)
        routed.model.load_state_dict(model.state_dict())
        before = {name: value.detach().clone() for name, value in routed.model.state_dict().items()}

        baseline.rng.seed(12345)
        expected = baseline.collect_game(STATE, evaluation=True, capture_action_limit=True)
        routed.rng.seed(12345)
        with mock.patch.object(routed.optimizer, "step") as optimizer_step:
            actual = routed.collect_game(
                STATE,
                evaluation=True,
                capture_action_limit=True,
                evaluation_models_by_seat=(routed.model,) * 3,
            )
        optimizer_step.assert_not_called()

        self.assertEqual(actual.action_trace, expected.action_trace)
        self.assertEqual(actual.final_scores, expected.final_scores)
        self.assertEqual(actual.completion_reason, expected.completion_reason)
        for name, value in routed.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_t0_uses_greedy_top_one_zero_epsilon_for_actions_and_workflows(self):
        trainer = GreedyHeadToHeadTrainer(model=HansaNN(), config=TrainingConfig(seed=451))
        tiers = trainer._assign_evaluation_tiers(5, 99)
        self.assertEqual([tier.number for tier in tiers], [0] * 5)
        self.assertEqual([tier.top_k for tier in tiers], [1] * 5)
        self.assertEqual([tier.epsilon for tier in tiers], [0.0] * 5)

        scores = torch.tensor((1.0, 9.0, 4.0, 3.0))
        paid = trainer._select_action(scores, (0, 1, 2), tiers[0])
        workflow = trainer._select_workflow_action(
            scores,
            (0, 1, 2, 3),
            (((0,), (1,)), ((2,), (3,))),
        )
        self.assertEqual((paid.action_index, paid.model_rank, paid.used_epsilon), (1, 1, False))
        self.assertEqual(
            (workflow.action_index, workflow.model_rank, workflow.used_epsilon),
            (1, 1, False),
        )

    def test_t0_seeded_game_is_deterministic_and_does_not_update_the_model(self):
        model = HansaNN()
        config = TrainingConfig(max_actions=20, disable_move_action=False, seed=991)
        first = GreedyHeadToHeadTrainer(model=model, config=config)
        second = GreedyHeadToHeadTrainer(model=HansaNN(), config=config)
        second.model.load_state_dict(model.state_dict())
        with torch.no_grad():
            second.model.policy_head.weight.fill_(100.0)
            second.model.policy_head.bias.copy_(
                torch.arange(second.model.policy_head.bias.numel(), dtype=torch.float32)
            )
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}

        first.rng.seed(8877)
        first_result = first.collect_game(STATE, evaluation=True, capture_action_limit=True)
        second.rng.seed(8877)
        second_result = second.collect_game(STATE, evaluation=True, capture_action_limit=True)

        self.assertEqual(first_result.action_trace, second_result.action_trace)
        self.assertEqual(first_result.final_scores, second_result.final_scores)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_normal_fixed_evaluation_tier_assignment_is_unchanged(self):
        trainer = SelfPlayTrainer(model=HansaNN(), config=TrainingConfig())
        self.assertEqual(
            [tier.number for tier in trainer._assign_evaluation_tiers(3, 0)],
            [1, 3, 5],
        )
        self.assertEqual(
            [tier.number for tier in trainer._assign_evaluation_tiers(4, 0)],
            [1, 2, 4, 5],
        )
        self.assertTrue(
            all(
                tier.epsilon == 0
                for player_count in (3, 4, 5)
                for tier in trainer._assign_evaluation_tiers(player_count, 0)
            )
        )

    def test_csv_writer_emits_the_declared_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "discovery.csv"
            row = dict.fromkeys(DISCOVERY_FIELDS, "value")
            _write_csv(path, DISCOVERY_FIELDS, (row,))
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                self.assertEqual(tuple(reader.fieldnames), DISCOVERY_FIELDS)
                self.assertEqual(next(reader), row)

    def test_head_to_head_retries_both_mirrors_after_a_dead_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state.hansa").touch()
            (root / "manifest.json").write_text(
                """[{"name":"state","save_file":"state.hansa","evaluation_set":"fresh",
                "scenario":"fresh_game","map_num":1,"player_count":3,"seed":100}]""",
                encoding="utf-8",
            )
            current = ModelArtifact("current", root / "current.pth", "current-hash")
            archive = ModelArtifact("v1", root / "v1.pth", "archive-hash")
            trajectory = SimpleNamespace(
                winner_indices=(0,),
                seat_tiers=(0, 0, 0),
                final_scores=(30, 20, 10),
                completion_reason="20_points",
                action_trace=(1, 2),
            )
            completed_pairs = []
            progress_messages = []

            with (
                mock.patch(
                    "tools.evaluate_model_lineage._load_model",
                    side_effect=(HansaNN(), HansaNN()),
                ),
                mock.patch.object(
                    GreedyHeadToHeadTrainer,
                    "collect_game",
                    side_effect=(
                        trajectory,
                        IncompleteGameError("dead end"),
                        trajectory,
                        trajectory,
                    ),
                ) as collect_game,
            ):
                details = evaluate_head_to_head_pair(
                    current,
                    archive,
                    root,
                    progress_callback=progress_messages.append,
                    completed_pair_callback=completed_pairs.append,
                )

            self.assertEqual(collect_game.call_count, 4)
            self.assertEqual(len(details), 2)
            self.assertEqual(len(completed_pairs), 1)
            self.assertEqual({row["retry_count"] for row in details}, {1})
            self.assertEqual({row["action_seed"] for row in details}, {100 + 1_000_000_007 + 1})
            self.assertEqual({row["selection_mode"] for row in details}, {T0_SELECTION_MODE})
            self.assertTrue(
                any(
                    "engine_dead_end: dead end; retrying both mirrors" in message
                    for message in progress_messages
                )
            )

    def test_exhausted_head_to_head_dead_end_is_recorded_instead_of_raised(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state.hansa").touch()
            (root / "manifest.json").write_text(
                """[{"name":"state","save_file":"state.hansa","evaluation_set":"fresh",
                "scenario":"fresh_game","map_num":1,"player_count":3,"seed":100}]""",
                encoding="utf-8",
            )
            current = ModelArtifact("current", root / "current.pth", "current-hash")
            archive = ModelArtifact("v1", root / "v1.pth", "archive-hash")

            with (
                mock.patch(
                    "tools.evaluate_model_lineage._load_model",
                    side_effect=(HansaNN(), HansaNN()),
                ),
                mock.patch.object(
                    GreedyHeadToHeadTrainer,
                    "collect_game",
                    side_effect=IncompleteGameError("dead end"),
                ) as collect_game,
            ):
                details = evaluate_head_to_head_pair(
                    current,
                    archive,
                    root,
                    progress_callback=None,
                )

            self.assertEqual(collect_game.call_count, 3)
            self.assertEqual(len(details), 2)
            self.assertEqual({row["completion_reason"] for row in details}, {"engine_dead_end"})
            self.assertEqual({row["failure_detail"] for row in details}, {"dead end"})
            self.assertEqual({row["retry_count"] for row in details}, {2})

    def test_head_to_head_details_are_labeled_t0(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state.hansa").touch()
            (root / "manifest.json").write_text(
                """[{"name":"state","save_file":"state.hansa","evaluation_set":"fresh",
                "scenario":"fresh_game","map_num":1,"player_count":3,"seed":100}]""",
                encoding="utf-8",
            )
            current = ModelArtifact("current", root / "current.pth", "current-hash")
            archive = ModelArtifact("v1", root / "v1.pth", "archive-hash")
            trajectory = SimpleNamespace(
                winner_indices=(0,),
                seat_tiers=(0, 0, 0),
                final_scores=(30, 20, 10),
                completion_reason="20_points",
                action_trace=(1, 2),
            )
            with (
                mock.patch(
                    "tools.evaluate_model_lineage._load_model",
                    side_effect=(HansaNN(), HansaNN()),
                ),
                mock.patch.object(
                    GreedyHeadToHeadTrainer,
                    "collect_game",
                    side_effect=(trajectory, trajectory),
                ),
            ):
                details = evaluate_head_to_head_pair(
                    current,
                    archive,
                    root,
                    progress_callback=None,
                )

            self.assertEqual({row["selection_mode"] for row in details}, {T0_SELECTION_MODE})
            self.assertTrue(all("Top-1" in row["selection_configuration"] for row in details))


if __name__ == "__main__":
    unittest.main()
