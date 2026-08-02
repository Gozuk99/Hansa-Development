from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from ai.ai_model import HansaNN
from ai.observation_encoder import ObservationEncoder
from game.action_schema import ACTION_SCHEMA_VERSION, ACTION_SPACE_SIZE
from game.persistence import load_game
from training.self_play import (
    SelfPlayTrainer,
    TrainingConfig,
    TrainingDecision,
    assign_reward_to_go,
    calculate_terminal_rewards,
    training_action_mask,
)
from tools.train_self_play import train_with_periodic_checkpoints


STATE = Path("training_data/5p-map2_YELLOW_19_points_1_turn_from_winning.hansa")


class SelfPlayTrainingTests(unittest.TestCase):
    def trainer(self, seed=124):
        return SelfPlayTrainer(
            config=TrainingConfig(
                epsilon=0.2,
                learning_rate=0.0001,
                max_actions=10,
                disable_move_action=True,
                seed=seed,
            )
        )

    def test_move_filter_leaves_the_winning_route_interaction(self):
        game = load_game(STATE)

        mask = training_action_mask(game, disable_move_action=True)

        self.assertEqual(mask.numel(), ACTION_SPACE_SIZE)
        self.assertEqual(mask.nonzero(as_tuple=False).flatten().tolist(), [285])
        self.assertEqual(game.ai_action_mask()[285], 1)

    def test_complete_trajectory_uses_acting_players_visible_observation(self):
        game = load_game(STATE)
        expected = ObservationEncoder().build(game)
        trainer = self.trainer()
        before = tuple(parameter.detach().clone() for parameter in trainer.model.parameters())

        trajectory = trainer.collect_game(STATE)

        self.assertEqual(len(trajectory.decisions), 1)
        decision = trajectory.decisions[0]
        self.assertEqual(decision.acting_player_index, 4)
        self.assertTrue(torch.equal(decision.observation, expected.features))
        self.assertEqual(decision.legal_action_mask.nonzero().flatten().tolist(), [285])
        self.assertEqual(decision.action_index, 285)
        self.assertEqual(decision.player_reward_deltas, (0, 0, 0, 0, 200))
        self.assertEqual(trajectory.action_trace, (285,))
        self.assertEqual(trajectory.winner_indices, (4,))
        self.assertEqual(decision.immediate_reward, 200)
        self.assertEqual(trajectory.terminal_rewards[4], 3250)
        self.assertEqual(decision.reward_to_go, 3450)
        self.assertTrue(
            all(
                torch.equal(old, current)
                for old, current in zip(before, trainer.model.parameters())
            )
        )

    def test_update_changes_shared_model_only_after_completed_game(self):
        trainer = self.trainer()
        trajectory = trainer.collect_game(STATE)
        before = tuple(parameter.detach().clone() for parameter in trainer.model.parameters())

        loss = trainer.update_model((trajectory,))

        self.assertGreater(loss, 0)
        self.assertTrue(
            any(
                not torch.equal(old, current)
                for old, current in zip(before, trainer.model.parameters())
            )
        )
        self.assertEqual(trainer.progress.completed_games, 1)
        self.assertEqual(trainer.progress.training_updates, 1)

    def test_epsilon_greedy_uses_random_exploration_or_highest_value(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 5.0, 2.0])
        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.1
        trainer.rng.choice.return_value = 2

        self.assertEqual(trainer._select_action(scores, [1, 2]), 2)
        trainer.rng.choice.assert_called_once_with([1, 2])

        trainer.rng.reset_mock()
        trainer.rng.random.return_value = 0.9
        self.assertEqual(trainer._select_action(scores, [1, 2]), 1)
        trainer.rng.choice.assert_not_called()

    def test_fixed_seed_reproduces_trajectory(self):
        first = self.trainer(seed=99).collect_game(STATE)
        second = self.trainer(seed=99).collect_game(STATE)

        self.assertEqual(first.action_trace, second.action_trace)
        self.assertEqual(first.terminal_rewards, second.terminal_rewards)
        self.assertEqual(
            [decision.reward_to_go for decision in first.decisions],
            [decision.reward_to_go for decision in second.decisions],
        )
        self.assertEqual(first.final_scores, second.final_scores)

    def test_discounted_reward_to_go_is_separate_for_each_player(self):
        empty = torch.empty(0)
        decisions = (
            TrainingDecision(empty, empty, 1, 0, (100, 0), 100),
            TrainingDecision(empty, empty, 2, 1, (0, 200), 200),
            TrainingDecision(empty, empty, 3, 0, (-50, 0), -50),
            TrainingDecision(empty, empty, 4, 1, (0, 0), 0),
        )

        completed = assign_reward_to_go(decisions, (1000, 0), gamma=0.5)

        self.assertEqual([item.reward_to_go for item in completed], [212.5, 200, 450, 0])

    def test_terminal_reward_is_winner_only_and_trigger_bonus_requires_winner(self):
        game = mock.Mock()
        game.players = [mock.Mock(final_score=40), mock.Mock(final_score=35)]

        self.assertEqual(calculate_terminal_rewards(game, (0,), 1), (4000, 0))
        self.assertEqual(calculate_terminal_rewards(game, (0,), 0), (4150, 0))

        empty = torch.empty(0)
        losing_decision = TrainingDecision(empty, empty, 1, 1, (0, 200), 200)
        completed = assign_reward_to_go((losing_decision,), (4000, 0), gamma=0.99)
        self.assertEqual(completed[0].reward_to_go, 200)

    def test_checkpoint_resumes_model_optimizer_progress_and_rng(self):
        trainer = self.trainer()
        trainer.train((STATE,), episodes=1, batch_size=1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            trainer.save_checkpoint(checkpoint, (STATE,))
            restored = SelfPlayTrainer.from_checkpoint(checkpoint)

            self.assertEqual(restored.progress.completed_games, 1)
            self.assertEqual(restored.progress.training_updates, 1)
            self.assertEqual(restored.progress.checkpoint_saves, 1)
            self.assertEqual(restored.progress.checkpoint_loads, 1)
            self.assertEqual(restored.config.gamma, 0.99)
            self.assertEqual(restored.rng.getstate(), trainer.rng.getstate())
            self.assertTrue(
                all(
                    torch.equal(first, second)
                    for first, second in zip(
                        trainer.model.parameters(), restored.model.parameters()
                    )
                )
            )
            restored.train((STATE,), episodes=1, batch_size=1)

    def test_long_run_saves_periodic_resume_points(self):
        trainer = mock.Mock()
        winner = mock.Mock(decisions=(mock.Mock(acting_player_index=1),), winner_indices=(1,))
        loser = mock.Mock(decisions=(mock.Mock(acting_player_index=2),), winner_indices=(1,))
        trainer.train.side_effect = [(winner, loser), (winner,)]

        trajectories = train_with_periodic_checkpoints(
            trainer,
            (STATE,),
            episodes=3,
            batch_size=2,
            checkpoint_every=2,
            checkpoint_path=Path("checkpoint.pth"),
        )

        self.assertEqual(trajectories.completed_games, 3)
        self.assertEqual(trajectories.decisions, 3)
        self.assertEqual(trajectories.acting_player_wins, 2)
        self.assertEqual(trainer.train.call_count, 2)
        self.assertEqual(trainer.save_checkpoint.call_count, 2)

    def test_checkpoint_rejects_incompatible_action_schema(self):
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            trainer.save_checkpoint(checkpoint, (STATE,))
            contents = torch.load(checkpoint, map_location="cpu")
            contents["action_schema_version"] = ACTION_SCHEMA_VERSION + 1
            torch.save(contents, checkpoint)

            with self.assertRaisesRegex(ValueError, "action schema"):
                SelfPlayTrainer.from_checkpoint(checkpoint)

    def test_training_checkpoint_is_usable_for_inference(self):
        trainer = self.trainer()
        trainer.train((STATE,), episodes=1, batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            trainer.save_checkpoint(checkpoint, (STATE,))

            inference_model = HansaNN(model_file=checkpoint)

        self.assertFalse(inference_model.training)


if __name__ == "__main__":
    unittest.main()
