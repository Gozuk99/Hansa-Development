from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from ai.ai_model import HansaNN, device
from ai.observation_encoder import ObservationEncoder
from game.action_schema import ACTION_SCHEMA_VERSION, ACTION_SPACE_SIZE
from game.persistence import load_game
from game.turn_state import TurnPhase
from map_data.constants import MAX_POSTS
from training.self_play import (
    PolicyTier,
    SelfPlayTrainer,
    TrainingConfig,
    TrainingDecision,
    assign_reward_to_go,
    calculate_terminal_rewards,
    training_action_mask,
)
from tools.train_self_play import train_with_periodic_checkpoints


STATE = Path("training_data/5p-map2_YELLOW_19_points_1_turn_from_winning.hansa")


def training_decision(action, player, rewards, immediate):
    empty = torch.empty(0)
    return TrainingDecision(
        empty,
        empty,
        action,
        player,
        rewards,
        immediate,
        1,
        0.05,
        2,
        False,
        1,
        1,
    )


class SelfPlayTrainingTests(unittest.TestCase):
    def trainer(self, seed=124):
        return SelfPlayTrainer(
            config=TrainingConfig(
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

    def test_move_filter_allows_moves_when_only_moves_exist(self):
        class Player:
            pass

        class Post:
            def __init__(self, owner):
                self.owner = owner
                self.required_shape = None
                self.region = None

            def is_owned(self):
                return self.owner is not None

        class Route:
            def __init__(self, posts):
                self.posts = posts

        class Game:
            def __init__(self, active_player, mask):
                self.players = [Player()]
                self.active_player = active_player
                self.turn_phase = TurnPhase.ACTIONS
                self.selected_map = type("Map", (), {"routes": [Route([Post(self.players[0])])]})()
                self._mask = mask

            @property
            def current_player(self):
                return self.players[self.active_player]

            def ai_action_mask(self):
                return self._mask

        base_mask = [False] * ACTION_SPACE_SIZE
        base_mask[0] = True
        base_mask[MAX_POSTS] = True
        game = Game(active_player=0, mask=base_mask)

        mask = training_action_mask(game, disable_move_action=True)

        self.assertEqual(mask.nonzero(as_tuple=False).flatten().tolist(), [0, MAX_POSTS])

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
        self.assertEqual(decision.policy_tier, trajectory.seat_tiers[4])
        self.assertEqual(decision.legal_action_count, 1)
        self.assertEqual(decision.model_rank, 1)
        self.assertIn(decision.policy_tier, {1, 2, 3, 4, 5})
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

    def test_epsilon_explores_all_legal_actions(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 5.0, 2.0])
        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.1
        trainer.rng.choice.return_value = 2

        selection = trainer._select_action(scores, [1, 2], PolicyTier(3, 10, 0.2))

        self.assertEqual(selection.action_index, 2)
        self.assertTrue(selection.used_epsilon)
        self.assertEqual(selection.model_rank, 2)
        trainer.rng.choice.assert_called_once_with([1, 2])

    def test_top_k_selects_uniformly_from_effective_legal_pool(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 5.0, 2.0, 7.0])
        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.9
        trainer.rng.choice.return_value = 1

        selection = trainer._select_action(scores, [1, 2, 3], PolicyTier(1, 2, 0.05))

        self.assertEqual(selection.action_index, 1)
        self.assertFalse(selection.used_epsilon)
        self.assertEqual(selection.model_rank, 2)
        self.assertEqual(selection.legal_action_count, 3)
        trainer.rng.choice.assert_called_once_with([3, 1])

    def test_effective_k_and_fully_random_tier(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 5.0, 2.0])
        trainer.rng = mock.Mock()
        trainer.rng.reset_mock()
        trainer.rng.random.return_value = 0.9
        trainer.rng.choice.return_value = 2

        selection = trainer._select_action(scores, [1, 2], PolicyTier(3, 10, 0.2))
        self.assertEqual(selection.action_index, 2)
        trainer.rng.choice.assert_called_once_with([1, 2])

        trainer.rng.reset_mock()
        trainer.rng.random.return_value = 0.2
        trainer.rng.choice.return_value = 1
        selection = trainer._select_action(scores, [1, 2], PolicyTier(5, None, 1.0))
        self.assertEqual(selection.action_index, 1)
        self.assertTrue(selection.used_epsilon)
        trainer.rng.choice.assert_called_once_with([1, 2])

    def test_single_legal_action_is_always_selected(self):
        trainer = self.trainer()
        scores = torch.tensor([10.0, -5.0])

        selection = trainer._select_action(scores, [1], PolicyTier(1, 2, 0.05))

        self.assertEqual(selection.action_index, 1)
        self.assertEqual(selection.model_rank, 1)
        self.assertEqual(selection.legal_action_count, 1)

    def test_default_tier_subsets_and_random_assignment(self):
        trainer = self.trainer(seed=99)

        self.assertEqual(set(tier.number for tier in trainer._assign_tiers(3)), {1, 3, 5})
        self.assertEqual(set(tier.number for tier in trainer._assign_tiers(4)), {1, 2, 4, 5})
        assignments = [tuple(tier.number for tier in trainer._assign_tiers(5)) for _ in range(8)]
        self.assertTrue(all(set(assignment) == {1, 2, 3, 4, 5} for assignment in assignments))
        self.assertGreater(len(set(assignments)), 1)

    def test_fixed_seed_reproduces_trajectory(self):
        first = self.trainer(seed=99).collect_game(STATE)
        second = self.trainer(seed=99).collect_game(STATE)

        self.assertEqual(first.action_trace, second.action_trace)
        self.assertEqual(first.seat_tiers, second.seat_tiers)
        self.assertEqual(first.terminal_rewards, second.terminal_rewards)
        self.assertEqual(
            [decision.reward_to_go for decision in first.decisions],
            [decision.reward_to_go for decision in second.decisions],
        )
        self.assertEqual(first.final_scores, second.final_scores)

    def test_discounted_reward_to_go_is_separate_for_each_player(self):
        decisions = (
            training_decision(1, 0, (100, 0), 100),
            training_decision(2, 1, (0, 200), 200),
            training_decision(3, 0, (-50, 0), -50),
            training_decision(4, 1, (0, 0), 0),
        )

        completed = assign_reward_to_go(decisions, (1000, 0), gamma=0.5)

        self.assertEqual([item.reward_to_go for item in completed], [212.5, 200, 450, 0])

    def test_terminal_reward_is_winner_only_and_trigger_bonus_requires_winner(self):
        game = mock.Mock()
        game.players = [mock.Mock(final_score=40), mock.Mock(final_score=35)]

        self.assertEqual(calculate_terminal_rewards(game, (0,), 1), (4000, 0))
        self.assertEqual(calculate_terminal_rewards(game, (0,), 0), (4150, 0))

        losing_decision = training_decision(1, 1, (0, 200), 200)
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
            self.assertEqual(restored.config.tier_top_k, (2, 5, 10, 15, None))
            self.assertEqual(restored.config.three_player_tiers, (1, 3, 5))
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

    def test_tier_metrics_are_recorded_by_tier_not_seat(self):
        trainer = self.trainer()
        trajectory = trainer.collect_game(STATE)
        tier = trajectory.decisions[0].policy_tier

        metrics = trainer.tier_metrics()[tier]

        self.assertEqual(metrics["games"], 1)
        self.assertEqual(metrics["wins"], 1)
        self.assertEqual(metrics["win_rate"], 1)
        self.assertEqual(metrics["average_selected_rank"], 1)
        self.assertEqual(metrics["epsilon_selections"] + metrics["top_k_selections"], 1)
        self.assertEqual(metrics["average_immediate_reward"], 200)
        self.assertEqual(metrics["average_reward_to_go"], 3450)

    def test_only_selected_output_receives_direct_training_gradient(self):
        class IndependentOutputs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.zeros(ACTION_SPACE_SIZE))

            def forward(self, observations):
                return self.values.unsqueeze(0).expand(len(observations), -1)

        model = IndependentOutputs().to(device)
        trainer = SelfPlayTrainer(model=model, config=TrainingConfig())
        decision = replace(training_decision(7, 0, (100,), 100), reward_to_go=100.0)
        trajectory = mock.Mock(decisions=(decision,))

        trainer.update_model((trajectory,))

        self.assertNotEqual(float(model.values[7]), 0)
        unchanged = torch.cat((model.values[:7], model.values[8:]))
        self.assertTrue(torch.equal(unchanged, torch.zeros_like(unchanged)))

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
