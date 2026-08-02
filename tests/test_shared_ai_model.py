import random
import unittest
from unittest import mock

import torch

from ai.observation_encoder import ObservationEncoder
from ai.observation_schema import OBSERVATION_SIZE
from drawing.game_window import GameWindow
from game.action_schema import ACTION_SPACE_SIZE
from game.game_config import GameConfiguration, PlayerControl
from game.game_runner import legal_action_indices, select_progress_action


class ProgressModel:
    """Deterministic logits that drive the existing complete-game test policy."""

    def __init__(self, game):
        self.game = game
        self.policy_rng = random.Random(game.seed)
        self.observer_indices = []

    def __call__(self, observation):
        expected = ObservationEncoder().build(self.game)
        if observation.shape != (1, OBSERVATION_SIZE):
            raise AssertionError(f"Unexpected observation shape: {observation.shape}")
        if not torch.equal(observation.squeeze(0).to(torch.int16), expected.features):
            raise AssertionError("Model observation does not match the acting player")

        self.observer_indices.append(expected.observer_index)
        legal = legal_action_indices(self.game)
        preferred = select_progress_action(self.game, legal, self.policy_rng)
        scores = torch.zeros((1, ACTION_SPACE_SIZE))
        scores[0, preferred] = 1
        return scores


class SharedAIModelTests(unittest.TestCase):
    def test_one_human_and_two_ai_players_complete_ai_turns(self):
        configuration = GameConfiguration(
            map_num=2,
            player_controls=(
                PlayerControl.HUMAN,
                PlayerControl.MAGNUS,
                PlayerControl.MAGNUS,
            ),
            seed=124,
        )
        with mock.patch.object(GameConfiguration, "_load_ai_model", return_value=None):
            game = configuration.create_game()
        model = ProgressModel(game)
        game.ai_model = model

        policy_rng = random.Random(game.seed)
        while game.current_player_index == 0:
            legal = legal_action_indices(game)
            game.apply_ai_action(select_progress_action(game, legal, policy_rng))

        window = object.__new__(GameWindow)
        window.game = game
        window.observation_encoder = ObservationEncoder()
        window.rng = random.Random(game.seed)

        for player_index in (1, 2):
            steps = 0
            while game.current_player_index == player_index:
                legal = list(legal_action_indices(game))
                if window.acting_player.control.is_human:
                    selected = select_progress_action(game, legal, policy_rng)
                else:
                    selected = window.choose_ai_action(legal)
                self.assertIn(selected, legal)
                game.apply_ai_action(selected)
                steps += 1
                self.assertLess(steps, 100)

        self.assertEqual(game.current_player_index, 0)
        self.assertIn(1, model.observer_indices)
        self.assertIn(2, model.observer_indices)


if __name__ == "__main__":
    unittest.main()
