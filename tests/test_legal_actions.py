import unittest

from game.game_runner import create_headless_game, legal_action_indices
from game.invariants import validate_game


class LegalActionTests(unittest.TestCase):
    def test_fresh_game_has_legal_actions(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        actions = legal_action_indices(game)
        self.assertTrue(actions)
        self.assertTrue(all(0 <= action < 619 for action in actions))
        self.assertTrue(validate_game(game))


if __name__ == "__main__":
    unittest.main()
