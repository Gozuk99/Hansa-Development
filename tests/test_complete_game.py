import contextlib
import io
import unittest

from game.game_runner import run_game


class CompleteGameTests(unittest.TestCase):
    def test_known_seed_finishes_and_is_deterministic(self):
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_game(map_num=2, num_players=3, seed=124)
            second = run_game(map_num=2, num_players=3, seed=124)
        self.assertEqual(first.terminal_reason, "game_end")
        self.assertEqual(first.action_trace, second.action_trace)
        self.assertEqual(first.final_scores, second.final_scores)

    def test_supported_smoke_matrix_finishes(self):
        cases = (
            (2, 3, 125),
            (1, 3, 124),
            (3, 3, 124),
        )
        for map_num, num_players, seed in cases:
            with self.subTest(map_num=map_num, num_players=num_players, seed=seed):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_game(map_num, num_players, seed)
                self.assertEqual(result.terminal_reason, "game_end")


if __name__ == "__main__":
    unittest.main()
