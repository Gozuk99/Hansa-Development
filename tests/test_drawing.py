import unittest

import pygame

from drawing.action_ui import action_label, phase_prompt
from drawing.drawing_utils import redraw_window
from game.game_config import GameConfiguration


class DrawingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.font.init()

    def test_render_pass_does_not_store_layout_on_engine_objects(self):
        game = GameConfiguration(map_num=1, seed=124).create_game()
        board = game.current_player.board
        marker_positions = [marker.position for marker in game.current_player.bonus_markers]
        before = (
            game.tile_rects,
            getattr(game, "opponents_used_bms_rects", None),
            board.start_x,
            board.actions_y,
            list(board.circle_buttons),
            list(board.button_labels),
            marker_positions,
        )
        surface = pygame.Surface((game.selected_map.map_width + 1100, game.selected_map.map_height))
        legal_actions = game.legal_action_mask().nonzero(as_tuple=True)[0].tolist()

        layout = redraw_window(surface, game, legal_actions)

        after = (
            game.tile_rects,
            getattr(game, "opponents_used_bms_rects", None),
            board.start_x,
            board.actions_y,
            list(board.circle_buttons),
            list(board.button_labels),
            [marker.position for marker in game.current_player.bonus_markers],
        )
        self.assertEqual(before, after)
        self.assertEqual(
            set(layout.action_rects),
            {action for action in legal_actions if 522 <= action < 527 or action == 618},
        )

    def test_piece_selection_reuses_income_indices_with_phase_specific_labels(self):
        game = GameConfiguration(map_num=2, seed=124).create_game()
        game.pending_route_piece_choices = [
            ("square", game.current_player, None),
            ("circle", game.current_player, None),
        ]

        self.assertEqual(phase_prompt(game), "Choose the required two-piece mix")
        self.assertEqual(action_label(522, game), "2 Traders")
        self.assertEqual(action_label(523, game), "1 Trader + 1 Merchant")
        self.assertEqual(action_label(524, game), "2 Merchants")

    def test_render_layout_is_stable_across_repeated_frames(self):
        game = GameConfiguration(map_num=3, seed=124).create_game()
        surface = pygame.Surface((game.selected_map.map_width + 1100, game.selected_map.map_height))
        legal_actions = game.legal_action_mask().nonzero(as_tuple=True)[0].tolist()

        first = redraw_window(surface, game, legal_actions)
        second = redraw_window(surface, game, legal_actions)

        self.assertEqual(first.action_rects, second.action_rects)
        self.assertEqual(first.tile_rects, second.tile_rects)


if __name__ == "__main__":
    unittest.main()
