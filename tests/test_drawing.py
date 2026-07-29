import unittest

import pygame

from ai.game_state import BoardData
from drawing.action_ui import action_label, fit_text, phase_prompt
from drawing.ai_observation import public_game_state
from drawing.drawing_utils import draw_upgrades, redraw_window
from drawing.game_window import GameWindow
from game.game_config import GameConfiguration, PlayerControl
from map_data.map_attributes import BonusMarker


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

    def test_special_prestige_track_is_rendered_on_every_map(self):
        surface = pygame.Surface((3000, 1500))
        for map_num in (1, 2, 3):
            with self.subTest(map_num=map_num):
                game = GameConfiguration(map_num=map_num, seed=124).create_game()
                before = surface.copy()
                draw_upgrades(surface, game.selected_map)
                prestige = game.selected_map.specialprestigepoints
                area = pygame.Rect(prestige.x_pos, prestige.y_pos, prestige.width, prestige.height)
                self.assertNotEqual(
                    pygame.image.tostring(before.subsurface(area), "RGB"),
                    pygame.image.tostring(surface.subsurface(area), "RGB"),
                )

    def test_wrong_city_middle_click_never_applies_other_endpoint_upgrade(self):
        game = GameConfiguration(map_num=1, seed=124).create_game()
        route = next(
            route
            for route in game.selected_map.routes
            if bool(route.cities[0].upgrade_city_type) != bool(route.cities[1].upgrade_city_type)
        )
        upgrade_city = next(city for city in route.cities if city.upgrade_city_type)
        other_city = next(city for city in route.cities if city is not upgrade_city)
        for post in route.posts:
            post.owner = game.current_player
            post.owner_piece_shape = "square"
        legal_actions = game.legal_action_mask().nonzero(as_tuple=True)[0].tolist()
        window = GameWindow.__new__(GameWindow)
        window.game = game
        window.action_rects = []
        center = (
            other_city.x_pos + other_city.width // 2,
            other_city.y_pos + other_city.height // 2,
        )

        self.assertIsNone(window.action_for_click(center, 2, legal_actions))

    def test_tribute_and_complete_bank_labels_describe_exact_pieces(self):
        game = GameConfiguration(map_num=1, seed=124).create_game()
        owner = game.players[1]
        owner.general_stock_squares = 1
        owner.general_stock_circles = 0
        game.begin_tribute_income_responses([owner])
        self.assertEqual(action_label(522, game), "1 Trader")

        game.pending_tribute_income_owners.clear()
        game.active_player = game.current_player_index
        game.current_player.bank = 50
        game.current_player.general_stock_squares = 3
        game.current_player.general_stock_circles = 2
        self.assertEqual(action_label(523, game), "Income: 3 Traders + 1 Merchant")
        self.assertEqual(action_label(524, game), "Income: 3 Traders + 2 Merchants")

    def test_acting_player_controller_follows_out_of_turn_responder(self):
        controls = (PlayerControl.HUMAN, PlayerControl.EASY, PlayerControl.HUMAN)
        game = GameConfiguration(player_controls=controls, seed=124).create_game()
        game.active_player = 1
        window = GameWindow.__new__(GameWindow)
        window.game = game

        self.assertIs(window.acting_player, game.players[1])
        self.assertIs(window.acting_player.control, PlayerControl.EASY)

    def test_gui_ai_observation_conceals_face_down_and_private_information(self):
        game = GameConfiguration(
            map_num=1,
            player_controls=(PlayerControl.HUMAN,) * 3,
            use_mission_cards=True,
            seed=124,
        ).create_game()
        observer = game.players[0]
        opponent = game.players[1]
        opponent.used_bonus_markers = [BonusMarker("Move3")]
        board_data = BoardData()

        state = public_game_state(board_data, game, observer)
        player_start = (
            board_data.game_tensor_size + board_data.city_tensor_size + board_data.route_tensor_size
        )

        self.assertEqual(state[24:36].count_nonzero().item(), 0)
        self.assertGreater(
            state[player_start + 20 : player_start + 23].count_nonzero().item(),
            0,
        )
        opponent_start = player_start + 55
        self.assertEqual(
            state[opponent_start + 20 : opponent_start + 23].count_nonzero().item(),
            0,
        )
        self.assertEqual(state[opponent_start + 35].item(), 1)
        self.assertEqual(
            state[opponent_start + 36 : opponent_start + 47].count_nonzero().item(),
            0,
        )

    def test_contextual_labels_fit_their_button_width(self):
        font = pygame.font.SysFont(None, 20)
        label = fit_text(font, "Income: 3 Traders + 2 Merchants", 160)

        self.assertLessEqual(font.size(label)[0], 160)


if __name__ == "__main__":
    unittest.main()
