import importlib
import random
import unittest
from unittest import mock

import hansa_game
from drawing.game_window import GameWindow
from drawing.new_game_menu import NewGameMenuState
from game.game_config import (
    EMPERORS_FAVOUR_TILES,
    GameConfiguration,
    PlayerControl,
    choose_ranked_ai_action,
)
from map_data.map_attributes import Map


class GameConfigurationTests(unittest.TestCase):
    def test_defaults_assign_every_active_player_to_human(self):
        configuration = GameConfiguration()

        self.assertEqual(configuration.player_count, 3)
        self.assertEqual(
            configuration.player_controls,
            (PlayerControl.HUMAN,) * 3,
        )
        self.assertFalse(configuration.has_ai_players)
        game = configuration.create_game()
        self.assertTrue(all(player.hansa_nn is None for player in game.players))

    def test_every_supported_map_and_player_count_builds_from_configuration(self):
        for map_num in (1, 2, 3):
            for player_count in (3, 4, 5):
                with self.subTest(map_num=map_num, player_count=player_count):
                    configuration = GameConfiguration(
                        map_num=map_num,
                        player_count=player_count,
                        player_controls=(PlayerControl.HUMAN,) * player_count,
                        use_mission_cards=map_num == 1,
                        seed=124,
                    )
                    game = configuration.create_game()
                    self.assertEqual(game.map_num, map_num)
                    self.assertEqual(len(game.players), player_count)

    def test_menu_player_count_adds_human_seats_and_map_hides_missions(self):
        state = NewGameMenuState()
        state.set_player_count(5)
        self.assertEqual(state.player_controls, [PlayerControl.HUMAN] * 5)

        state.use_mission_cards = True
        state.set_map(2)
        self.assertFalse(state.use_mission_cards)

    def test_configuration_rejects_invalid_module_combinations(self):
        with self.assertRaisesRegex(ValueError, "only be enabled on map 1"):
            GameConfiguration(
                map_num=2,
                player_count=3,
                player_controls=(PlayerControl.HUMAN,) * 3,
                use_mission_cards=True,
            )

        with self.assertRaisesRegex(ValueError, "exactly 3"):
            GameConfiguration(
                player_controls=(PlayerControl.HUMAN,) * 3,
                use_emperors_favour=True,
                emperor_tile_mode="manual",
                emperor_tiles=EMPERORS_FAVOUR_TILES[:2],
            )

        with self.assertRaisesRegex(ValueError, "Too many ExchangeBonusMarker"):
            GameConfiguration(
                player_controls=(PlayerControl.HUMAN,) * 3,
                use_promo_markers=True,
                promo_marker_mode="manual",
                promo_markers=("ExchangeBonusMarker",) * 3,
            )

    def test_manual_options_are_applied_to_game_initialization(self):
        controls = (
            PlayerControl.HUMAN,
            PlayerControl.EASY,
            PlayerControl.MAGNUS,
        )
        selected_tiles = EMPERORS_FAVOUR_TILES[:3]
        selected_promos = (
            "ExchangeBonusMarker",
            "Tribute4EstablishingTP",
            "BlockTradeRoute",
        )
        configuration = GameConfiguration(
            map_num=1,
            player_count=3,
            player_controls=controls,
            use_mission_cards=True,
            use_emperors_favour=True,
            emperor_tile_mode="manual",
            emperor_tiles=selected_tiles,
            use_promo_markers=True,
            promo_marker_mode="manual",
            promo_markers=selected_promos,
            seed=124,
        )

        game = configuration.create_game()

        self.assertIs(game.configuration, configuration)
        self.assertTrue(game.use_mission_cards)
        self.assertEqual(game.tile_pool, list(selected_tiles))
        self.assertEqual(
            tuple(player.control for player in game.players),
            controls,
        )
        self.assertIsNone(game.players[0].ai_top_k)
        self.assertEqual(game.players[1].ai_top_k, 15)
        self.assertEqual(game.players[2].ai_top_k, 1)
        marker_pool = game.selected_map.bonus_marker_pool
        self.assertEqual(len(marker_pool), 12)
        for marker in selected_promos:
            self.assertEqual(marker_pool.count(marker), 1)

    def test_seeded_random_optional_pools_are_reproducible_and_legal(self):
        configuration = GameConfiguration(
            map_num=3,
            player_count=5,
            player_controls=(PlayerControl.HUMAN,) * 5,
            use_emperors_favour=True,
            use_promo_markers=True,
            seed=124,
        )

        first = configuration.create_game()
        second = configuration.create_game()

        self.assertEqual(first.tile_pool, second.tile_pool)
        self.assertEqual(len(first.tile_pool), 5)
        self.assertEqual(len(set(first.tile_pool)), 5)
        self.assertEqual(
            first.selected_map.bonus_marker_pool,
            second.selected_map.bonus_marker_pool,
        )
        self.assertEqual(len(first.selected_map.bonus_marker_pool), 12)
        self.assertTrue(
            set(first.selected_map.bonus_marker_pool).intersection(Map.PROMO_BONUS_MARKERS)
        )

    def test_difficulty_selection_respects_configured_top_k(self):
        ranked = [(index, float(20 - index)) for index in range(20)]
        rng = random.Random(124)

        self.assertEqual(
            choose_ranked_ai_action(ranked, PlayerControl.MAGNUS, rng),
            0,
        )
        choices = {
            choose_ranked_ai_action(
                ranked,
                PlayerControl.HARD,
                rng,
                thresholds={
                    PlayerControl.EASY: 15,
                    PlayerControl.MEDIUM: 10,
                    PlayerControl.HARD: 3,
                    PlayerControl.MAGNUS: 1,
                },
            )
            for _ in range(30)
        }
        self.assertTrue(choices)
        self.assertTrue(choices.issubset({0, 1, 2}))

    def test_compatibility_launcher_has_no_import_time_game_loop(self):
        launcher = importlib.import_module("sample_hansa_game")
        self.assertTrue(callable(launcher.main))

    def test_primary_launcher_creates_game_from_menu_configuration(self):
        configuration = mock.Mock()
        game = mock.Mock()
        configuration.create_game.return_value = game
        window = mock.Mock()

        with (
            mock.patch.object(hansa_game, "run_new_game_menu", return_value=configuration),
            mock.patch.object(hansa_game, "GameWindow", return_value=window),
        ):
            self.assertEqual(hansa_game.main(), 0)

        configuration.create_game.assert_called_once_with()
        window.run.assert_called_once_with()

    def test_game_window_mouse_mapping_submits_only_legal_actions(self):
        game = GameConfiguration(map_num=2, seed=124).create_game()
        window = GameWindow.__new__(GameWindow)
        window.game = game
        window.action_rects = []
        legal_actions = game.legal_action_mask().nonzero(as_tuple=True)[0].tolist()
        target_action = next(action for action in legal_actions if action < 121)
        target_post_index = target_action
        posts = [post for route in game.selected_map.routes for post in route.posts]
        target_post = posts[target_post_index]

        self.assertEqual(
            window.action_for_click(
                target_post.pos,
                1,
                legal_actions,
            ),
            target_action,
        )
        self.assertIsNone(
            window.action_for_click(
                target_post.pos,
                1,
                [],
            )
        )


if __name__ == "__main__":
    unittest.main()
