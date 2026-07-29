import importlib
import random
import subprocess
import sys
import unittest
from unittest import mock

import hansa_game
from drawing.game_window import GameWindow, action_label
from drawing.new_game_menu import NewGameMenu, NewGameMenuState
from drawing.scaled_display import ScaledDisplay
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

    def test_stale_inactive_player_callback_is_ignored(self):
        state = NewGameMenuState()
        state.set_player_count(5)
        menu = object.__new__(NewGameMenu)
        menu.state = state

        state.set_player_count(3)
        menu._set_control(4, PlayerControl.MAGNUS)

        self.assertEqual(state.player_controls, [PlayerControl.HUMAN] * 3)

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
                promo_markers=("ExchangeBonusMarker",) * 3
                + ("PlaceAdjacent",) * 3
                + ("SwapOffice",) * 2
                + ("Move3",)
                + ("UpgradeAbility",) * 2
                + ("3Actions",),
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
        selected_supply = (
            selected_promos
            + ("PlaceAdjacent",) * 3
            + ("SwapOffice",) * 2
            + ("Move3",)
            + ("UpgradeAbility",) * 2
            + ("3Actions",)
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
            promo_markers=selected_supply,
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
        self.assertCountEqual(marker_pool, selected_supply)

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

    def test_primary_launcher_import_does_not_initialize_pygame(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import pygame; assert not pygame.get_init(); "
                "import hansa_game; assert not pygame.get_init()",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_primary_launcher_creates_game_from_menu_configuration(self):
        configuration = mock.Mock()
        game = mock.Mock()
        configuration.create_game.return_value = game
        window = mock.Mock()

        with (
            mock.patch(
                "drawing.new_game_menu.run_new_game_menu",
                return_value=configuration,
            ),
            mock.patch("drawing.game_window.GameWindow", return_value=window),
            mock.patch("pygame.quit") as quit_pygame,
        ):
            self.assertEqual(hansa_game.main(), 0)

        configuration.create_game.assert_called_once_with()
        window.run.assert_called_once_with()
        quit_pygame.assert_called_once_with()

    def test_scaled_display_fits_large_logical_canvas_inside_viewport(self):
        self.assertEqual(
            ScaledDisplay.fit_size((980, 900), (1280, 620)),
            (675, 620),
        )
        game_size = ScaledDisplay.fit_size((2600, 1370), (1840, 980))
        self.assertLessEqual(game_size[0], 1840)
        self.assertLessEqual(game_size[1], 980)

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

    def test_game_window_city_click_maps_to_legal_route_outcome(self):
        game = GameConfiguration(map_num=1, seed=124).create_game()
        player = game.current_player
        route = game.selected_map.routes[0]
        for post in route.posts:
            post.owner = player
            post.owner_piece_shape = "square"

        legal_actions = game.legal_action_mask().nonzero(as_tuple=True)[0].tolist()
        points_action = 242
        city = route.cities[0]
        center = (city.x_pos + city.width // 2, city.y_pos + city.height // 2)
        window = GameWindow.__new__(GameWindow)
        window.game = game
        window.action_rects = []

        self.assertIn(points_action, legal_actions)
        self.assertEqual(
            window.action_for_click(center, 3, legal_actions),
            points_action,
        )

    def test_game_window_labels_and_maps_bonus_marker_replacement_route(self):
        game = GameConfiguration(map_num=1, seed=124).create_game()
        route = game.selected_map.routes[0]
        window = GameWindow.__new__(GameWindow)
        window.game = game
        window.action_rects = []

        self.assertEqual(
            action_label(543, game),
            f"Place marker: {route.cities[0].name}—{route.cities[1].name}",
        )
        self.assertEqual(action_label(527, game), "Use Swap Office")
        self.assertEqual(
            window.action_for_click(route.posts[0].pos, 1, [543]),
            543,
        )

    def test_middle_click_zones_expose_each_city_upgrade(self):
        game = GameConfiguration(map_num=2, seed=124).create_game()
        city = next(city for city in game.selected_map.cities if city.name == "Waren")
        route = city.routes[0]
        for post in route.posts:
            post.owner = game.current_player
            post.owner_piece_shape = "square"
        route_index = game.selected_map.routes.index(route)
        base = 242 + 120 + route_index * 4
        legal_actions = game.legal_action_mask().nonzero(as_tuple=True)[0].tolist()
        choices = [action for action in range(base, base + 4) if action in legal_actions]
        window = GameWindow.__new__(GameWindow)
        window.game = game
        window.action_rects = []

        left = (city.x_pos + 1, city.y_pos + city.height // 2)
        right = (city.x_pos + city.width - 1, city.y_pos + city.height // 2)

        self.assertEqual(len(choices), 2)
        self.assertEqual(window.action_for_click(left, 2, legal_actions), choices[0])
        self.assertEqual(window.action_for_click(right, 2, legal_actions), choices[1])

    def test_special_prestige_city_zones_expose_all_four_values(self):
        game = GameConfiguration(map_num=1, seed=124).create_game()
        city = next(
            city
            for city in game.selected_map.cities
            if "SpecialPrestigePoints" in city.upgrade_city_type
        )
        route = city.routes[0]
        for post in route.posts:
            post.owner = game.current_player
            post.owner_piece_shape = "circle"
        route_index = game.selected_map.routes.index(route)
        base = 242 + 120 + route_index * 4
        choices = list(range(base, base + 4))
        window = GameWindow.__new__(GameWindow)
        window.game = game
        window.action_rects = []
        positions = [
            (
                city.x_pos + int((index + 0.5) * city.width / 4),
                city.y_pos + city.height // 2,
            )
            for index in range(4)
        ]

        self.assertEqual(
            [window.action_for_click(position, 2, choices) for position in positions],
            choices,
        )

    def test_additional_office_route_choices_have_contextual_labels(self):
        game = GameConfiguration(map_num=1, seed=124).create_game()
        route = game.selected_map.routes[0]
        game.waiting_for_bm_place_adjacent = True
        base = 242 + 120

        self.assertEqual(
            action_label(base, game),
            f"Additional Trader office in {route.cities[0].name}",
        )
        self.assertEqual(
            action_label(base + 1, game),
            f"Additional Merchant office in {route.cities[0].name}",
        )


if __name__ == "__main__":
    unittest.main()
