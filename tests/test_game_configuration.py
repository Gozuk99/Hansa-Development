import importlib
import random
import subprocess
import sys
import unittest
from unittest import mock

import hansa_game
from drawing.action_ui import action_label
from drawing.game_window import GameWindow
from drawing.new_game_menu import NewGameMenu, NewGameMenuState
from drawing.scaled_display import ScaledDisplay
from game.game_config import (
    EMPERORS_FAVOUR_TILES,
    GameConfiguration,
    PlayerControl,
    choose_ranked_ai_action,
)
from map_data.constants import MAX_POSTS
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

    def test_menu_requests_double_size_without_changing_logical_coordinates(self):
        requested = ScaledDisplay.fit_size((980 * 2, 940 * 2), (3840, 2160))

        self.assertEqual(requested, (1960, 1880))

    def test_presentation_canvas_scales_up_to_fill_enlarged_window(self):
        presentation = ScaledDisplay.fit_size(
            (980, 940),
            (1960, 1880),
            allow_upscale=True,
        )

        self.assertEqual(presentation, (1960, 1880))

    def test_tiny_display_never_requests_a_window_larger_than_desktop(self):
        available = ScaledDisplay.available_size((600, 400), (980, 940))
        requested = ScaledDisplay.fit_size((980 * 2, 940 * 2), available)

        self.assertLessEqual(requested[0], 600)
        self.assertLessEqual(requested[1], 400)

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
                2,
                legal_actions,
            )
        )
        self.assertIsNone(
            window.action_for_click(
                target_post.pos,
                1,
                [],
            )
        )

    def test_zero_stock_displaced_merchant_is_right_clickable_then_may_finish(self):
        game = GameConfiguration(map_num=2, seed=124).create_game()
        actor, opponent = game.players[:2]
        route = next(
            candidate
            for candidate in game.selected_map.routes
            if {city.name for city in candidate.cities} == {"Malmo", "Visby"}
        )
        displaced_post = next(post for post in route.posts if post.required_shape is None)
        displaced_post.claim(opponent, "circle")
        opponent.general_stock_circles = 0
        opponent.general_stock_squares = 2
        posts = [post for candidate in game.selected_map.routes for post in candidate.posts]
        displaced_index = posts.index(displaced_post)
        game.apply_action(displaced_index)
        legal_actions = game.legal_action_mask().nonzero(as_tuple=True)[0].tolist()
        circle_action = next(
            action for action in legal_actions if MAX_POSTS <= action < MAX_POSTS * 2
        )
        target = posts[circle_action - MAX_POSTS]
        window = GameWindow.__new__(GameWindow)
        window.game = game
        window.action_rects = []

        self.assertEqual(
            window.action_for_click(target.pos, 3, legal_actions),
            circle_action,
        )
        self.assertIsNone(window.action_for_click(target.pos, 1, legal_actions))

        game.apply_action(circle_action)
        updated_actions = game.legal_action_mask().nonzero(as_tuple=True)[0].tolist()
        self.assertFalse(any(MAX_POSTS <= action < MAX_POSTS * 2 for action in updated_actions))
        square_action = next(action for action in updated_actions if action < MAX_POSTS)
        square_target = posts[square_action]
        self.assertEqual(
            window.action_for_click(square_target.pos, 1, updated_actions),
            square_action,
        )
        self.assertIsNone(window.action_for_click(square_target.pos, 3, updated_actions))
        self.assertIn(618, updated_actions)
        self.assertEqual(
            action_label(618, game),
            "Finish displacement (decline optional pieces)",
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

    def test_left_click_on_each_waren_upgrade_box_selects_that_upgrade(self):
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

        upgrade_boxes = [
            upgrade for upgrade in game.selected_map.upgrade_cities if upgrade.city_name == "Waren"
        ]
        self.assertEqual(len(choices), 2)
        city_index = route.cities.index(city)
        for upgrade in upgrade_boxes:
            with self.subTest(upgrade=upgrade.upgrade_type):
                center = (
                    upgrade.x_pos + upgrade.width // 2,
                    upgrade.y_pos + upgrade.height // 2,
                )
                expected = (
                    base + city_index * 2 + city.upgrade_city_type.index(upgrade.upgrade_type)
                )
                self.assertEqual(
                    window.action_for_click(center, 1, legal_actions),
                    expected,
                )
        city_center = (
            city.x_pos + city.width // 2,
            city.y_pos + city.height // 2,
        )
        self.assertIsNone(window.action_for_click(city_center, 2, legal_actions))

    def test_every_legal_adjacent_route_is_reachable_from_upgrade_box(self):
        for map_num in (1, 2, 3):
            game = GameConfiguration(map_num=map_num, seed=124).create_game()
            for upgrade in game.selected_map.upgrade_cities:
                with self.subTest(map_num=map_num, upgrade=upgrade.upgrade_type):
                    city = next(
                        candidate
                        for candidate in game.selected_map.cities
                        if candidate.name == upgrade.city_name
                    )
                    for route in city.routes:
                        for post in route.posts:
                            post.owner = game.current_player
                            post.owner_piece_shape = "square"
                    legal_actions = game.legal_action_mask().nonzero(as_tuple=True)[0].tolist()
                    upgrade_index = city.upgrade_city_type.index(upgrade.upgrade_type)
                    expected = {
                        242
                        + 120
                        + game.selected_map.routes.index(route) * 4
                        + route.cities.index(city) * 2
                        + upgrade_index
                        for route in city.routes
                    }
                    expected.intersection_update(legal_actions)
                    window = GameWindow.__new__(GameWindow)
                    window.game = game
                    window.action_rects = []
                    center_x = int(upgrade.x_pos + upgrade.width / 2)
                    center_y = int(upgrade.y_pos + upgrade.height / 2)
                    positions = [
                        *(
                            (x, center_y)
                            for x in range(
                                int(upgrade.x_pos),
                                int(upgrade.x_pos + upgrade.width),
                            )
                        ),
                        *(
                            (center_x, y)
                            for y in range(
                                int(upgrade.y_pos),
                                int(upgrade.y_pos + upgrade.height),
                            )
                        ),
                    ]
                    reached = {
                        action
                        for position in positions
                        if (
                            action := window.action_for_click(
                                position,
                                1,
                                legal_actions,
                            )
                        )
                        is not None
                    }
                    self.assertTrue(expected)
                    self.assertTrue(expected.issubset(reached))

    def test_left_click_on_special_prestige_box_exposes_all_four_values(self):
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
        prestige = game.selected_map.specialprestigepoints
        positions = [
            (
                prestige.x_pos + int((index + 0.5) * prestige.width / 4),
                prestige.y_pos + prestige.height // 2,
            )
            for index in range(4)
        ]

        self.assertEqual(
            [window.action_for_click(position, 1, choices) for position in positions],
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
