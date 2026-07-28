import random
import unittest

from game.game_runner import create_headless_game
from game.setup import starting_inventory
from map_data.constants import BANK_MAX_VALUES


class GameSetupTests(unittest.TestCase):
    def test_headless_game_does_not_create_models(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        self.assertTrue(all(player.hansa_nn is None for player in game.players))

    def test_seeded_setup_does_not_change_global_random_state(self):
        random.seed(99)
        expected = random.random()
        random.seed(99)
        create_headless_game(map_num=2, num_players=3, seed=124)
        self.assertEqual(random.random(), expected)

    def test_same_seed_has_same_setup(self):
        for map_num in range(1, 4):
            options = {
                "use_mission_cards": map_num == 1,
                "use_emperors_favour": True,
            }
            first = create_headless_game(
                map_num=map_num, num_players=5, seed=124, **options
            )
            second = create_headless_game(
                map_num=map_num, num_players=5, seed=124, **options
            )
            self.assertEqual(first.tile_pool, second.tile_pool)
            self.assertEqual(
                [route.bonus_marker.type if route.bonus_marker else None for route in first.selected_map.routes],
                [route.bonus_marker.type if route.bonus_marker else None for route in second.selected_map.routes],
            )
            self.assertEqual(
                [player.mission_card for player in first.players],
                [player.mission_card for player in second.players],
            )

    def test_bank_track_uses_complete_rulebook_values(self):
        self.assertEqual(BANK_MAX_VALUES, [3, 4, 7, 50])
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        player = game.current_player
        for expected in (4, 7, 50):
            player.upgrade_bank()
            self.assertEqual(player.bank, expected)
        self.assertTrue(all(player.has_unlocked_bank(index) for index in range(4)))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            create_headless_game(map_num=99, num_players=3)
        with self.assertRaises(ValueError):
            create_headless_game(map_num=2, num_players=2)

    def test_starting_inventory_matches_player_order(self):
        expected = {
            1: (5, 1, 6, 0),
            2: (6, 1, 5, 0),
            3: (7, 1, 4, 0),
            4: (8, 1, 3, 0),
            5: (9, 1, 2, 0),
        }
        for player_order, counts in expected.items():
            with self.subTest(player_order=player_order):
                inventory = starting_inventory(player_order)
                self.assertEqual(
                    (
                        inventory.personal_supply_squares,
                        inventory.personal_supply_circles,
                        inventory.general_stock_squares,
                        inventory.general_stock_circles,
                    ),
                    counts,
                )
                self.assertEqual(inventory.total_squares, 11)
                self.assertEqual(inventory.total_circles, 1)

    def test_every_supported_player_count_uses_exact_starting_inventory(self):
        for num_players in range(3, 6):
            with self.subTest(num_players=num_players):
                game = create_headless_game(map_num=2, num_players=num_players, seed=124)
                for player in game.players:
                    expected = starting_inventory(player.order)
                    self.assertEqual(player.personal_supply_squares, expected.personal_supply_squares)
                    self.assertEqual(player.personal_supply_circles, expected.personal_supply_circles)
                    self.assertEqual(player.general_stock_squares, expected.general_stock_squares)
                    self.assertEqual(player.general_stock_circles, expected.general_stock_circles)

    def test_writing_desk_and_available_pieces_account_for_full_component_set(self):
        for num_players in range(3, 6):
            game = create_headless_game(map_num=2, num_players=num_players, seed=124)
            for player in game.players:
                available_traders = (
                    player.personal_supply_squares + player.general_stock_squares
                )
                available_merchants = (
                    player.personal_supply_circles + player.general_stock_circles
                )
                self.assertEqual(player.locked_ability_traders, 15)
                self.assertEqual(player.locked_ability_merchants, 3)
                # One additional trader represents the player's score-track marker.
                self.assertEqual(available_traders + player.locked_ability_traders + 1, 27)
                self.assertEqual(
                    available_merchants + player.locked_ability_merchants,
                    4,
                )

    def test_every_supported_map_and_player_count_constructs(self):
        for map_num in range(1, 4):
            for num_players in range(3, 6):
                with self.subTest(map_num=map_num, num_players=num_players):
                    game = create_headless_game(map_num, num_players, seed=124)
                    self.assertEqual(len(game.players), num_players)
                    self.assertEqual(game.current_player_index, 0)
                    self.assertEqual(game.active_player, 0)
                    self.assertEqual(game.turn_number, 1)
                    self.assertEqual(game.round_number, 1)

    def test_bonus_marker_setup_has_three_starting_and_twelve_in_supply(self):
        expected_starting_types = {"Move3", "SwapOffice", "PlaceAdjacent"}
        for map_num in range(1, 4):
            game = create_headless_game(map_num, 3, seed=124)
            starting_markers = [
                route.bonus_marker.type
                for route in game.selected_map.routes
                if route.bonus_marker is not None
            ]
            self.assertEqual(len(starting_markers), 3)
            self.assertEqual(set(starting_markers), expected_starting_types)
            self.assertEqual(len(game.selected_map.bonus_marker_pool), 12)
            for route in game.selected_map.routes:
                if route.bonus_marker is not None:
                    self.assertTrue(route.has_bonus_marker)
                    self.assertIsNone(route.permanent_bonus_marker)
                    self.assertIsNone(route.region)

    def test_tile_pool_matches_player_count_without_duplicates(self):
        expected_tiles = {
            "DisplaceAnywhere",
            "+1Action",
            "+1IncomeIfOthersIncome",
            "+1DisplacedPiece",
            "+4PtsPerOwnedCity",
            "+7PtsPerCompletedAbility",
        }
        for map_num in range(1, 4):
            for num_players in range(3, 6):
                game = create_headless_game(map_num, num_players, seed=124)
                self.assertEqual(game.tile_pool, [])
                game = create_headless_game(
                    map_num,
                    num_players,
                    seed=124,
                    use_emperors_favour=True,
                )
                self.assertEqual(len(game.tile_pool), num_players)
                self.assertEqual(len(set(game.tile_pool)), num_players)
                self.assertTrue(set(game.tile_pool).issubset(expected_tiles))

    def test_mission_cards_are_only_assigned_on_map_one(self):
        disabled = create_headless_game(1, 5, seed=124)
        self.assertTrue(all(player.mission_card is None for player in disabled.players))

        for num_players in range(3, 6):
            map_one = create_headless_game(
                1,
                num_players,
                seed=124,
                use_mission_cards=True,
            )
            cards = [tuple(player.mission_card) for player in map_one.players]
            self.assertTrue(all(len(card) == 3 for card in cards))
            self.assertEqual(len(set(cards)), len(cards))
            self.assertEqual(len(map_one.selected_map.mission_cards), 9 - num_players)

        for map_num in (2, 3):
            game = create_headless_game(map_num, 5, seed=124)
            self.assertTrue(all(player.mission_card is None for player in game.players))
            with self.assertRaises(ValueError):
                create_headless_game(
                    map_num,
                    5,
                    seed=124,
                    use_mission_cards=True,
                )

    def test_optional_setup_modules_are_disabled_by_default(self):
        for map_num in range(1, 4):
            game = create_headless_game(map_num, 3, seed=124)
            self.assertFalse(game.use_mission_cards)
            self.assertFalse(game.use_emperors_favour)
            self.assertEqual(game.tile_pool, [])
            self.assertTrue(all(player.mission_card is None for player in game.players))

    def test_map_specific_end_targets_and_connection_endpoints(self):
        expected = {
            1: (10, {"Stendal", "Arnheim"}),
            2: (10, {"Lubeck", "Danzig"}),
            3: (8, {"York", "Oxford"}),
        }
        for map_num, (max_cities, endpoints) in expected.items():
            game = create_headless_game(map_num, 3, seed=124)
            self.assertEqual(game.selected_map.max_full_cities, max_cities)
            self.assertEqual(set(game.selected_map.east_west_cities), endpoints)

    def test_supported_maps_start_with_pristine_board_and_counters(self):
        for map_num in range(1, 4):
            for num_players in range(3, 6):
                game = create_headless_game(map_num, num_players, seed=124)
                self.assertEqual(game.current_full_cities_count, 0)
                self.assertEqual(game.east_west_completed_count, 0)
                self.assertEqual(game.players_who_completed_east_west, set())
                self.assertEqual(game.replace_bonus_marker, 0)
                self.assertFalse(game.game_end)
                self.assertTrue(all(player.score == 0 for player in game.players))
                self.assertTrue(
                    all(
                        post.owner is None and post.owner_piece_shape is None
                        for route in game.selected_map.routes
                        for post in route.posts
                    )
                )
                self.assertTrue(
                    all(
                        office.controller is None
                        for city in game.selected_map.cities
                        for office in city.offices
                    )
                )

    def test_map_two_permanent_bonus_markers_match_eastern_map(self):
        game = create_headless_game(2, 3, seed=124)
        actual = {
            tuple(city.name for city in route.cities): route.permanent_bonus_marker.type
            for route in game.selected_map.routes
            if route.permanent_bonus_marker is not None
        }
        self.assertEqual(
            actual,
            {
                ("Mismar", "Stralsund"): "MoveAny2",
                ("Stralsund", "Malmo"): "+1Priv",
                ("Malmo", "Visby"): "ClaimGreenCity",
                ("Malmo", "Danzig"): "Place2TradesmenFromRoute",
            },
        )

    def test_britannia_permanent_markers_match_player_count_map(self):
        expected_common = {
            ("Southhampton", "Calais"): "Place2ScotlandOrWales",
            ("Canterbury", "Calais"): "MoveAny2",
        }
        for num_players in range(3, 6):
            game = create_headless_game(3, num_players, seed=124)
            actual = {
                tuple(city.name for city in route.cities): route.permanent_bonus_marker.type
                for route in game.selected_map.routes
                if route.permanent_bonus_marker is not None
            }
            expected = dict(expected_common)
            if num_players == 3:
                expected[("Carlisle", "IsleOfMan")] = "MoveAny2"
            self.assertEqual(actual, expected)

    def test_player_count_selects_correct_map_topology(self):
        expected = {
            (1, 3): (27, 32),
            (1, 4): (27, 34),
            (1, 5): (27, 34),
            (2, 3): (28, 32),
            (2, 4): (28, 32),
            (2, 5): (28, 32),
            (3, 3): (26, 35),
            (3, 4): (30, 40),
            (3, 5): (30, 40),
        }
        for configuration, counts in expected.items():
            game = create_headless_game(*configuration, seed=124)
            self.assertEqual(
                (len(game.selected_map.cities), len(game.selected_map.routes)),
                counts,
            )


if __name__ == "__main__":
    unittest.main()
