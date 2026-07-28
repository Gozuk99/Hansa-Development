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
            first = create_headless_game(map_num=map_num, num_players=5, seed=124)
            second = create_headless_game(map_num=map_num, num_players=5, seed=124)
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

    def test_tile_pool_matches_player_count_without_duplicates(self):
        for num_players in range(3, 6):
            game = create_headless_game(2, num_players, seed=124)
            self.assertEqual(len(game.tile_pool), num_players)
            self.assertEqual(len(set(game.tile_pool)), num_players)

    def test_mission_cards_are_only_assigned_on_map_one(self):
        map_one = create_headless_game(1, 5, seed=124)
        cards = [tuple(player.mission_card) for player in map_one.players]
        self.assertTrue(all(len(card) == 3 for card in cards))
        self.assertEqual(len(set(cards)), len(cards))

        for map_num in (2, 3):
            game = create_headless_game(map_num, 5, seed=124)
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


if __name__ == "__main__":
    unittest.main()
