import contextlib
import io
import unittest

from game.game_runner import create_headless_game
from map_data.constants import (
    ACTIONS_MAX_VALUES,
    BANK_MAX_VALUES,
    BOOK_OF_KNOWLEDGE_MAX_VALUES,
    CITY_KEYS_MAX_VALUES,
    PRIVILEGE_COLORS,
)


class PlayerAbilityTests(unittest.TestCase):
    def make_player(self):
        return create_headless_game(2, 3, seed=124).current_player

    def upgrade(self, player, ability):
        with contextlib.redirect_stdout(io.StringIO()):
            return player.perform_upgrade(ability)

    def test_rulebook_tracks_and_starting_values(self):
        self.assertEqual(CITY_KEYS_MAX_VALUES, [1, 2, 2, 3, 4])
        self.assertEqual(ACTIONS_MAX_VALUES, [2, 3, 3, 4, 4, 5])
        self.assertEqual(PRIVILEGE_COLORS, ["WHITE", "ORANGE", "PINK", "BLACK"])
        self.assertEqual(BOOK_OF_KNOWLEDGE_MAX_VALUES, [2, 3, 4, 5])
        self.assertEqual(BANK_MAX_VALUES, [3, 4, 7, 50])

        for num_players in range(3, 6):
            game = create_headless_game(2, num_players, seed=124)
            for player in game.players:
                self.assertEqual(player.keys, 1)
                self.assertEqual(player.keys_index, 0)
                self.assertEqual(player.actions, 2)
                self.assertEqual(player.actions_index, 0)
                self.assertEqual(player.actions_remaining, 2)
                self.assertEqual(player.privilege, "WHITE")
                self.assertEqual(player.book, 2)
                self.assertEqual(player.bank, 3)

    def test_keys_progression_releases_one_trader_per_space(self):
        player = self.make_player()
        starting_squares = player.personal_supply_squares

        for index, expected_value in enumerate(CITY_KEYS_MAX_VALUES[1:], start=1):
            self.assertTrue(self.upgrade(player, "keys"))
            self.assertEqual(player.keys_index, index)
            self.assertEqual(player.keys, expected_value)
            self.assertEqual(player.personal_supply_squares, starting_squares + index)
            self.assertEqual(player.locked_ability_traders, 15 - index)

        self.assertFalse(self.upgrade(player, "keys"))
        self.assertEqual(player.personal_supply_squares, starting_squares + 4)
        self.assertTrue(all(player.has_unlocked_key(index) for index in range(5)))

    def test_actions_progression_and_immediate_extra_actions(self):
        player = self.make_player()
        starting_squares = player.personal_supply_squares
        expected_remaining = 2

        for index, expected_value in enumerate(ACTIONS_MAX_VALUES[1:], start=1):
            previous_value = player.actions
            self.assertTrue(self.upgrade(player, "actions"))
            if expected_value > previous_value:
                expected_remaining += 1
            self.assertEqual(player.actions_index, index)
            self.assertEqual(player.actions, expected_value)
            self.assertEqual(player.actions_remaining, expected_remaining)
            self.assertEqual(player.personal_supply_squares, starting_squares + index)

        self.assertFalse(self.upgrade(player, "actions"))
        self.assertEqual(player.actions_remaining, 5)
        self.assertEqual(player.personal_supply_squares, starting_squares + 5)
        self.assertTrue(all(player.has_unlocked_action(index) for index in range(6)))

    def test_privilege_progression_releases_one_trader_per_space(self):
        player = self.make_player()
        starting_squares = player.personal_supply_squares

        for index, expected_color in enumerate(PRIVILEGE_COLORS[1:], start=1):
            self.assertTrue(self.upgrade(player, "privilege"))
            self.assertEqual(player.privilege, expected_color)
            self.assertEqual(player.personal_supply_squares, starting_squares + index)

        self.assertFalse(self.upgrade(player, "privilege"))
        self.assertEqual(player.personal_supply_squares, starting_squares + 3)
        self.assertTrue(all(player.has_unlocked_privilege(index) for index in range(4)))

    def test_book_progression_releases_one_merchant_per_space(self):
        player = self.make_player()
        starting_circles = player.personal_supply_circles

        for index, expected_value in enumerate(
            BOOK_OF_KNOWLEDGE_MAX_VALUES[1:], start=1
        ):
            self.assertTrue(self.upgrade(player, "book"))
            self.assertEqual(player.book, expected_value)
            self.assertEqual(player.personal_supply_circles, starting_circles + index)
            self.assertEqual(player.locked_ability_merchants, 3 - index)

        self.assertFalse(self.upgrade(player, "book"))
        self.assertEqual(player.personal_supply_circles, starting_circles + 3)
        self.assertTrue(all(player.has_unlocked_book(index) for index in range(4)))

    def test_bank_progression_releases_one_trader_per_space(self):
        player = self.make_player()
        starting_squares = player.personal_supply_squares

        for index, expected_value in enumerate(BANK_MAX_VALUES[1:], start=1):
            self.assertTrue(self.upgrade(player, "bank"))
            self.assertEqual(player.bank, expected_value)
            self.assertEqual(player.personal_supply_squares, starting_squares + index)

        self.assertFalse(self.upgrade(player, "bank"))
        self.assertEqual(player.personal_supply_squares, starting_squares + 3)
        self.assertTrue(all(player.has_unlocked_bank(index) for index in range(4)))


if __name__ == "__main__":
    unittest.main()
