import contextlib
import io
import unittest

from game.game_runner import create_headless_game
from map_data.constants import UPGRADE_MAX_VALUES
from map_data.map_attributes import BonusMarker


class FinalScoringTests(unittest.TestCase):
    def game(self, map_num=1):
        return create_headless_game(map_num, 3, seed=124)

    def finalize(self, game):
        with contextlib.redirect_stdout(io.StringIO()):
            game.finalize_end_of_game_points()

    def test_bonus_marker_scoring_all_boundaries(self):
        game = self.game()
        expected = {
            0: 0,
            1: 1,
            2: 3,
            3: 3,
            4: 6,
            5: 6,
            6: 10,
            7: 10,
            8: 15,
            9: 15,
            10: 21,
            15: 21,
        }
        self.assertEqual(
            {count: game.get_bonus_marker_points(count) for count in expected},
            expected,
        )

    def test_only_four_non_key_abilities_score_when_fully_developed(self):
        game = self.game()
        player = game.players[0]
        for ability, maximum in UPGRADE_MAX_VALUES.items():
            setattr(player, ability, maximum)

        self.finalize(game)

        self.assertEqual(player.final_score_breakdown["Ability Points"], 16)

    def test_pending_replacement_does_not_count_as_collected_marker(self):
        game = self.game()
        player = game.players[0]
        player.bonus_markers = [BonusMarker("Move3")]
        game.pending_bonus_markers = ["SwapOffice", "3Actions"]
        game.replace_bonus_marker = 2

        self.finalize(game)

        self.assertEqual(player.final_score_breakdown["Bonus Marker Points"], 1)

    def test_special_prestige_city_control_and_network_score_exact_values(self):
        game = self.game()
        player, opponent = game.players[:2]
        first = game.selected_map.cities[0]
        second = next(
            city
            for route in first.routes
            for city in route.cities
            if city is not first
        )
        first.offices[0].controller = player
        if len(first.offices) > 1:
            first.offices[1].controller = player
        second.offices[0].controller = player
        player.keys = 3

        unrelated = next(city for city in game.selected_map.cities if city not in (first, second))
        unrelated.offices[0].controller = opponent

        prestige = game.selected_map.specialprestigepoints
        prestige.circle_data[0]["owner"] = player
        expected_special = prestige.circle_data[0]["value"]

        self.finalize(game)

        office_count = sum(
            office.controller is player
            for city in (first, second)
            for office in city.offices
        )
        self.assertEqual(player.final_score_breakdown["Special Prestige Points"], expected_special)
        self.assertEqual(player.final_score_breakdown["City Control Points"], 4)
        self.assertEqual(
            player.final_score_breakdown["Largest Network Points"],
            office_count * 3,
        )

    def test_empty_supply_only_ends_after_failed_required_draw(self):
        game = self.game()
        game.selected_map.bonus_marker_pool.clear()

        game.check_for_game_end()
        self.assertFalse(game.game_end)

        game.bonus_pool_exhausted_during_claim = True
        game.check_for_game_end()
        self.assertTrue(game.game_end)
        self.assertEqual(game.current_player.actions_remaining, 0)

    def test_tenth_or_britannia_eighth_completed_city_ends_game(self):
        for map_num, expected in ((1, 10), (3, 8)):
            with self.subTest(map=map_num):
                game = self.game(map_num)
                player = game.current_player
                for city in game.selected_map.cities[:expected]:
                    for office in city.offices:
                        office.controller = player
                game.check_for_game_end()
                self.assertTrue(game.game_end)
                self.assertGreaterEqual(game.current_full_cities_count, expected)

    def test_tie_breaks_by_least_actions_then_network_then_shared_win(self):
        game = self.game()
        first, second, third = game.players
        for player in game.players:
            player.final_score = 50
            player.final_score_breakdown = {"Largest Network Points": 12}

        first.actions_index = 2
        second.actions_index = 1
        third.actions_index = 1
        second.final_score_breakdown["Largest Network Points"] = 15

        self.assertEqual(game.end_the_game(), [second])

        third.final_score_breakdown["Largest Network Points"] = 15
        self.assertEqual(game.end_the_game(), [second, third])

    def test_finalization_is_idempotent(self):
        game = self.game()
        game.players[0].score = 7
        self.finalize(game)
        first_scores = [player.final_score for player in game.players]
        self.finalize(game)
        self.assertEqual(
            [player.final_score for player in game.players],
            first_scores,
        )


if __name__ == "__main__":
    unittest.main()
