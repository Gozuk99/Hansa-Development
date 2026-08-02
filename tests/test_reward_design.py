import unittest

from game.game_runner import create_headless_game
from game.structured_actions import IncomeInteraction
from map_data.constants import UPGRADE_MAX_VALUES
from map_data.map_attributes import BonusMarker


class RewardDesignTests(unittest.TestCase):
    def game(self, map_num=1):
        return create_headless_game(map_num, 3, seed=124)

    def test_projection_and_final_scoring_use_the_same_categories(self):
        game = self.game()
        player = game.players[0]
        player.score = 7
        player.bank = UPGRADE_MAX_VALUES["bank"]
        player.bonus_markers = [BonusMarker("Move3")]
        game.selected_map.cities[0].offices[0].controller = player

        projected = game.projected_scores()
        game.finalize_end_of_game_points()

        self.assertEqual(projected, tuple(item.final_score for item in game.players))
        self.assertEqual(player.final_score, sum(player.final_score_breakdown.values()))

    def test_actual_prestige_and_completed_ability_change_projection_exactly(self):
        game = self.game()
        player = game.players[0]
        baseline = game.projected_scores()[0]

        player.score += 1
        self.assertEqual(game.projected_scores()[0] - baseline, 1)

        after_point = game.projected_scores()[0]
        player.bank = UPGRADE_MAX_VALUES["bank"]
        self.assertEqual(game.projected_scores()[0] - after_point, 4)

        game.SevenPtsPerCompletedAbilityOwner = player
        self.assertEqual(game.projected_scores()[0] - after_point, 7)

    def test_city_control_bonus_markers_prestige_and_network_are_projected(self):
        game = self.game()
        player, opponent = game.players[:2]
        city = game.selected_map.cities[0]
        baseline = game.projected_score_breakdown(player)

        city.offices[0].controller = player
        controlled = game.projected_score_breakdown(player)
        self.assertEqual(controlled["City Control Points"] - baseline["City Control Points"], 2)
        self.assertGreater(
            controlled["Largest Network Points"] - baseline["Largest Network Points"], 0
        )

        city.offices[0].controller = opponent
        lost = game.projected_score_breakdown(player)
        self.assertEqual(lost["City Control Points"] - controlled["City Control Points"], -2)

        player.bonus_markers.append(BonusMarker("Move3"))
        one_marker = game.projected_score_breakdown(player)
        self.assertEqual(one_marker["Bonus Marker Points"], 1)
        player.bonus_markers.append(BonusMarker("SwapOffice"))
        two_markers = game.projected_score_breakdown(player)
        self.assertEqual(two_markers["Bonus Marker Points"] - one_marker["Bonus Marker Points"], 2)

        prestige = game.selected_map.specialprestigepoints
        printed_value = prestige.circle_data[0]["value"]
        prestige.circle_data[0]["owner"] = player
        self.assertEqual(
            game.projected_score_breakdown(player)["Special Prestige Points"],
            printed_value,
        )

    def test_emperors_favour_projected_scoring_uses_authoritative_values(self):
        game = self.game()
        player = game.players[0]
        player.bank = UPGRADE_MAX_VALUES["bank"]
        city = game.selected_map.cities[0]
        city.offices[0].controller = player
        baseline = game.projected_score_breakdown(player)

        game.SevenPtsPerCompletedAbilityOwner = player
        game.FourPtsPerOwnedCityOwner = player
        enhanced = game.projected_score_breakdown(player)

        self.assertEqual(enhanced["Ability Points"] - baseline["Ability Points"], 3)
        self.assertEqual(enhanced["City Control Points"] - baseline["City Control Points"], 2)

    def test_income_is_neutral_when_projected_score_does_not_change(self):
        game = self.game()
        income = next(
            action for action in game.get_legal_actions() if isinstance(action, IncomeInteraction)
        )
        before = game.projected_scores()[game.current_player_index]

        game.apply_structured_action(income)

        self.assertEqual(game.projected_scores()[game.current_player_index], before)

    def test_new_players_have_no_mutable_legacy_reward_state(self):
        game = self.game()

        self.assertTrue(all(not hasattr(player, "reward") for player in game.players))
        self.assertTrue(all(not hasattr(player, "reward_structure") for player in game.players))


if __name__ == "__main__":
    unittest.main()
