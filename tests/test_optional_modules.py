import contextlib
import io
import unittest

from ai.game_state import BoardData
from game.game_actions import buy_tile
from game.game_runner import create_headless_game
from map_data.map_attributes import BonusMarker
from player_info.player_attributes import UPGRADE_MAX_VALUES


TILE_ACTION_START = 535
TILES = (
    "DisplaceAnywhere",
    "+1Action",
    "+1IncomeIfOthersIncome",
    "+1DisplacedPiece",
    "+4PtsPerOwnedCity",
    "+7PtsPerCompletedAbility",
)
MARKER_TYPES = (
    "SwapOffice",
    "Move3",
    "UpgradeAbility",
    "3Actions",
    "4Actions",
    "ExchangeBonusMarker",
    "Tribute4EstablishingTP",
    "BlockTradeRoute",
)


class OptionalModuleTests(unittest.TestCase):
    def game(self, **options):
        return create_headless_game(
            map_num=1,
            num_players=3,
            seed=124,
            use_mission_cards=True,
            use_emperors_favour=True,
            **options,
        )

    def test_mission_scores_each_occupied_city_without_requiring_control(self):
        game = self.game()
        player, opponent = game.players[:2]
        cities = [
            next(city for city in game.selected_map.cities if city.name == name)
            for name in player.mission_card
        ]
        for city in cities:
            city.offices[0].controller = player
            if len(city.offices) > 1:
                city.offices[-1].controller = opponent

        self.assertEqual(game.get_mission_card_points(player), 3)

    def test_mission_scores_five_point_bonus_only_for_controlling_all_cities(self):
        game = self.game()
        player = game.players[0]
        cities = [
            next(city for city in game.selected_map.cities if city.name == name)
            for name in player.mission_card
        ]
        for city in cities:
            for office in city.offices:
                office.controller = player

        self.assertEqual(game.get_mission_card_points(player), 8)
        cities[0].offices[0].controller = player
        for office in cities[0].offices[1:]:
            office.controller = game.players[1]
        self.assertEqual(game.get_mission_card_points(player), 3)

    def test_pre_game_end_ai_observation_includes_own_mission_but_hides_opponents(self):
        game = self.game()
        board_data = BoardData()

        self.assertFalse(game.game_end)
        self.assertNotEqual(
            board_data.assign_mission_card_mapping(game, game.current_player),
            (0, 0, 0),
        )
        self.assertEqual(
            board_data.assign_mission_card_mapping(game, game.players[1]),
            (0, 0, 0),
        )

    def test_two_markers_buy_tile_and_forfeit_entire_turn(self):
        game = self.game()
        player = game.current_player
        game.tile_pool = [TILES[0]]
        markers = [BonusMarker("SwapOffice"), BonusMarker("Move3")]
        player.bonus_markers = markers.copy()

        with contextlib.redirect_stdout(io.StringIO()):
            game.apply_action(TILE_ACTION_START)

        self.assertEqual(player.tiles, [TILES[0]])
        self.assertEqual(player.bonus_markers, [])
        self.assertEqual(player.used_bonus_markers, markers)
        self.assertEqual(game.current_player, game.players[1])

    def test_more_than_two_markers_requires_two_explicit_distinct_payments(self):
        game = self.game()
        player = game.current_player
        game.tile_pool = [TILES[1]]
        markers = [
            BonusMarker("SwapOffice"),
            BonusMarker("SwapOffice"),
            BonusMarker("Move3"),
        ]
        player.bonus_markers = markers.copy()

        game.apply_action(TILE_ACTION_START + 1)
        self.assertTrue(game.waiting_for_buy_tile_with_bm)
        game.apply_action(TILE_ACTION_START)
        self.assertIs(game.first_bm_to_spend_on_tile, markers[0])
        self.assertEqual(game.legal_action_mask()[TILE_ACTION_START].item(), 1)

        with contextlib.redirect_stdout(io.StringIO()):
            game.apply_action(TILE_ACTION_START)

        self.assertEqual(player.used_bonus_markers, markers[:2])
        self.assertEqual(player.bonus_markers, [markers[2]])
        self.assertEqual(player.tiles, [TILES[1]])

    def test_tile_purchase_rejects_used_actions_and_invalid_payment(self):
        game = self.game()
        player = game.current_player
        game.tile_pool = [TILES[0]]
        first = BonusMarker("SwapOffice")
        second = BonusMarker("Move3")
        player.bonus_markers = [first, second]
        player.spend_action()

        with self.assertRaises(ValueError):
            buy_tile(game, TILES[0], first, second)

        player.start_turn()
        with self.assertRaises(ValueError):
            buy_tile(game, TILES[0], first, first)

    def test_four_permanent_gameplay_tiles_apply_their_effects(self):
        game = self.game()
        owner, other = game.players[:2]

        game.OneActionOwner = owner
        owner.start_turn(extra_actions=1)
        self.assertEqual(owner.actions_remaining, owner.actions + 1)

        game.OneIncomeIfOthersIncomeOwner = owner
        stock_before = owner.general_stock_circles + owner.general_stock_squares
        supply_before = owner.personal_supply_circles + owner.personal_supply_squares
        game.begin_income_favour_response(other)
        self.assertIs(game.pending_income_favour_owner, owner)
        game.apply_action(TILE_ACTION_START)
        self.assertEqual(
            owner.general_stock_circles + owner.general_stock_squares,
            stock_before - 1,
        )
        self.assertEqual(
            owner.personal_supply_circles + owner.personal_supply_squares,
            supply_before + 1,
        )

        game.OneDisplacedPieceOwner = owner
        game.displaced_player.populate_displaced_player(game, owner, "square")
        self.assertEqual(game.displaced_player.total_pieces_to_place, 3)

        game.DisplaceAnywhereOwner = owner
        self.assertIs(game.DisplaceAnywhereOwner, owner)

    def test_two_scoring_tiles_replace_four_point_values_with_seven_and_four(self):
        baseline = self.game()
        enhanced = self.game()
        baseline_owner = baseline.players[0]
        enhanced_owner = enhanced.players[0]
        enhanced.SevenPtsPerCompletedAbilityOwner = enhanced_owner
        enhanced.FourPtsPerOwnedCityOwner = enhanced_owner

        baseline_owner.book = UPGRADE_MAX_VALUES["book"]
        enhanced_owner.book = UPGRADE_MAX_VALUES["book"]
        baseline.selected_map.cities[0].offices[0].controller = baseline_owner
        enhanced.selected_map.cities[0].offices[0].controller = enhanced_owner

        with contextlib.redirect_stdout(io.StringIO()):
            baseline.finalize_end_of_game_points()
            enhanced.finalize_end_of_game_points()

        self.assertEqual(enhanced_owner.final_score, baseline_owner.final_score + 5)

    def test_income_favour_never_triggers_on_owners_own_income(self):
        game = self.game()
        owner = game.players[0]
        game.OneIncomeIfOthersIncomeOwner = owner
        before = (
            owner.general_stock_squares,
            owner.general_stock_circles,
            owner.personal_supply_squares,
            owner.personal_supply_circles,
        )

        game.begin_income_favour_response(owner)

        self.assertEqual(
            (
                owner.general_stock_squares,
                owner.general_stock_circles,
                owner.personal_supply_squares,
                owner.personal_supply_circles,
            ),
            before,
        )
        self.assertIsNone(game.pending_income_favour_owner)

    def test_income_favour_owner_may_choose_either_shape_or_decline(self):
        for action_offset, expected_shape in ((0, "square"), (1, "circle"), (2, None)):
            with self.subTest(choice=expected_shape):
                game = self.game()
                owner, other = game.players[:2]
                game.OneIncomeIfOthersIncomeOwner = owner
                owner.personal_supply_circles -= 1
                owner.general_stock_circles += 1
                before = {
                    "square": owner.personal_supply_squares,
                    "circle": owner.personal_supply_circles,
                }

                game.begin_income_favour_response(other)
                mask = game.legal_action_mask()
                self.assertEqual(mask[TILE_ACTION_START:TILE_ACTION_START + 3].tolist(), [1, 1, 1])
                game.apply_action(TILE_ACTION_START + action_offset)

                for shape in ("square", "circle"):
                    expected_gain = int(shape == expected_shape)
                    self.assertEqual(
                        getattr(owner, f"personal_supply_{shape}s"),
                        before[shape] + expected_gain,
                    )
                self.assertIsNone(game.pending_income_favour_owner)


if __name__ == "__main__":
    unittest.main()
