import contextlib
import io
import unittest

from game.game_actions import claim_post_action
from game.game_runner import create_headless_game
from game.invariants import validate_game
from map_data.map_attributes import BonusMarker, Map


POST_ACTION_COUNT = 121
BM_ACTION_START = 527
BM_CITY_ACTION_START = 583
INCOME_ACTION_START = 522


def first_post_action(game, route):
    index = 0
    for candidate in game.selected_map.routes:
        if candidate is route:
            return index
        index += len(candidate.posts)
    raise AssertionError("Route is not on the map")


def promo_supply():
    supply = [
        marker
        for marker, count in Map.STANDARD_BONUS_MARKER_SUPPLY.items()
        for _ in range(count)
    ]
    supply.remove("PlaceAdjacent")
    supply.remove("SwapOffice")
    supply.remove("3Actions")
    supply.extend(
        [
            "ExchangeBonusMarker",
            "Tribute4EstablishingTP",
            "BlockTradeRoute",
        ]
    )
    return supply


class PromoBonusMarkerTests(unittest.TestCase):
    def game(self):
        return create_headless_game(
            map_num=1,
            num_players=3,
            seed=124,
            bonus_marker_supply=promo_supply(),
        )

    def test_default_supply_never_contains_promos(self):
        game = create_headless_game(map_num=1, num_players=3, seed=124)
        self.assertEqual(len(game.selected_map.bonus_marker_pool), 12)
        self.assertTrue(
            set(game.selected_map.bonus_marker_pool).isdisjoint(
                Map.PROMO_BONUS_MARKERS
            )
        )

    def test_explicit_mix_preserves_fifteen_total_and_is_seeded(self):
        first = self.game()
        second = self.game()
        self.assertEqual(first.selected_map.bonus_marker_pool, second.selected_map.bonus_marker_pool)
        self.assertEqual(len(first.selected_map.bonus_marker_pool) + 3, 15)
        for marker_type in Map.PROMO_BONUS_MARKERS:
            self.assertIn(marker_type, first.selected_map.bonus_marker_pool)

    def test_invalid_explicit_mix_is_rejected(self):
        with self.assertRaises(ValueError):
            create_headless_game(1, 3, bonus_marker_supply=["Move3"] * 12)
        with self.assertRaises(ValueError):
            create_headless_game(1, 3, bonus_marker_supply=["unknown"] * 12)
        with self.assertRaises(ValueError):
            create_headless_game(1, 3, bonus_marker_supply=promo_supply()[:-1])

    def test_exchange_spends_marker_at_chosen_opponent_and_takes_used_marker(self):
        game = self.game()
        player, opponent = game.players[:2]
        exchange = BonusMarker("ExchangeBonusMarker", owner=player)
        desired = BonusMarker("Move3", owner=opponent)
        opponent.used_bonus_markers = [desired]
        player.bonus_markers = [exchange]

        game.apply_action(BM_ACTION_START + 5)
        self.assertTrue(game.waiting_for_bm_exchange_bm)
        game.apply_action(BM_CITY_ACTION_START + 1)
        game.apply_action(BM_ACTION_START + 1)

        self.assertEqual(player.bonus_markers, [desired])
        self.assertEqual(opponent.used_bonus_markers, [exchange])
        self.assertIs(exchange.owner, opponent)
        self.assertFalse(game.waiting_for_bm_exchange_bm)

    def test_tribute_costs_one_trader_and_self_trigger_allows_income_choice(self):
        game = self.game()
        player = game.current_player
        marker = BonusMarker("Tribute4EstablishingTP", owner=player)
        player.bonus_markers = [marker]
        route = game.selected_map.routes[0]
        supply_before = player.personal_supply_squares

        game.apply_action(BM_ACTION_START + 6)
        game.apply_action(first_post_action(game, route))
        self.assertEqual(player.personal_supply_squares, supply_before - 1)
        self.assertEqual(route.tribute_owners, [player])
        self.assertIn(marker, player.used_bonus_markers)
        validate_game(game)

        stock_before = player.general_stock_squares
        route.award_tributes(game)
        self.assertIs(game.pending_tribute_income_owners[0], player)
        with contextlib.redirect_stdout(io.StringIO()):
            game.apply_action(INCOME_ACTION_START)
        self.assertEqual(player.general_stock_squares, stock_before - 2)
        self.assertEqual(player.personal_supply_squares, supply_before + 1)
        self.assertEqual(route.tribute_owners, [player])
        validate_game(game)

    def test_tribute_and_block_markers_may_target_any_route(self):
        for marker_type, bm_offset in (
            ("Tribute4EstablishingTP", 6),
            ("BlockTradeRoute", 7),
        ):
            with self.subTest(marker=marker_type):
                game = self.game()
                game.current_player.bonus_markers = [
                    BonusMarker(marker_type, owner=game.current_player)
                ]
                game.apply_action(BM_ACTION_START + bm_offset)
                legal_posts = game.legal_action_mask()[:POST_ACTION_COUNT].nonzero().flatten()
                self.assertEqual(len(legal_posts), len(game.selected_map.routes))
                self.assertEqual(
                    legal_posts.tolist(),
                    [
                        first_post_action(game, route)
                        for route in game.selected_map.routes
                    ],
                )

    def test_tribute_triggers_only_for_the_completed_routes_neighboring_cities(self):
        game = self.game()
        first_owner, second_owner = game.players[:2]
        first_route, second_route = game.selected_map.routes[:2]
        first_route.tribute_owners.append(first_owner)
        second_route.tribute_owners.append(second_owner)

        first_route.award_tributes(game)

        self.assertEqual(game.pending_tribute_income_owners, [first_owner])

    def test_block_costs_one_trader_and_charges_every_ordinary_placement(self):
        game = self.game()
        owner, placing_player = game.players[:2]
        marker = BonusMarker("BlockTradeRoute", owner=owner)
        owner.bonus_markers = [marker]
        route = game.selected_map.routes[0]

        game.apply_action(BM_ACTION_START + 7)
        game.apply_action(first_post_action(game, route))
        self.assertEqual(route.block_marker_owners, [owner])
        self.assertTrue(all(post.blocked_bm for post in route.posts))
        validate_game(game)

        game.current_player_index = 1
        game.current_player = placing_player
        game.active_player = 1
        total_before = (
            placing_player.personal_supply_squares
            + placing_player.personal_supply_circles
        )
        stock_before = (
            placing_player.general_stock_squares
            + placing_player.general_stock_circles
        )
        with contextlib.redirect_stdout(io.StringIO()):
            claim_post_action(game, route, route.posts[0], "square")

        self.assertEqual(
            placing_player.personal_supply_squares
            + placing_player.personal_supply_circles,
            total_before - 2,
        )
        self.assertEqual(
            placing_player.general_stock_squares
            + placing_player.general_stock_circles,
            stock_before + 1,
        )
        self.assertIs(route.posts[0].owner, placing_player)
        validate_game(game)


if __name__ == "__main__":
    unittest.main()
