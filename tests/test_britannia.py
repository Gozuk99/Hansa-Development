import contextlib
import io
import unittest

from ai.action_options import (
    map_income_action,
    mask_income_actions,
    mask_post_action,
    mask_replace_bm,
)
from game.game_actions import assign_new_bonus_marker_on_route
from game.game_runner import create_headless_game


def city(game, name):
    return next(city for city in game.selected_map.cities if city.name == name)


class BritanniaTests(unittest.TestCase):
    def game(self, players=4):
        return create_headless_game(3, players, seed=124)

    def test_regional_permissions_are_recomputed_from_current_control(self):
        game = self.game()
        first, second = game.players[:2]
        cardiff = city(game, "Cardiff")
        cardiff.offices[0].controller = first

        first.refresh_map3_priv_actions(game)
        self.assertEqual(first.brown_priv_count, 1)

        cardiff.offices[1].controller = second
        cardiff.offices[2].controller = second
        first.refresh_map3_priv_actions(game)
        second.refresh_map3_priv_actions(game)
        self.assertEqual(first.brown_priv_count, 0)
        self.assertEqual(second.brown_priv_count, 1)
        self.assertIs(game.cardiff_priv, second)

    def test_london_grants_one_shared_permission(self):
        game = self.game()
        player = game.current_player
        city(game, "London").offices[0].controller = player
        player.refresh_map3_priv_actions(game)
        self.assertEqual(player.london_priv_count, 1)

        wales = next(route for route in game.selected_map.routes if route.region == "Wales")
        scotland = next(route for route in game.selected_map.routes if route.region == "Scotland")
        game.consume_region_privilege(wales)
        self.assertFalse(game.check_brown_blue_priv(scotland))

    def test_replacement_bonus_markers_are_limited_to_england(self):
        game = self.game()
        game.current_player.actions_remaining = 0
        game.current_player.ending_turn = True
        game.replace_bonus_marker = 1
        mask = mask_replace_bm(game)
        for index, route in enumerate(game.selected_map.routes):
            if route.region in ("Wales", "Scotland"):
                self.assertEqual(mask[index].item(), 0)

        wales = next(
            route for route in game.selected_map.routes
            if route.region == "Wales" and not route.bonus_marker
        )
        game.pending_bonus_markers = ["Move3"]
        assign_new_bonus_marker_on_route(game, wales)
        self.assertIsNone(wales.bonus_marker)

    def test_britannia_permanent_marker_layouts(self):
        expected_common = {
            ("Calais", "Canterbury"): "MoveAny2",
            ("Calais", "Southhampton"): "Place2ScotlandOrWales",
        }
        for players in (3, 4, 5):
            game = self.game(players)
            actual = {
                tuple(sorted(city.name for city in route.cities)):
                    route.permanent_bonus_marker.type
                for route in game.selected_map.routes
                if route.permanent_bonus_marker
            }
            self.assertTrue(expected_common.items() <= actual.items())
            if players == 3:
                self.assertEqual(actual[("Carlisle", "IsleOfMan")], "MoveAny2")
            else:
                self.assertNotIn(("Carlisle", "IsleOfMan"), actual)

    def test_place_two_uses_general_stock_and_allows_both_regions(self):
        game = self.game()
        player = game.current_player
        player.general_stock_squares = 2
        player.general_stock_circles = 0
        game.pending_britannia_place2 = True

        self.assertEqual(mask_income_actions(game)[0].item(), 1)
        map_income_action(game, 0)
        self.assertEqual(player.general_stock_squares, 0)
        self.assertEqual(len(player.holding_pieces), 2)
        self.assertTrue(game.waiting_for_place2_in_scotland_or_wales)

    def test_place_two_cannot_skip_a_higher_priority_source_for_shape(self):
        game = self.game()
        player = game.current_player
        player.general_stock_squares = 2
        player.general_stock_circles = 0
        player.personal_supply_circles = 2
        game.pending_britannia_place2 = True

        mask = mask_income_actions(game)
        self.assertEqual(mask[0].item(), 1)
        self.assertEqual(mask[1].item(), 0)
        self.assertEqual(mask[2].item(), 0)

    def test_normal_move_and_move_markers_use_their_distinct_country_rules(self):
        game = self.game()
        player = game.current_player
        self.assertTrue(player.is_valid_region_transition("Wales", None))
        self.assertFalse(player.is_valid_region_transition(None, "Wales"))
        self.assertFalse(player.is_valid_region_transition("Wales", "Scotland"))

        wales_post = next(
            post
            for route in game.selected_map.routes if route.region == "Wales"
            for post in route.posts
            if post.required_shape in (None, "square")
        )
        england_post = next(
            post
            for route in game.selected_map.routes if route.region is None
            for post in route.posts
            if post.required_shape in (None, "square")
        )
        all_posts = [
            post for route in game.selected_map.routes for post in route.posts
        ]
        player.holding_pieces = [("square", game.players[1], "Wales")]
        player.pieces_to_pickup = 0

        game.waiting_for_bm_move3 = True
        move3_mask = mask_post_action(game)
        self.assertEqual(move3_mask[all_posts.index(wales_post)].item(), 1)
        self.assertEqual(move3_mask[all_posts.index(england_post)].item(), 0)

        game.waiting_for_bm_move3 = False
        game.waiting_for_bm_move_any_2 = True
        move2_mask = mask_post_action(game)
        self.assertEqual(move2_mask[all_posts.index(wales_post)].item(), 1)
        self.assertEqual(move2_mask[all_posts.index(england_post)].item(), 0)

    def test_regional_scoring_includes_isle_of_man_in_both_regions(self):
        game = self.game()
        first, second, third, _ = game.players
        city(game, "Cardiff").offices[0].controller = first
        city(game, "Carlisle").offices[0].controller = second
        isle = city(game, "IsleOfMan")
        isle.offices[0].controller = third

        points = game.calculate_britannia_region_points()
        self.assertEqual(points[first], 5)
        self.assertEqual(points[second], 5)
        self.assertEqual(points[third], 10)

    def test_regional_scoring_splits_tied_places_rounding_down(self):
        game = self.game(3)
        first, second, third = game.players
        cardiff = city(game, "Cardiff")
        cardiff.offices[0].controller = first
        cardiff.offices[1].controller = second
        city(game, "Pembroke").offices[0].controller = third

        points = game.calculate_britannia_region_points()
        self.assertEqual(sorted(points.values(), reverse=True), [5, 5, 2])

    def test_final_score_breakdown_contains_britannia_awards(self):
        game = self.game(3)
        player = game.current_player
        city(game, "Cardiff").offices[0].controller = player
        with contextlib.redirect_stdout(io.StringIO()):
            game.finalize_end_of_game_points()
        self.assertEqual(
            player.final_score_breakdown["Britannia Region Points"], 7
        )


if __name__ == "__main__":
    unittest.main()
