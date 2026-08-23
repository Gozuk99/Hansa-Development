import contextlib
import io
import unittest

from tests.action_helpers import legal_action_mask

from game.game_actions import handle_bonus_marker
from game.game_runner import create_headless_game
from game.invariants import validate_game
from map_data.map_attributes import BonusMarker
from map_data.constants import DARK_GREEN


MAX_POSTS = 121
ROUTE_ACTION_START = 256
ROUTE_OFFICE_OFFSET = 40
ROUTE_UPGRADE_OFFSET = 120
INCOME_ACTION_START = 576
BM_CITY_ACTION_START = 656
END_CONTEXT_ACTION = 736
PLACE_ADJACENT_ACTION = 600


def post_index(game, target, shape="square"):
    index = 0
    for route in game.selected_map.routes:
        for post in route.posts:
            if post is target:
                return index + (MAX_POSTS if shape == "circle" else 0)
            index += 1
    raise AssertionError("Post is not on the map")


def occupy_route(player, route, shapes=None):
    shapes = shapes or [
        "circle" if post.required_shape == "circle" else "square" for post in route.posts
    ]
    for post, shape in zip(route.posts, shapes):
        post.claim(player, shape)
        if shape == "circle":
            player.personal_supply_circles -= 1
        else:
            player.personal_supply_squares -= 1


class EasternHanseaticTests(unittest.TestCase):
    def game(self):
        return create_headless_game(2, 3, seed=124)

    def test_waren_offers_bank_actions_or_additional_office_but_not_normal_office(self):
        game = self.game()
        player = game.current_player
        route = next(
            route
            for route in game.selected_map.routes
            if any(city.name == "Waren" for city in route.cities)
        )
        waren_index = next(index for index, city in enumerate(route.cities) if city.name == "Waren")
        occupy_route(player, route)
        route_index = game.selected_map.routes.index(route)
        mask = legal_action_mask(game)
        office_action = ROUTE_ACTION_START + ROUTE_OFFICE_OFFSET + route_index * 2 + waren_index
        upgrade_base = ROUTE_ACTION_START + ROUTE_UPGRADE_OFFSET + route_index * 4 + waren_index * 2

        self.assertEqual(mask[office_action].item(), 0)
        self.assertEqual(mask[upgrade_base : upgrade_base + 2].tolist(), [1, 1])

        marker = next(
            route.bonus_marker
            for route in game.selected_map.routes
            if route.bonus_marker and route.bonus_marker.type == "PlaceAdjacent"
        )
        marker.owner = player
        player.bonus_markers.append(marker)
        self.assertEqual(legal_action_mask(game)[PLACE_ADJACENT_ACTION].item(), 1)

    def test_maritime_routes_expose_and_require_exact_merchant_posts(self):
        game = self.game()
        for route in game.selected_map.routes:
            if route.required_circles:
                required = [post.required_shape for post in route.posts]
                self.assertEqual(required.count("circle"), route.required_circles)
                self.assertTrue(all(shape in (None, "circle") for shape in required))

    def test_permanent_privilege_develops_and_releases_trader_without_collection(self):
        game = self.game()
        player = game.current_player
        route = next(
            route for route in game.selected_map.routes if route.has_permanent_bm_type == "+1Priv"
        )
        supply_before = player.personal_supply_squares
        privilege_before = player.privilege

        handle_bonus_marker(game, player, route, [])

        self.assertNotEqual(player.privilege, privilege_before)
        self.assertEqual(player.personal_supply_squares, supply_before + 1)
        self.assertNotIn(route.permanent_bonus_marker, player.bonus_markers)

    def test_green_city_marker_chooses_shape_and_places_to_right(self):
        game = self.game()
        player, opponent = game.players[:2]
        green_city = next(city for city in game.selected_map.cities if city.color == DARK_GREEN)
        green_city.offices[0].controller = opponent
        route = next(
            route
            for route in game.selected_map.routes
            if route.has_permanent_bm_type == "ClaimGreenCity"
        )
        game.waiting_for_bm_green_city = True
        player.personal_supply_circles = max(1, player.personal_supply_circles)

        choices = legal_action_mask(game)[BM_CITY_ACTION_START:720]
        self.assertGreaterEqual(choices.sum().item(), 2)
        # The first green city contributes square then circle.
        game.apply_action(BM_CITY_ACTION_START + 47)

        self.assertIs(green_city.offices[0].controller, opponent)
        self.assertIs(green_city.offices[1].controller, player)
        self.assertEqual(green_city.offices[1].shape, "circle")
        self.assertIsNotNone(route.permanent_bonus_marker)

    def test_additional_office_does_not_remove_occupied_green_city_office(self):
        game = self.game()
        player, opponent = game.players[:2]
        city = next(
            candidate for candidate in game.selected_map.cities if candidate.name == "Belgard"
        )
        occupied_office = city.offices[-1]
        occupied_office.controller = opponent
        occupied_office.owner_piece_shape = "square"
        occupied_office.color = opponent.color
        opponent.personal_supply_squares -= 1
        marker = BonusMarker("PlaceAdjacent", owner=player)
        player.bonus_markers.append(marker)
        player.personal_supply_squares -= 1

        city.claim_office_with_bonus_marker(player)

        self.assertEqual(len(city.offices), 6)
        self.assertIn(occupied_office, city.offices)
        self.assertIs(occupied_office.controller, opponent)
        self.assertIs(city.offices[0].controller, player)
        validate_game(game)

    def test_move_any_two_moves_own_and_opponent_pieces_and_may_finish_early(self):
        game = self.game()
        player, opponent = game.players[:2]
        route = game.selected_map.routes[0]
        own_post, opponent_post, destination = route.posts[:3]
        own_post.claim(player, "square")
        opponent_post.claim(opponent, "square")
        player.personal_supply_squares -= 1
        opponent.personal_supply_squares -= 1
        game.waiting_for_bm_move_any_2 = True
        player.pieces_to_pickup = 2

        game.apply_action(post_index(game, own_post))
        game.apply_action(END_CONTEXT_ACTION)
        game.apply_action(post_index(game, destination))

        self.assertIs(destination.owner, player)
        self.assertIs(opponent_post.owner, opponent)
        self.assertFalse(game.waiting_for_bm_move_any_2)
        validate_game(game)

    def test_place_two_from_route_selects_composition_and_places_exactly_two(self):
        game = self.game()
        player = game.current_player
        route = next(
            route
            for route in game.selected_map.routes
            if route.has_permanent_bm_type == "Place2TradesmenFromRoute"
        )
        occupy_route(
            player,
            route,
            ["circle"] + ["square"] * (len(route.posts) - 1),
        )
        route_index = game.selected_map.routes.index(route)
        with contextlib.redirect_stdout(io.StringIO()):
            game.apply_action(ROUTE_ACTION_START + route_index)

        game.apply_action(INCOME_ACTION_START + 1)
        self.assertEqual(
            [shape for shape, _owner, _region in player.holding_pieces],
            ["circle", "square"],
        )
        empty_posts = [
            post
            for candidate in game.selected_map.routes
            for post in candidate.posts
            if post.required_shape is None
        ][:2]
        with contextlib.redirect_stdout(io.StringIO()):
            game.apply_action(post_index(game, empty_posts[0], "circle"))
            game.apply_action(post_index(game, empty_posts[1], "square"))

        self.assertFalse(game.waiting_for_place2_from_route)
        self.assertEqual(player.holding_pieces, [])
        validate_game(game)


if __name__ == "__main__":
    unittest.main()
