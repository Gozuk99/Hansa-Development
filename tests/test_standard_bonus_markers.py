import contextlib
import io
import unittest

import torch

from tests.action_helpers import legal_action_mask

from ai.observation_encoder import ObservationEncoder
from game.game_runner import create_headless_game
from game.invariants import validate_game
from map_data.map_attributes import BonusMarker


MAX_POSTS = 121
BM_ACTION_START = 592
BM_CITY_ACTION_START = 656
BM_UPGRADE_ACTION_START = 720
END_CONTEXT_ACTION = 736
PLACE_ADJACENT_ACTION = 600
ADDITIONAL_CHOICE_START = 376


def post_index(game, target, shape="square"):
    index = 0
    for route in game.selected_map.routes:
        for post in route.posts:
            if post is target:
                return index + (MAX_POSTS if shape == "circle" else 0)
            index += 1
    raise AssertionError("Post is not on the map")


def occupy_route(player, route, shapes=None):
    shapes = shapes or ["square"] * len(route.posts)
    for post, shape in zip(route.posts, shapes):
        post.claim(player, shape)
        if shape == "square":
            player.personal_supply_squares -= 1
        else:
            player.personal_supply_circles -= 1


class StandardBonusMarkerTests(unittest.TestCase):
    def game(self):
        return create_headless_game(1, 3, seed=124)

    def test_exchange_lists_each_adjacent_pair_and_ignores_privilege_and_control(self):
        game = self.game()
        player, opponent = game.players[:2]
        city = next(city for city in game.selected_map.cities if len(city.offices) >= 3)
        city.offices[0].controller = player
        city.offices[1].controller = opponent
        city.offices[2].controller = player
        marker = BonusMarker("SwapOffice", owner=player)
        player.bonus_markers = [marker]
        score_before = player.score

        game.apply_action(BM_ACTION_START)
        pair_mask = legal_action_mask(game)[BM_CITY_ACTION_START:720]
        self.assertGreaterEqual(pair_mask.sum().item(), 2)
        pair_slot = pair_mask.nonzero(as_tuple=True)[0][0].item()
        game.apply_action(BM_CITY_ACTION_START + pair_slot)

        self.assertIs(city.offices[0].controller, opponent)
        self.assertIs(city.offices[1].controller, player)
        self.assertEqual(player.score, score_before)
        self.assertIn(marker, player.used_bonus_markers)

    def test_exchange_never_includes_additional_trading_posts(self):
        game = self.game()
        player, opponent = game.players[:2]
        city = game.selected_map.cities[0]
        city.offices[0].controller = opponent
        extra = city.create_new_office(player.color)
        extra.controller = player
        extra.place_adjacent_office = True

        self.assertEqual(city.eligible_swap_pairs(player), [])

    def test_exchange_includes_all_offices_in_map_two_green_cities(self):
        game = create_headless_game(2, 3, seed=124)
        player, opponent = game.players[:2]
        city = next(city for city in game.selected_map.cities if city.name == "Belgard")
        extra = city.create_new_office(player.color)
        extra.controller = player
        extra.place_adjacent_office = True
        city.offices[1].controller = opponent

        self.assertIn((0, 1), city.eligible_swap_pairs(player, game))

    def test_develop_ability_releases_piece_and_masks_fully_developed_tracks(self):
        game = self.game()
        player = game.current_player
        marker = BonusMarker("UpgradeAbility", owner=player)
        player.bonus_markers = [marker]
        supply_before = player.personal_supply_squares

        game.apply_action(BM_ACTION_START + 2)
        mask = legal_action_mask(game)[BM_UPGRADE_ACTION_START:728]
        self.assertEqual(mask.sum().item(), 5)
        game.apply_action(BM_UPGRADE_ACTION_START)

        self.assertEqual(player.personal_supply_squares, supply_before + 1)
        self.assertIn(marker, player.used_bonus_markers)
        self.assertFalse(game.waiting_for_bm_upgrade_ability)

    def test_three_and_four_action_markers_add_actions_without_spending_one(self):
        for marker_type, offset, amount in (("3Actions", 3, 3), ("4Actions", 4, 4)):
            with self.subTest(marker=marker_type):
                game = self.game()
                player = game.current_player
                marker = BonusMarker(marker_type, owner=player)
                player.bonus_markers = [marker]
                before = player.actions_remaining

                game.apply_action(BM_ACTION_START + offset)

                self.assertEqual(player.actions_remaining, before + amount)
                self.assertIn(marker, player.used_bonus_markers)

    def test_additional_trading_post_chooses_city_and_route_piece_shape(self):
        game = self.game()
        player, opponent = game.players[:2]
        route = next(route for route in game.selected_map.routes if len(route.posts) <= 3)
        city = route.cities[0]
        city.offices[0].controller = opponent
        if city.offices[0].shape == "square":
            opponent.personal_supply_squares -= 1
        else:
            opponent.personal_supply_circles -= 1
        marker = BonusMarker("PlaceAdjacent", owner=player)
        player.bonus_markers = [marker]
        player.personal_supply_circles = max(player.personal_supply_circles, 1)
        shapes = ["circle"] + ["square"] * (len(route.posts) - 1)
        occupy_route(player, route, shapes)
        route_index = game.selected_map.routes.index(route)
        encoder = ObservationEncoder()
        encoder.get_game_state(game)
        revision_before = game._observation_structure_revision

        game.apply_action(PLACE_ADJACENT_ACTION)
        circle_choice = ADDITIONAL_CHOICE_START + route_index * 4 + 1
        self.assertEqual(legal_action_mask(game)[circle_choice].item(), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            game.apply_action(circle_choice)

        self.assertTrue(city.offices[0].place_adjacent_office)
        self.assertEqual(city.offices[0].shape, "circle")
        self.assertIs(city.offices[0].controller, player)
        self.assertIn(marker, player.used_bonus_markers)
        self.assertTrue(all(post.owner is None for post in route.posts))
        self.assertEqual(game._observation_structure_revision, revision_before + 1)
        self.assertTrue(
            torch.equal(
                encoder.get_game_state(game),
                ObservationEncoder().get_game_state(game),
            )
        )
        validate_game(game)

    def test_move_three_can_pick_multiple_opponents_swap_and_finish_early(self):
        game = self.game()
        player, first_owner, second_owner = game.players
        route = game.selected_map.routes[0]
        first, second = route.posts[:2]
        first.claim(first_owner, "square")
        second.claim(second_owner, "circle")
        first_owner.personal_supply_squares -= 1
        second_owner.personal_supply_circles -= 1
        marker = BonusMarker("Move3", owner=player)
        player.bonus_markers = [marker]

        game.apply_action(BM_ACTION_START + 1)
        game.apply_action(post_index(game, first))
        game.apply_action(post_index(game, second))
        game.apply_action(END_CONTEXT_ACTION)
        game.apply_action(post_index(game, second, "square"))
        game.apply_action(post_index(game, first, "circle"))

        self.assertIs(first.owner, second_owner)
        self.assertEqual(first.owner_piece_shape, "circle")
        self.assertIs(second.owner, first_owner)
        self.assertEqual(second.owner_piece_shape, "square")
        self.assertFalse(game.waiting_for_bm_move3)
        self.assertIn(marker, player.used_bonus_markers)
        validate_game(game)


if __name__ == "__main__":
    unittest.main()
