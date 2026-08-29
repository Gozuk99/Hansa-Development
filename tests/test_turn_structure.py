import unittest

from tests.action_helpers import legal_action_mask

from game.game_runner import create_headless_game
from game.turn_state import TurnPhase, TurnStateError
from game.action_resolvers import resolve_control_interaction
from game.game_actions import (
    InvalidActionError,
    claim_route_for_points,
    claim_route_for_upgrade,
    move_action,
)
from map_data.map_attributes import BonusMarker


class TurnStructureTests(unittest.TestCase):
    def test_fresh_game_starts_in_action_phase(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        self.assertEqual(game.turn_phase, TurnPhase.ACTIONS)
        self.assertEqual(game.current_player.actions_remaining, 2)

    def test_spending_last_action_completes_turn(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.current_player.spend_action()
        self.assertEqual(game.turn_phase, TurnPhase.ACTIONS)
        game.current_player.spend_action()
        self.assertEqual(game.turn_phase, TurnPhase.TURN_COMPLETE)

    def test_player_cannot_spend_action_below_zero(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.current_player.spend_action()
        game.current_player.spend_action()
        with self.assertRaises(RuntimeError):
            game.current_player.spend_action()

    def test_paid_action_history_counts_actions_and_resets_move_streak(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        player = game.current_player

        player.spend_action(is_move=True)
        self.assertEqual(player.consecutive_paid_move_actions, 1)
        self.assertEqual(player.paid_actions_spent_this_turn, 1)
        self.assertEqual(player.paid_move_actions_spent_this_turn, 1)

        player.grant_actions(2)
        player.spend_action(is_move=True)
        self.assertEqual(player.consecutive_paid_move_actions, 2)
        self.assertEqual(player.paid_actions_spent_this_turn, 2)
        self.assertEqual(player.paid_move_actions_spent_this_turn, 2)

        player.spend_action()
        self.assertEqual(player.consecutive_paid_move_actions, 0)
        self.assertEqual(player.paid_actions_spent_this_turn, 3)
        self.assertEqual(player.paid_move_actions_spent_this_turn, 2)

    def test_action_grants_and_forfeit_do_not_count_as_paid_actions(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        player = game.current_player
        player.grant_actions(4)
        player.forfeit_remaining_actions()
        self.assertEqual(player.paid_actions_spent_this_turn, 0)
        self.assertEqual(player.paid_move_actions_spent_this_turn, 0)
        self.assertEqual(player.consecutive_paid_move_actions, 0)

    def test_turn_transition_resets_paid_action_history(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.players[1].paid_actions_spent_this_turn = 4
        game.players[1].paid_move_actions_spent_this_turn = 3
        game.players[1].consecutive_paid_move_actions = 2
        game.current_player.forfeit_remaining_actions()

        game.advance_turn()

        self.assertEqual(game.current_player.paid_actions_spent_this_turn, 0)
        self.assertEqual(game.current_player.paid_move_actions_spent_this_turn, 0)
        self.assertEqual(game.current_player.consecutive_paid_move_actions, 0)

    def test_income_claim_and_upgrade_count_as_non_move_paid_actions(self):
        income_game = create_headless_game(map_num=2, num_players=3, seed=124)
        income_player = income_game.current_player
        income_player.income_action(1, 0)
        self.assertEqual(income_player.paid_actions_spent_this_turn, 1)
        self.assertEqual(income_player.paid_move_actions_spent_this_turn, 0)

        claim_game = create_headless_game(map_num=2, num_players=3, seed=124)
        claim_player = claim_game.current_player
        claim_route = next(
            route for route in claim_game.selected_map.routes if not route.required_circles
        )
        for post in claim_route.posts:
            post.claim(claim_player, "square")
        claim_route_for_points(claim_game, claim_route)
        self.assertEqual(claim_player.paid_actions_spent_this_turn, 1)
        self.assertEqual(claim_player.paid_move_actions_spent_this_turn, 0)

        upgrade_game = create_headless_game(map_num=2, num_players=3, seed=124)
        upgrade_player = upgrade_game.current_player
        upgrade_route = next(
            route
            for route in upgrade_game.selected_map.routes
            if any(
                upgrade in ("Keys", "Privilege", "Book", "Actions", "Bank")
                for city in route.cities
                for upgrade in city.upgrade_city_type
            )
        )
        upgrade_city = next(
            city
            for city in upgrade_route.cities
            if any(
                upgrade in ("Keys", "Privilege", "Book", "Actions", "Bank")
                for upgrade in city.upgrade_city_type
            )
        )
        upgrade_choice = next(
            upgrade
            for upgrade in upgrade_city.upgrade_city_type
            if upgrade in ("Keys", "Privilege", "Book", "Actions", "Bank")
        )
        for post in upgrade_route.posts:
            post.claim(upgrade_player, post.required_shape or "square")
        claim_route_for_upgrade(upgrade_game, upgrade_city, upgrade_route, upgrade_choice)
        self.assertEqual(upgrade_player.paid_actions_spent_this_turn, 1)
        self.assertEqual(upgrade_player.paid_move_actions_spent_this_turn, 0)

    def test_multiclick_move_counts_once_when_final_piece_is_placed(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        player = game.current_player
        route = next(
            route
            for route in game.selected_map.routes
            if len([post for post in route.posts if post.required_shape is None]) >= 3
        )
        origin, second, destination = [post for post in route.posts if post.required_shape is None][
            :3
        ]
        origin.claim(player, "square")
        second.claim(player, "square")

        move_action(game, origin)
        move_action(game, second)
        self.assertEqual(player.paid_actions_spent_this_turn, 0)
        self.assertEqual(player.paid_move_actions_spent_this_turn, 0)

        move_action(game, destination)
        self.assertEqual(player.paid_actions_spent_this_turn, 0)
        move_action(game, origin)
        self.assertEqual(player.paid_actions_spent_this_turn, 1)
        self.assertEqual(player.paid_move_actions_spent_this_turn, 1)
        self.assertEqual(player.consecutive_paid_move_actions, 1)

    def test_advance_turn_requires_completed_turn(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        with self.assertRaises(TurnStateError):
            game.advance_turn()

    def test_advance_turn_resets_next_players_actions(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.current_player.forfeit_remaining_actions()
        game.advance_turn()
        self.assertEqual(game.current_player_index, 1)
        self.assertIs(game.current_player, game.players[1])
        self.assertEqual(game.current_player.actions_remaining, game.current_player.actions)
        self.assertEqual(game.turn_phase, TurnPhase.ACTIONS)

    def test_pending_displacement_blocks_turn_advance(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.current_player.forfeit_remaining_actions()
        game.waiting_for_displaced_player = True
        game.displaced_player.player = game.players[1]
        game.displaced_player.displaced_shape = "square"
        self.assertEqual(game.turn_phase, TurnPhase.DISPLACEMENT)
        with self.assertRaises(TurnStateError):
            game.advance_turn()

    def test_pending_bonus_replacement_blocks_turn_advance(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.current_player.forfeit_remaining_actions()
        game.replace_bonus_marker = 1
        self.assertEqual(game.turn_phase, TurnPhase.REPLACE_BONUS_MARKERS)
        with self.assertRaises(TurnStateError):
            game.advance_turn()

    def test_one_action_tile_applies_at_start_of_owners_turn(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.OneActionOwner = game.players[1]
        game.current_player.forfeit_remaining_actions()
        game.advance_turn()
        self.assertEqual(
            game.current_player.actions_remaining,
            game.current_player.actions + 1,
        )

    def test_turn_and_round_numbers_advance(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        for expected_turn in (2, 3, 4):
            game.current_player.forfeit_remaining_actions()
            game.advance_turn()
            self.assertEqual(game.turn_number, expected_turn)
        self.assertEqual(game.current_player_index, 0)
        self.assertEqual(game.round_number, 2)

    def test_conflicting_pending_workflows_are_rejected(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.waiting_for_displaced_player = True
        game.waiting_for_bm_move3 = True
        with self.assertRaises(TurnStateError):
            _ = game.turn_phase

    def test_tribute_income_precedes_permanent_bonus_marker_follow_up(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        claimant = game.current_player
        tribute_owner = game.players[1]
        game.waiting_for_bm_move_any_2 = True
        game.begin_tribute_income_responses([tribute_owner])

        self.assertEqual(game.turn_phase, TurnPhase.TRIBUTE_INCOME_RESPONSE)
        self.assertIs(game.players[game.active_player], tribute_owner)

        game.resolve_tribute_income(0)

        self.assertEqual(game.turn_phase, TurnPhase.BONUS_MARKER_CHOICE)
        self.assertIs(game.current_player, claimant)
        self.assertEqual(game.active_player, game.current_player_index)

    def test_displacement_finishes_before_end_of_turn_marker_replacement(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.current_player.forfeit_remaining_actions()
        game.waiting_for_displaced_player = True
        game.replace_bonus_marker = 1

        self.assertEqual(game.turn_phase, TurnPhase.DISPLACEMENT)

        game.waiting_for_displaced_player = False
        self.assertEqual(game.turn_phase, TurnPhase.REPLACE_BONUS_MARKERS)

    def test_end_turn_is_illegal_while_actions_remain(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        mask = legal_action_mask(game)
        self.assertEqual(mask[737].item(), 0)
        with self.assertRaises(InvalidActionError):
            resolve_control_interaction(game)

    def test_player_may_forgo_usable_bonus_marker_and_end_turn(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.current_player.forfeit_remaining_actions()
        game.current_player.bonus_markers.append(BonusMarker("3Actions"))
        mask = legal_action_mask(game)
        self.assertEqual(mask[737].item(), 1)

        game.apply_action(737)

        self.assertEqual(game.current_player_index, 1)
        self.assertEqual(game.turn_number, 2)

    def test_end_turn_enters_replacement_phase_before_switching(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.current_player.forfeit_remaining_actions()
        game.replace_bonus_marker = 1
        mask = legal_action_mask(game)
        self.assertEqual(mask[737].item(), 1)

        game.apply_action(737)

        self.assertEqual(game.current_player_index, 0)
        self.assertTrue(game.current_player.ending_turn)
        self.assertEqual(game.turn_phase, TurnPhase.REPLACE_BONUS_MARKERS)

    def test_displacement_phase_masks_every_non_post_action(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.waiting_for_displaced_player = True
        game.displaced_player.player = game.players[1]
        game.displaced_player.displaced_shape = "square"
        game.displaced_player.total_pieces_to_place = 1
        mask = legal_action_mask(game)
        enabled = mask.nonzero(as_tuple=True)[0].tolist()
        self.assertTrue(all(index < 242 or index in (728, 736) for index in enabled))

    def test_tile_payment_phase_masks_every_other_action_family(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.waiting_for_buy_tile_with_bm = True
        game.tile_to_buy = "DisplaceAnywhere"
        game.current_player.bonus_markers.append(BonusMarker("3Actions"))
        mask = legal_action_mask(game)
        self.assertGreater(mask[592:601].count_nonzero().item(), 0)
        self.assertEqual(mask[:592].count_nonzero().item(), 0)
        self.assertEqual(mask[601:].count_nonzero().item(), 0)

    def test_bonus_marker_follow_up_does_not_offer_tile_purchase(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.use_emperors_favour = True
        game.tile_pool = ["DisplaceAnywhere"]
        game.current_player.bonus_markers = [
            BonusMarker("SwapOffice"),
            BonusMarker("Move3"),
        ]
        game.waiting_for_bm_upgrade_ability = True

        mask = legal_action_mask(game)

        self.assertEqual(mask[601:609].count_nonzero().item(), 0)
        self.assertEqual(game.turn_phase, TurnPhase.BONUS_MARKER_CHOICE)

    def test_marker_replacement_phase_masks_every_other_action_family(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        game.current_player.forfeit_remaining_actions()
        game.current_player.ending_turn = True
        game.replace_bonus_marker = 1
        mask = legal_action_mask(game)
        self.assertGreater(mask[256:296].count_nonzero().item(), 0)
        self.assertEqual(mask[:256].count_nonzero().item(), 0)
        self.assertEqual(mask[296:].count_nonzero().item(), 0)

    def test_apply_action_rejects_out_of_range_and_masked_actions(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        with self.assertRaises(InvalidActionError):
            game.apply_action(-1)
        with self.assertRaises(InvalidActionError):
            game.apply_action(768)
        with self.assertRaises(InvalidActionError):
            game.apply_action(737)

    def test_apply_action_is_authoritative_supported_boundary(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        legal_indices = legal_action_mask(game).nonzero(as_tuple=True)[0].tolist()
        action_index = legal_indices[0]
        before_actions = game.current_player.actions_remaining

        game.apply_action(action_index)

        self.assertEqual(game.players[0].actions_remaining, before_actions - 1)

    def test_legality_checks_do_not_consume_britannia_permission(self):
        game = create_headless_game(map_num=3, num_players=3, seed=124)
        game.cardiff_priv = game.current_player
        game.current_player.refresh_map3_priv_actions(game)
        before = game.current_player.brown_priv_count

        legal_action_mask(game)
        legal_action_mask(game)

        self.assertEqual(game.current_player.brown_priv_count, before)


if __name__ == "__main__":
    unittest.main()
