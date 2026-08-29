import hashlib
import json
import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest import mock

import torch

from ai.observation_encoder import ObservationEncoder
from game.game_actions import move_action, refresh_displacement_targets
from game.game_runner import create_headless_game
from game.structured_actions import RouteInteraction, SupplyInteraction, TileInteraction
from game.turn_state import TurnPhase
from map_data.constants import (
    BLACKISH_BROWN,
    DARK_BLUE,
    DARK_GREEN,
    DARK_RED,
    GREY,
    WHITE,
)
from map_data.map_attributes import BonusMarker
from training.self_play import (
    TrainingDecision,
    apply_all_move_turn_target,
    clear_move_route_focus_after_claim,
    consecutive_move_penalty,
    move_route_focus_reward,
    pointless_movement_penalty,
    update_move_claim_combo,
)


class ObservationEncoderTests(unittest.TestCase):
    def game(self, map_num=1, players=3, **options):
        return create_headless_game(map_num, players, seed=124, **options)

    def observation(self, game):
        return ObservationEncoder().build(game)

    @staticmethod
    def _open_post(route, *, excluded=()):
        return next(
            post
            for post in route.posts
            if post not in excluded and post.required_shape is None and not post.is_owned()
        )

    def _active_move_state(self):
        game = self.game()
        player = game.current_player
        route = next(
            route
            for route in game.selected_map.routes
            if sum(post.required_shape is None for post in route.posts) >= 2
        )
        origin = self._open_post(route)
        origin.claim(player, "square")
        move_action(game, origin)
        return game

    @staticmethod
    def _decision(observation, workflow_id):
        action_index = observation.legal_action_mask.nonzero(as_tuple=False).flatten()[0].item()
        return TrainingDecision(
            observation=observation.features,
            legal_action_mask=observation.legal_action_mask,
            action_index=action_index,
            acting_player_index=observation.observer_index,
            player_reward_deltas=(0.0, 0.0, 0.0),
            immediate_reward=0.0,
            policy_tier=1,
            epsilon=0.0,
            top_k=2,
            used_epsilon=False,
            model_rank=1,
            legal_action_count=int(observation.legal_action_mask.sum()),
            movement_workflow_id=workflow_id,
        )

    def test_paid_action_history_breaks_consecutive_move_alias(self):
        encoder = ObservationEncoder()
        base = self._active_move_state()
        histories = [deepcopy(base) for _ in range(3)]
        for preceding_moves, game in enumerate(histories):
            player = game.current_player
            player.consecutive_paid_move_actions = preceding_moves
            player.paid_actions_spent_this_turn = preceding_moves
            player.paid_move_actions_spent_this_turn = preceding_moves

        observations = [encoder.build(game) for game in histories]
        for observation in observations[1:]:
            self.assertTrue(
                torch.equal(observations[0].features[:4641], observation.features[:4641])
            )
            self.assertTrue(
                torch.equal(
                    observations[0].legal_action_mask,
                    observation.legal_action_mask,
                )
            )
        self.assertEqual(
            [observation.features[4641:4644].tolist() for observation in observations],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        )
        self.assertEqual(
            [consecutive_move_penalty(4, preceding + 1) for preceding in range(3)],
            [0.0, -200.0, -1500.0],
        )

    def test_paid_action_history_breaks_all_move_turn_alias(self):
        encoder = ObservationEncoder()
        all_move_game = self._active_move_state()
        mixed_game = deepcopy(all_move_game)
        all_move_game.current_player.consecutive_paid_move_actions = 2
        all_move_game.current_player.paid_actions_spent_this_turn = 2
        all_move_game.current_player.paid_move_actions_spent_this_turn = 2
        mixed_game.current_player.consecutive_paid_move_actions = 0
        mixed_game.current_player.paid_actions_spent_this_turn = 2
        mixed_game.current_player.paid_move_actions_spent_this_turn = 1

        all_move_observation = encoder.build(all_move_game)
        mixed_observation = encoder.build(mixed_game)
        self.assertTrue(
            torch.equal(
                all_move_observation.features[:4641],
                mixed_observation.features[:4641],
            )
        )
        self.assertTrue(
            torch.equal(
                all_move_observation.legal_action_mask,
                mixed_observation.legal_action_mask,
            )
        )
        self.assertEqual(all_move_observation.features[4641:4644].tolist(), [2, 2, 2])
        self.assertEqual(mixed_observation.features[4641:4644].tolist(), [0, 2, 1])

        first = self._decision(all_move_observation, 1)
        second = replace(first, movement_workflow_id=2)
        all_move_decisions = [first, second]
        mixed_decisions = [first, replace(second, movement_workflow_id=None)]
        self.assertTrue(apply_all_move_turn_target(all_move_decisions, (1, 2), 2))
        self.assertFalse(apply_all_move_turn_target(mixed_decisions, (1,), 2))
        self.assertEqual(
            [decision.local_training_target for decision in all_move_decisions],
            [-500.0, -500.0],
        )
        self.assertEqual(
            [decision.local_training_target for decision in mixed_decisions],
            [None, None],
        )

    def test_normal_move_snapshot_breaks_the_proven_origin_alias(self):
        encoder = ObservationEncoder()
        exact = self.game()
        cross_route = self.game()
        player_exact = exact.current_player
        player_cross = cross_route.current_player
        exact_route = exact.selected_map.routes[0]
        cross_origin_route = exact.selected_map.routes[1]
        exact_destination = self._open_post(exact_route)
        cross_origin = self._open_post(cross_origin_route)
        exact.selected_map.routes[0].posts[exact_route.posts.index(exact_destination)].claim(
            player_exact, "square"
        )
        cross_route.selected_map.routes[1].posts[
            cross_origin_route.posts.index(cross_origin)
        ].claim(player_cross, "square")

        exact_origin = exact_destination
        cross_origin = cross_route.selected_map.routes[1].posts[
            cross_origin_route.posts.index(cross_origin)
        ]
        cross_destination = cross_route.selected_map.routes[0].posts[
            exact_route.posts.index(exact_destination)
        ]
        move_action(exact, exact_origin)
        move_action(cross_route, cross_origin)

        exact_observation = encoder.build(exact)
        cross_observation = encoder.build(cross_route)
        differing = (
            (exact_observation.features != cross_observation.features).nonzero().flatten().tolist()
        )
        snapshot_start = encoder.MOVE_SNAPSHOT_START

        self.assertEqual(
            differing,
            [snapshot_start, snapshot_start + 1, snapshot_start + 10, snapshot_start + 11],
        )
        self.assertTrue(
            torch.equal(
                exact_observation.legal_action_mask,
                cross_observation.legal_action_mask,
            )
        )

        exact_origins = [(exact_origin, player_exact, "square")]
        cross_origins = [(cross_origin, player_cross, "square")]
        move_action(exact, exact_origin)
        move_action(cross_route, cross_destination)
        exact_post_routes = {
            post: route for route in exact.selected_map.routes for post in route.posts
        }
        cross_post_routes = {
            post: route for route in cross_route.selected_map.routes for post in route.posts
        }
        self.assertEqual(
            pointless_movement_penalty(exact_origins, [exact_origin], exact_post_routes),
            -1000,
        )
        self.assertEqual(
            pointless_movement_penalty(cross_origins, [cross_destination], cross_post_routes),
            0,
        )

    def test_normal_move_snapshot_is_immutable_and_clears_on_completion(self):
        game = self.game()
        player = game.current_player
        route = next(
            route
            for route in game.selected_map.routes
            if sum(post.required_shape is None for post in route.posts) >= 3
        )
        first = self._open_post(route)
        second = self._open_post(route, excluded=(first,))
        destination = self._open_post(route, excluded=(first, second))
        first.claim(player, "square")
        second.claim(player, "circle")

        neutral = ObservationEncoder().get_game_state(game)
        snapshot_start = ObservationEncoder.MOVE_SNAPSHOT_START
        snapshot_stop = ObservationEncoder.PAID_ACTION_HISTORY_START
        self.assertFalse(neutral[snapshot_start:snapshot_stop].any())

        move_action(game, first)
        captured = game.normal_move_pre_board_snapshot
        route_index = game.selected_map.routes.index(route)
        self.assertEqual(
            captured[route_index][route.posts.index(first)],
            (player, "square"),
        )
        self.assertEqual(
            captured[route_index][route.posts.index(second)],
            (player, "circle"),
        )
        after_first = ObservationEncoder().get_game_state(game)
        encoded_after_first = after_first[snapshot_start:snapshot_stop].clone()
        move_action(game, second)
        after_second = ObservationEncoder().get_game_state(game)
        self.assertIs(game.normal_move_pre_board_snapshot, captured)
        self.assertTrue(
            torch.equal(
                encoded_after_first,
                after_second[snapshot_start:snapshot_stop],
            )
        )
        self.assertFalse(torch.equal(after_first, after_second))

        move_action(game, destination)
        after_placement = ObservationEncoder().get_game_state(game)
        self.assertIs(game.normal_move_pre_board_snapshot, captured)
        self.assertTrue(
            torch.equal(
                encoded_after_first,
                after_placement[snapshot_start:snapshot_stop],
            )
        )
        self.assertFalse(torch.equal(after_second, after_placement))
        move_action(game, first)
        post_routes = {
            post: candidate_route
            for candidate_route in game.selected_map.routes
            for post in candidate_route.posts
        }
        self.assertEqual(
            pointless_movement_penalty(
                [(first, player, "square"), (second, player, "circle")],
                [destination, first],
                post_routes,
            ),
            0,
        )
        self.assertIsNone(game.normal_move_pre_board_snapshot)
        self.assertFalse(
            ObservationEncoder().get_game_state(game)[snapshot_start:snapshot_stop].any()
        )

    def test_move_any_2_reuses_and_clears_pre_move_snapshot(self):
        game = self.game()
        player = game.current_player
        opponent = game.players[1]
        route = next(
            route
            for route in game.selected_map.routes
            if sum(post.required_shape is None for post in route.posts) >= 4
        )
        own_post = self._open_post(route)
        opponent_post = self._open_post(route, excluded=(own_post,))
        first_destination = self._open_post(route, excluded=(own_post, opponent_post))
        second_destination = self._open_post(
            route,
            excluded=(own_post, opponent_post, first_destination),
        )
        own_post.claim(player, "square")
        opponent_post.claim(opponent, "square")

        game.waiting_for_bm_move_any_2 = True
        player.pieces_to_pickup = 2
        move_action(game, own_post)
        snapshot = game.normal_move_pre_board_snapshot
        self.assertIsNotNone(snapshot)
        move_action(game, opponent_post)
        self.assertIs(game.normal_move_pre_board_snapshot, snapshot)
        move_action(game, first_destination)
        move_action(game, second_destination)
        self.assertFalse(game.waiting_for_bm_move_any_2)
        self.assertIsNone(game.normal_move_pre_board_snapshot)

    def test_move_3_does_not_capture_pre_move_snapshot(self):
        game = self.game()
        player = game.current_player
        opponent = game.players[1]
        route = game.selected_map.routes[0]
        opponent_post = self._open_post(route)
        opponent_post.claim(opponent, "square")
        player.pieces_to_pickup = 3
        game.waiting_for_bm_move3 = True
        move_action(game, opponent_post)
        self.assertIsNone(game.normal_move_pre_board_snapshot)

    def test_move_any_2_snapshot_breaks_origin_alias(self):
        encoder = ObservationEncoder()
        first = self.game()
        player = first.current_player
        route = next(
            route
            for route in first.selected_map.routes
            if sum(post.required_shape is None for post in route.posts) >= 2
        )
        origin = self._open_post(route)
        origin.claim(player, "square")
        first.waiting_for_bm_move_any_2 = True
        player.pieces_to_pickup = 2
        move_action(first, origin)
        second = deepcopy(first)

        route_slot = first.selected_map.routes.index(route)
        origin_slot = route.posts.index(origin)
        altered = [list(posts) for posts in second.normal_move_pre_board_snapshot]
        altered[route_slot][origin_slot] = (None, None)
        second.normal_move_pre_board_snapshot = tuple(tuple(posts) for posts in altered)

        first_observation = encoder.build(first)
        second_observation = encoder.build(second)
        snapshot_start = encoder.MOVE_SNAPSHOT_START
        snapshot_stop = encoder.PAID_ACTION_HISTORY_START
        self.assertTrue(
            torch.equal(
                first_observation.features[:snapshot_start],
                second_observation.features[:snapshot_start],
            )
        )
        self.assertFalse(
            torch.equal(
                first_observation.features[snapshot_start:snapshot_stop],
                second_observation.features[snapshot_start:snapshot_stop],
            )
        )
        self.assertTrue(
            torch.equal(first_observation.legal_action_mask, second_observation.legal_action_mask)
        )

    def test_move_claim_route_flags_break_claim_provenance_alias(self):
        encoder = ObservationEncoder()
        pending = self.game()
        route_slot = 3
        not_pending = deepcopy(pending)
        pending.current_player.pending_move_claim_route_slots = frozenset({route_slot})

        pending_observation = encoder.build(pending)
        plain_observation = encoder.build(not_pending)
        self.assertTrue(
            torch.equal(pending_observation.features[:4644], plain_observation.features[:4644])
        )
        self.assertTrue(
            torch.equal(pending_observation.legal_action_mask, plain_observation.legal_action_mask)
        )
        base = encoder.ROUTE_REWARD_HISTORY_START + route_slot * 2
        self.assertEqual(pending_observation.features[base].item(), 1)
        self.assertEqual(plain_observation.features[base].item(), 0)

        action = RouteInteraction(route_slot, 0)
        cleared, rewarded = update_move_claim_combo(
            pending.current_player.pending_move_claim_route_slots,
            action=action,
            turn_phase=TurnPhase.ACTIONS,
            action_was_spent=True,
        )
        self.assertEqual(rewarded, 250)
        self.assertEqual(cleared, frozenset())
        _, unrewarded = update_move_claim_combo(
            not_pending.current_player.pending_move_claim_route_slots,
            action=action,
            turn_phase=TurnPhase.ACTIONS,
            action_was_spent=True,
        )
        self.assertEqual(unrewarded, 0)

    def test_move_focus_route_flags_break_cooldown_alias_and_clear_on_claim(self):
        encoder = ObservationEncoder()
        available = self._active_move_state()
        route_slot = 4
        consumed = deepcopy(available)
        consumed.current_player.rewarded_move_focus_route_slots = frozenset({route_slot})

        available_observation = encoder.build(available)
        consumed_observation = encoder.build(consumed)
        self.assertTrue(
            torch.equal(
                available_observation.features[:4644],
                consumed_observation.features[:4644],
            )
        )
        self.assertTrue(
            torch.equal(
                available_observation.legal_action_mask,
                consumed_observation.legal_action_mask,
            )
        )
        base = encoder.ROUTE_REWARD_HISTORY_START + route_slot * 2
        self.assertEqual(available_observation.features[base + 1].item(), 0)
        self.assertEqual(consumed_observation.features[base + 1].item(), 1)
        _, available_reward = move_route_focus_reward(frozenset(), {route_slot: 2})
        _, consumed_reward = move_route_focus_reward(frozenset({route_slot}), {route_slot: 2})
        self.assertEqual(available_reward, 10)
        self.assertEqual(consumed_reward, 0)
        self.assertEqual(
            clear_move_route_focus_after_claim(
                frozenset({route_slot}),
                RouteInteraction(route_slot, 0),
                TurnPhase.ACTIONS,
            ),
            frozenset(),
        )

    def test_pending_move_claim_flags_reset_at_turn_start(self):
        game = self.game()
        player = game.current_player
        player.pending_move_claim_route_slots = frozenset({1, 2})
        player.rewarded_move_focus_route_slots = frozenset({3})
        player.start_turn()
        self.assertEqual(player.pending_move_claim_route_slots, frozenset())
        self.assertEqual(player.rewarded_move_focus_route_slots, frozenset({3}))

    def test_structural_cache_preserves_observation_contract_after_dynamic_changes(self):
        expected = {
            1: (
                "84d220246145fb38026d37cd09e0a7fed07f6617c524086d1e436d3684088525",
                "8f206d126436f924cbf0a44e23eeaf19f367e83347906045f06cb002bf7fb06a",
            ),
            2: (
                "3555589e010a79805cb080148144603179c2b77e9e9e9a881aa82fc9b1c69fc9",
                "d4daaa7e4ae574b920c6066374aac15207a024ae3932e24380e4135321b05a37",
            ),
            3: (
                "1cc0d1eb2c86694d306f3b3b5af91f4d664a51d49c4bf171b048dc37cda4b581",
                "6576a26b40ff98c704a1cf8f94703275174908156a57e2e5ec68307f103e2967",
            ),
        }
        for map_num in (1, 2, 3):
            game = self.game(map_num=map_num, players=5)
            encoder = ObservationEncoder()
            initial = encoder.get_game_state(game)
            self.assertEqual(
                hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                expected[map_num][0],
            )

            player = game.current_player
            player.score += 2
            player.general_stock_squares += 1
            route = game.selected_map.routes[0]
            route.posts[0].claim(player, route.posts[0].required_shape or "square")
            city = game.selected_map.cities[0]
            city.offices[0].controller = player
            city.create_new_office(player.color).controller = game.players[1]
            game.active_player = 1
            changed = encoder.get_game_state(game)
            self.assertEqual(
                hashlib.sha256(changed.numpy().tobytes()).hexdigest(),
                expected[map_num][1],
            )

    def test_structural_signature_is_not_recomputed_without_invalidation(self):
        game = self.game()
        encoder = ObservationEncoder()

        with mock.patch.object(
            encoder,
            "_structural_signature",
            wraps=encoder._structural_signature,
        ) as signature:
            first = encoder.get_game_state(game)
            second = encoder.get_game_state(game)

        self.assertTrue(torch.equal(first, second))
        signature.assert_called_once_with(game)

    def test_fixed_shape_and_legal_mask_share_acting_player(self):
        game = self.game(use_mission_cards=True)
        observation = self.observation(game)

        self.assertEqual(observation.features.shape, (ObservationEncoder.FEATURE_SIZE,))
        self.assertEqual(observation.features.dtype, torch.int16)
        self.assertEqual(observation.legal_action_mask.shape, (768,))
        self.assertEqual(observation.observer_index, game.active_player)
        self.assertTrue(
            torch.equal(
                observation.legal_action_mask,
                torch.tensor(game.ai_action_mask(), dtype=torch.uint8),
            )
        )

    def test_identical_states_are_deterministic(self):
        first = self.game(use_mission_cards=True, use_emperors_favour=True)
        second = self.game(use_mission_cards=True, use_emperors_favour=True)

        self.assertTrue(
            torch.equal(self.observation(first).features, self.observation(second).features)
        )

    def test_public_state_changes_observation(self):
        game = self.game()
        before = self.observation(game).features
        game.players[1].score += 1
        after = self.observation(game).features

        self.assertFalse(torch.equal(before, after))

    def test_only_acting_players_mission_card_is_visible(self):
        game = self.game(use_mission_cards=True)
        encoder = ObservationEncoder()
        acting = game.players[0]
        opponent = game.players[1]

        original = encoder.get_game_state(game)
        opponent.mission_card = list(reversed(opponent.mission_card))
        self.assertTrue(torch.equal(original, encoder.get_game_state(game)))

        acting.mission_card = list(reversed(acting.mission_card))
        self.assertFalse(torch.equal(original, encoder.get_game_state(game)))

    def test_opponent_used_marker_types_are_hidden_until_exchange_target_is_selected(self):
        game = self.game()
        encoder = ObservationEncoder()
        opponent = game.players[1]
        opponent.used_bonus_markers = [
            BonusMarker("Move3", owner=opponent),
            BonusMarker("SwapOffice", owner=opponent),
        ]
        hidden = encoder.get_game_state(game)
        opponent_start = encoder.GAME_SIZE + encoder.PLAYER_SIZE
        used_start = opponent_start + 28
        self.assertEqual(
            hidden[used_start : used_start + 15].tolist(),
            [encoder.HIDDEN_USED_BONUS_MARKER_ID] * 2 + [0] * 13,
        )

        opponent.used_bonus_markers = [
            BonusMarker("3Actions", owner=opponent),
            BonusMarker("BlockTradeRoute", owner=opponent),
        ]
        self.assertTrue(torch.equal(hidden, encoder.get_game_state(game)))

        game.waiting_for_bm_exchange_bm = True
        game.exchange_target_player = opponent
        visible = encoder.get_game_state(game)
        self.assertEqual(
            visible[used_start : used_start + 15].tolist(),
            [
                encoder.BONUS_MARKER_TYPE_TO_ID["3Actions"],
                encoder.BONUS_MARKER_TYPE_TO_ID["BlockTradeRoute"],
                *([0] * 13),
            ],
        )
        self.assertFalse(torch.equal(hidden, visible))

    def test_relative_player_slots_rotate_with_active_player(self):
        game = self.game()
        game.players[0].score = 7
        game.players[1].score = 11
        encoder = ObservationEncoder()

        first = encoder.get_game_state(game)
        game.active_player = 1
        second = encoder.get_game_state(game)
        player_start = ObservationEncoder.GAME_SIZE

        self.assertEqual(first[player_start + 1].item(), 7)
        self.assertEqual(second[player_start + 1].item(), 11)

    def test_displaced_player_is_observer_and_mask_owner(self):
        game = self.game(map_num=2)
        displaced = game.players[1]
        route = game.selected_map.routes[0]
        game.original_route_of_displacement = route
        game.waiting_for_displaced_player = True
        game.displaced_player.populate_displaced_player(game, displaced, "square")
        game.active_player = 1
        refresh_displacement_targets(game)

        observation = self.observation(game)

        self.assertEqual(observation.observer_index, 1)
        self.assertTrue(
            torch.equal(
                observation.legal_action_mask,
                torch.tensor(game.ai_action_mask(), dtype=torch.uint8),
            )
        )

        current = game.current_player
        current.general_stock_squares = 0
        current.general_stock_circles = 0
        displaced.general_stock_squares = 1
        with_optional_piece = game.get_legal_actions()
        current.general_stock_squares = 5
        current.general_stock_circles = 5
        self.assertEqual(with_optional_piece, game.get_legal_actions())
        displaced.general_stock_squares = 0
        displaced.general_stock_circles = 0
        displaced.personal_supply_squares = 0
        displaced.personal_supply_circles = 0
        self.assertNotEqual(
            any(isinstance(action, SupplyInteraction) for action in with_optional_piece),
            any(isinstance(action, SupplyInteraction) for action in game.get_legal_actions()),
        )

    def test_income_response_uses_responder_resources_not_current_player(self):
        game = self.game(map_num=2)
        current = game.current_player
        responder = game.players[1]
        current.general_stock_squares = 2
        current.general_stock_circles = 0
        responder.general_stock_squares = 0
        responder.general_stock_circles = 1
        game.OneIncomeIfOthersIncomeOwner = responder
        game.begin_income_favour_response(current)

        observation = self.observation(game)
        choices = {
            action.tile_slot
            for action in game.get_legal_actions()
            if isinstance(action, TileInteraction)
        }

        self.assertEqual(observation.observer_index, 1)
        self.assertEqual(choices, {1, 2})

    def test_visible_future_transition_state_changes_observation(self):
        game = self.game()
        encoder = ObservationEncoder()
        baseline = encoder.get_game_state(game)

        game.players_who_completed_east_west.add(game.players[1])
        self.assertFalse(torch.equal(baseline, encoder.get_game_state(game)))
        baseline = encoder.get_game_state(game)
        game.bonus_pool_exhausted_during_claim = True
        self.assertFalse(torch.equal(baseline, encoder.get_game_state(game)))
        baseline = encoder.get_game_state(game)
        game.pending_tribute_income_owners = [game.players[1]]
        game.active_player = 1
        self.assertFalse(torch.equal(baseline, encoder.get_game_state(game)))
        baseline = encoder.get_game_state(game)
        game.game_end_pending_immediate_resolution = True
        self.assertFalse(torch.equal(baseline, encoder.get_game_state(game)))

    def test_hidden_bonus_marker_order_does_not_change_observation(self):
        game = self.game()
        before = self.observation(game).features
        game.selected_map.bonus_marker_pool.reverse()

        self.assertTrue(torch.equal(before, self.observation(game).features))

    def test_drawn_replacement_marker_is_hidden_until_placement(self):
        game = self.game()
        encoder = ObservationEncoder()
        game.pending_bonus_markers = ["Move3"]
        game.replace_bonus_marker = 1

        hidden = encoder.get_game_state(game)
        game.current_player.actions_remaining = 0
        game.current_player.ending_turn = True
        visible = encoder.get_game_state(game)

        optional_start = (
            encoder.GAME_SIZE
            + encoder.MAX_PLAYERS * encoder.PLAYER_SIZE
            + encoder.MAX_CITIES * encoder.CITY_SIZE
            + encoder.MAX_ROUTES * encoder.ROUTE_SIZE
        )
        pending_start = optional_start + 18
        self.assertEqual(hidden[pending_start].item(), 0)
        self.assertEqual(visible[pending_start].item(), encoder.BONUS_MARKER_TYPE_TO_ID["Move3"])

    def test_capacity_overflow_fails_instead_of_truncating(self):
        game = self.game(map_num=2)
        city = game.selected_map.cities[0]
        city.offices.extend([city.offices[0]] * (11 - len(city.offices)))

        with self.assertRaisesRegex(ValueError, "capacity is 10"):
            self.observation(game)

    def test_all_supported_setups_fit_the_schema(self):
        for map_num in (1, 2, 3):
            for player_count in (3, 4, 5):
                with self.subTest(map_num=map_num, player_count=player_count):
                    game = self.game(
                        map_num=map_num,
                        players=player_count,
                        use_mission_cards=map_num == 1,
                        use_emperors_favour=True,
                    )
                    self.assertEqual(
                        self.observation(game).features.numel(),
                        ObservationEncoder.FEATURE_SIZE,
                    )

    def test_category_ids_are_stable(self):
        self.assertEqual(
            ObservationEncoder.PIECE_TYPE_TO_ID,
            {None: 0, "square": 1, "circle": 2},
        )
        self.assertEqual(
            ObservationEncoder.BONUS_MARKER_TYPE_TO_ID,
            {
                "PlaceAdjacent": 1,
                "SwapOffice": 2,
                "Move3": 3,
                "UpgradeAbility": 4,
                "3Actions": 5,
                "4Actions": 6,
                "ExchangeBonusMarker": 7,
                "Tribute4EstablishingTP": 8,
                "BlockTradeRoute": 9,
            },
        )
        self.assertEqual(
            ObservationEncoder.PERMANENT_MARKER_TYPE_TO_ID,
            {
                None: 0,
                "MoveAny2": 1,
                "+1Priv": 2,
                "ClaimGreenCity": 3,
                "Place2TradesmenFromRoute": 4,
                "Place2ScotlandOrWales": 5,
            },
        )
        self.assertEqual(
            ObservationEncoder.UPGRADE_TYPE_TO_ID,
            {
                "Keys": 1,
                "Privilege": 2,
                "Book": 3,
                "Actions": 4,
                "Bank": 5,
                "SpecialPrestigePoints": 6,
            },
        )
        self.assertEqual(
            ObservationEncoder.PRIVILEGE_TO_ID,
            {None: 0, "WHITE": 1, "ORANGE": 2, "PINK": 3, "BLACK": 4},
        )
        self.assertEqual(
            ObservationEncoder.REGION_TO_ID,
            {None: 0, "Scotland": 1, "Wales": 2},
        )
        self.assertEqual(
            ObservationEncoder.PHASE_TO_ID,
            {
                TurnPhase.ACTIONS: 0,
                TurnPhase.DISPLACEMENT: 1,
                TurnPhase.MOVE_PIECES: 2,
                TurnPhase.BONUS_MARKER_CHOICE: 3,
                TurnPhase.BUY_TILE_PAYMENT: 4,
                TurnPhase.INCOME_FAVOUR_RESPONSE: 5,
                TurnPhase.TRIBUTE_INCOME_RESPONSE: 6,
                TurnPhase.PLACE_ADJACENT_ROUTE: 7,
                TurnPhase.PERMANENT_ROUTE_PIECE_SELECTION: 8,
                TurnPhase.REPLACE_BONUS_MARKERS: 9,
                TurnPhase.TURN_COMPLETE: 10,
                TurnPhase.GAME_OVER: 11,
            },
        )
        self.assertEqual(
            ObservationEncoder.CITY_TYPE_TO_ID,
            {
                tuple(GREY): 1,
                tuple(BLACKISH_BROWN): 2,
                tuple(DARK_RED): 3,
                tuple(DARK_GREEN): 4,
                tuple(DARK_BLUE): 5,
                (65, 103, 114): 6,
            },
        )
        self.assertEqual(
            ObservationEncoder.ROUTE_TYPE_TO_ID,
            {tuple(WHITE): 1, tuple(BLACKISH_BROWN): 2},
        )

    def test_feature_groups_and_representative_field_offsets_are_stable(self):
        encoder = ObservationEncoder()
        game = self.game(map_num=2, use_emperors_favour=True)

        player_start = encoder.GAME_SIZE
        city_start = player_start + encoder.MAX_PLAYERS * encoder.PLAYER_SIZE
        route_start = city_start + encoder.MAX_CITIES * encoder.CITY_SIZE
        optional_start = route_start + encoder.MAX_ROUTES * encoder.ROUTE_SIZE
        workflow_start = optional_start + encoder.OPTIONAL_COMPONENTS_SIZE
        snapshot_start = workflow_start + encoder.WORKFLOW_SIZE
        paid_action_history_start = snapshot_start + encoder.MOVE_SNAPSHOT_SIZE
        route_reward_history_start = paid_action_history_start + encoder.PAID_ACTION_HISTORY_SIZE
        self.assertEqual(
            route_reward_history_start + encoder.ROUTE_REWARD_HISTORY_SIZE,
            encoder.FEATURE_SIZE,
        )
        self.assertEqual(paid_action_history_start, 4641)
        self.assertEqual(route_reward_history_start, 4644)

        def changed_indices(before, after):
            return (before != after).nonzero().flatten().tolist()

        before = encoder.get_game_state(game)
        game.current_player.score += 1
        after = encoder.get_game_state(game)
        self.assertEqual(changed_indices(before, after), [player_start + 1])

        before = after
        game.selected_map.cities[0].offices[0].awards_points += 1
        after = encoder.get_game_state(game)
        self.assertEqual(changed_indices(before, after), [city_start + 12])

        route = game.selected_map.routes[0]
        post = next(post for post in route.posts if post.required_shape is None)
        post_index = route.posts.index(post)
        before = after
        post.required_shape = "circle"
        game.mark_observation_structure_changed()
        after = encoder.get_game_state(game)
        self.assertEqual(
            changed_indices(before, after),
            [route_start + 18 + post_index * 4 + 1],
        )

        tile = game.tile_pool[0]
        tile_slot = list(encoder.TILE_TYPE_TO_ID).index(tile)
        before = after
        game.tile_pool.remove(tile)
        after = encoder.get_game_state(game)
        self.assertEqual(changed_indices(before, after), [optional_start + tile_slot])

        before = after
        game.current_player.pieces_to_pickup += 1
        after = encoder.get_game_state(game)
        self.assertEqual(changed_indices(before, after), [workflow_start + 15])
        self.assertEqual(
            ObservationEncoder.TILE_TYPE_TO_ID,
            {
                "DisplaceAnywhere": 1,
                "+1Action": 2,
                "+1IncomeIfOthersIncome": 3,
                "+1DisplacedPiece": 4,
                "+4PtsPerOwnedCity": 5,
                "+7PtsPerCompletedAbility": 6,
            },
        )

    def test_map_city_and_route_catalogue_order_is_stable(self):
        expected = {
            1: (
                "eedad223ad98b7ca3b28fe881c51cf0a0d1d8119933a57055ec9b3f06b20e8bc",
                "25c41ec96d415f768d96a254b3d67a67c330cc0d5334593fda8f56e4f3c00e33",
            ),
            2: (
                "c00ee2a6037784db40245c27eca04c8e918685319233fa76bf70641a941b953f",
                "030e9c065b5d67644543326a6062d3040db9c6a2080eb68d5627d3fa54ca53e6",
            ),
            3: (
                "5a893f10b8549555adce98ba156c23de93dc3ed59abf2a8c09e7eae3155748d1",
                "80fd09bc749e7bb147aec79a6294892c6ab1b7953c74177b2dcb3c7a0b0ab552",
            ),
        }
        for map_num, fingerprints in expected.items():
            game = self.game(map_num=map_num)
            cities = json.dumps(
                [city.name for city in game.selected_map.cities], separators=(",", ":")
            )
            routes = json.dumps(
                [[city.name for city in route.cities] for route in game.selected_map.routes],
                separators=(",", ":"),
            )
            self.assertEqual(
                (
                    hashlib.sha256(cities.encode()).hexdigest(),
                    hashlib.sha256(routes.encode()).hexdigest(),
                ),
                fingerprints,
            )

    def test_observation_module_has_no_gui_dependency(self):
        imported_before = set(sys.modules)
        game = self.game()
        self.observation(game)
        newly_imported = set(sys.modules) - imported_before

        self.assertFalse(
            any(name == "drawing" or name.startswith("drawing.") for name in newly_imported)
        )


if __name__ == "__main__":
    unittest.main()
