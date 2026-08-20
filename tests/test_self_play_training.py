from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from ai.ai_model import HansaNN, device
from ai.observation_encoder import ObservationEncoder
from ai.observation_schema import LEGACY_OBSERVATION_SCHEMA_V1_FINGERPRINT
from game.action_codec import DEFAULT_ACTION_CODEC
from game.action_schema import ACTION_SCHEMA_VERSION, ACTION_SPACE_SIZE
from game.persistence import load_game
from game.turn_state import TurnPhase
from map_data.constants import MAX_POSTS
from training.self_play import (
    ALL_MOVE_TURN_LOCAL_TARGET,
    MOVE_CLAIM_COMBO_REWARD,
    MOVE_ROUTE_FOCUS_REWARD,
    MovementBehaviorMetrics,
    POINTLESS_ROUTE_CLAIM_PENALTY,
    NO_REPLACEMENT_ROUTE_PENALTY,
    PolicyTier,
    SelfPlayTrainer,
    TrainingConfig,
    TrainingDecision,
    _is_normal_move_in_progress,
    action_phase_selection_groups,
    apply_all_move_turn_target,
    apply_income_efficiency_penalty,
    apply_movement_efficiency_penalty,
    apply_opponent_route_score_penalty,
    apply_route_completion_reward,
    assign_reward_to_go,
    assign_training_targets,
    calculate_terminal_rewards,
    completed_game_reason,
    completed_route_move_reward,
    consecutive_move_penalty,
    credited_movement_workflows,
    grant_movement_workflow_terminal_credit,
    income_efficiency_penalty,
    inverse_sqrt_rank_weights,
    intermediate_ability_upgrade_reward,
    mark_movement_workflow_target,
    move_route_focus_reward,
    movement_efficiency_penalty,
    move_workflow_exploration_categories,
    normalized_rank_weights,
    pointless_movement_penalty,
    pointless_route_claim_penalty,
    route_building_post_reward,
    should_fully_validate,
    training_action_mask,
    update_move_claim_combo,
    valuable_completed_route_slots,
)
from game.structured_actions import (
    IncomeInteraction,
    PieceShape,
    PostInteraction,
    RouteInteraction,
)
from tests.action_helpers import self_play_test_state


STATE = self_play_test_state()


def training_decision(
    action,
    player,
    rewards,
    immediate,
    turn=1,
    movement_workflow_id=None,
    equivalent_action_indices=(),
    receives_terminal_credit=True,
):
    empty = torch.empty(0)
    return TrainingDecision(
        empty,
        empty,
        action,
        player,
        rewards,
        immediate,
        1,
        0.05,
        2,
        False,
        1,
        1,
        turn,
        movement_workflow_id,
        equivalent_action_indices=equivalent_action_indices,
        receives_terminal_credit=receives_terminal_credit,
    )


class SelfPlayTrainingTests(unittest.TestCase):
    def test_movement_behavior_metric_rates_are_blank_without_denominators(self):
        metrics = MovementBehaviorMetrics()

        self.assertIsNone(metrics.move_ratio)
        self.assertIsNone(metrics.move_claim_conversion_rate)

        metrics.move_action_count = 3
        metrics.spent_action_count = 12
        metrics.moves_creating_claimable_route = 4
        metrics.move_claim_conversions = 3

        self.assertEqual(metrics.move_ratio, 0.25)
        self.assertEqual(metrics.move_claim_conversion_rate, 0.75)

    def test_completed_game_reason_reports_authoritative_end_conditions(self):
        game = mock.Mock()
        game.players = [mock.Mock(score=20), mock.Mock(score=12)]
        game.bonus_pool_exhausted_during_claim = True
        game.current_full_cities_count = 10
        game.selected_map.max_full_cities = 10

        self.assertEqual(
            completed_game_reason(game),
            "20_points+bonus_markers_exhausted+full_cities",
        )

    def test_completed_game_reason_reports_each_condition_individually(self):
        game = mock.Mock()
        game.players = [mock.Mock(score=19)]
        game.bonus_pool_exhausted_during_claim = False
        game.current_full_cities_count = 9
        game.selected_map.max_full_cities = 10

        game.players[0].score = 20
        self.assertEqual(completed_game_reason(game), "20_points")
        game.players[0].score = 19
        game.bonus_pool_exhausted_during_claim = True
        self.assertEqual(completed_game_reason(game), "bonus_markers_exhausted")
        game.bonus_pool_exhausted_during_claim = False
        game.current_full_cities_count = 10
        self.assertEqual(completed_game_reason(game), "full_cities")

    def test_valuable_completed_routes_include_marker_upgrade_and_two_point_routes(self):
        player = mock.Mock()
        player.actions = 2
        city1 = mock.Mock(upgrade_city_type=[], determine_controller=lambda: None)
        city2 = mock.Mock(upgrade_city_type=[], determine_controller=lambda: None)
        route = mock.Mock(
            posts=(),
            cities=(city1, city2),
            bonus_marker=object(),
            permanent_bonus_marker=None,
        )
        route.is_controlled_by.return_value = True
        game = mock.Mock(
            selected_map=mock.Mock(routes=(route,), east_west_cities=("A", "B")),
            players_who_completed_east_west={player},
        )

        self.assertEqual(valuable_completed_route_slots(game, player), {0})

        route.bonus_marker = None
        city1.upgrade_city_type = ["Actions"]
        self.assertEqual(valuable_completed_route_slots(game, player), {0})

        city1.upgrade_city_type = []
        city1.determine_controller = lambda: player
        city2.determine_controller = lambda: player
        self.assertEqual(valuable_completed_route_slots(game, player), {0})

    def test_full_validation_runs_periodically_and_at_workflow_boundaries(self):
        game = mock.Mock(turn_number=4, turn_phase=TurnPhase.ACTIONS, game_end=False)
        self.assertFalse(should_fully_validate(1, 50, 4, TurnPhase.ACTIONS, game))
        self.assertTrue(should_fully_validate(50, 50, 4, TurnPhase.ACTIONS, game))
        self.assertTrue(should_fully_validate(1, 50, 3, TurnPhase.ACTIONS, game))
        self.assertTrue(should_fully_validate(1, 50, 4, TurnPhase.DISPLACEMENT, game))

    def trainer(self, seed=124):
        return SelfPlayTrainer(
            config=TrainingConfig(
                learning_rate=0.0001,
                max_actions=100,
                disable_move_action=True,
                move_general_stock_threshold=0,
                seed=seed,
            )
        )

    def test_move_filter_leaves_the_winning_route_interaction(self):
        game = load_game(STATE)

        mask = training_action_mask(game, disable_move_action=True, move_general_stock_threshold=0)

        self.assertEqual(mask.numel(), ACTION_SPACE_SIZE)
        legal_indices = mask.nonzero(as_tuple=False).flatten().tolist()
        self.assertTrue(legal_indices)
        self.assertTrue(
            all(
                isinstance(DEFAULT_ACTION_CODEC.decode(index), RouteInteraction)
                for index in legal_indices
            )
        )

    def test_training_mask_reuses_observation_mask(self):
        game = mock.Mock()
        game.ai_action_mask.side_effect = AssertionError("legality was recalculated")
        base_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.uint8)
        base_mask[285] = 1

        mask = training_action_mask(
            game,
            disable_move_action=False,
            base_mask=base_mask,
        )

        self.assertEqual(mask.nonzero(as_tuple=False).flatten().tolist(), [285])
        game.ai_action_mask.assert_not_called()

    def test_route_claim_penalizes_opponent_points_given(self):
        adjusted = apply_opponent_route_score_penalty(
            (200.0, -100.0, 100.0),
            action=RouteInteraction(0, 0),
            turn_phase=TurnPhase.ACTIONS,
            acting_player_index=0,
            projected_reward_deltas=(200.0, -100.0, 100.0),
        )

        self.assertEqual(adjusted, (100.0, -100.0, 100.0))

    def test_route_completion_adds_small_incentive_before_opponent_costs(self):
        action = RouteInteraction(0, 0)
        rewarded = apply_route_completion_reward(
            (0.0, 0.0, 100.0),
            action=action,
            turn_phase=TurnPhase.ACTIONS,
            acting_player_index=0,
        )
        adjusted = apply_opponent_route_score_penalty(
            rewarded,
            action=action,
            turn_phase=TurnPhase.ACTIONS,
            acting_player_index=0,
            projected_reward_deltas=(0.0, 0.0, 100.0),
        )

        self.assertEqual(rewarded, (50.0, 0.0, 100.0))
        self.assertEqual(adjusted, (-50.0, 0.0, 100.0))

    def test_non_route_action_does_not_penalize_opponent_score_changes(self):
        original = (100.0, 100.0, 200.0)

        adjusted = apply_opponent_route_score_penalty(
            original,
            action=PostInteraction(0, "square"),
            turn_phase=TurnPhase.ACTIONS,
            acting_player_index=0,
            projected_reward_deltas=(100.0, 100.0, 200.0),
        )

        self.assertEqual(adjusted, original)

    def test_full_income_has_no_efficiency_penalty(self):
        for capacity in (3, 5, 7):
            with self.subTest(capacity=capacity):
                self.assertEqual(income_efficiency_penalty(capacity, capacity, 100), 0)

    def test_partial_income_penalty_is_proportional(self):
        examples = (
            (3, 2, -100 / 3),
            (3, 1, -200 / 3),
            (5, 4, -20),
            (5, 2, -60),
            (7, 6, -100 / 7),
            (7, 3, -400 / 7),
        )
        for capacity, received, expected in examples:
            with self.subTest(capacity=capacity, received=received):
                self.assertAlmostEqual(income_efficiency_penalty(capacity, received, 100), expected)
        self.assertLess(
            income_efficiency_penalty(3, 1, 100),
            income_efficiency_penalty(3, 2, 100),
        )

    def test_bank_all_has_no_efficiency_penalty(self):
        self.assertEqual(income_efficiency_penalty(50, 1, 100), 0)

    def test_income_penalty_scale_is_configurable(self):
        self.assertEqual(income_efficiency_penalty(5, 2, 50), -30)
        self.assertEqual(income_efficiency_penalty(5, 2, 0), 0)

    def test_income_penalty_changes_only_its_acting_players_reward(self):
        original = (100.0, 200.0, 300.0)
        adjusted = apply_income_efficiency_penalty(
            original,
            action=IncomeInteraction(0),
            turn_phase=TurnPhase.ACTIONS,
            acting_player_index=1,
            bank_capacity=5,
            pieces_received=2,
            scale=100,
        )

        self.assertEqual(adjusted, (100.0, 140.0, 300.0))
        self.assertEqual(original, (100.0, 200.0, 300.0))
        self.assertEqual(
            apply_income_efficiency_penalty(
                original,
                action=PostInteraction(0, "square"),
                turn_phase=TurnPhase.ACTIONS,
                acting_player_index=1,
                bank_capacity=5,
                pieces_received=2,
                scale=100,
            ),
            original,
        )
        self.assertEqual(
            apply_income_efficiency_penalty(
                original,
                action=IncomeInteraction(0),
                turn_phase=TurnPhase.TRIBUTE_INCOME_RESPONSE,
                acting_player_index=1,
                bank_capacity=5,
                pieces_received=2,
                scale=100,
            ),
            original,
        )

    def test_move_filter_allows_moves_when_only_moves_exist(self):
        class Player:
            pass

        class Post:
            def __init__(self, owner):
                self.owner = owner
                self.required_shape = None
                self.region = None

            def is_owned(self):
                return self.owner is not None

        class Route:
            def __init__(self, posts):
                self.posts = posts

        class Game:
            def __init__(self, active_player, mask):
                self.players = [Player()]
                self.active_player = active_player
                self.turn_phase = TurnPhase.ACTIONS
                self.selected_map = type("Map", (), {"routes": [Route([Post(self.players[0])])]})()
                self._mask = mask

            @property
            def current_player(self):
                return self.players[self.active_player]

            def ai_action_mask(self):
                return self._mask

        base_mask = [False] * ACTION_SPACE_SIZE
        base_mask[0] = True
        base_mask[MAX_POSTS] = True
        game = Game(active_player=0, mask=base_mask)

        mask = training_action_mask(game, disable_move_action=True)

        self.assertEqual(mask.nonzero(as_tuple=False).flatten().tolist(), [0, MAX_POSTS])

    def test_move_filter_allows_moves_with_fewer_than_three_general_stock_pieces(self):
        game = load_game(STATE)
        player = game.current_player
        player.general_stock_squares = 1
        player.general_stock_circles = 1
        owned_post = next(
            post
            for route in game.selected_map.routes
            for post in route.posts
            if post.owner == player
        )
        post_slot = next(
            index
            for index, post in enumerate(
                post for route in game.selected_map.routes for post in route.posts
            )
            if post is owned_post
        )

        mask = training_action_mask(game, disable_move_action=True)

        self.assertTrue(mask[post_slot] or mask[MAX_POSTS + post_slot])

    def test_normal_move_efficiency_penalties(self):
        self.assertEqual(movement_efficiency_penalty(1, 2), -200)
        self.assertEqual(movement_efficiency_penalty(1, 5), -200)
        self.assertEqual(movement_efficiency_penalty(2, 2), 0)
        self.assertEqual(movement_efficiency_penalty(2, 3), -100)
        self.assertEqual(movement_efficiency_penalty(2, 5), -100)
        self.assertEqual(movement_efficiency_penalty(3, 5), 0)
        self.assertEqual(movement_efficiency_penalty(4, 5), 0)

    def test_consecutive_move_penalties_follow_book_capacity(self):
        self.assertEqual(consecutive_move_penalty(2, 1), 0)
        self.assertEqual(consecutive_move_penalty(2, 2), 0)
        self.assertEqual(consecutive_move_penalty(2, 3), -1500)
        self.assertEqual(consecutive_move_penalty(3, 3), -1500)
        self.assertEqual(consecutive_move_penalty(4, 1), 0)
        self.assertEqual(consecutive_move_penalty(4, 2), -200)
        self.assertEqual(consecutive_move_penalty(4, 3), -1500)
        self.assertEqual(consecutive_move_penalty(4, 4), -1500)
        self.assertEqual(consecutive_move_penalty(5, 3), -1500)

    def test_normal_move_tracking_uses_the_held_piece_phase(self):
        player = type("Player", (), {"holding_pieces": [("square", None, None)]})()

        self.assertTrue(_is_normal_move_in_progress(TurnPhase.MOVE_PIECES, player))
        self.assertFalse(_is_normal_move_in_progress(TurnPhase.ACTIONS, player))
        self.assertFalse(_is_normal_move_in_progress(TurnPhase.BONUS_MARKER_CHOICE, player))

    def test_pointless_movement_penalty_requires_an_unchanged_board(self):
        blue = object()
        green = object()
        first_post = mock.Mock(owner=blue, owner_piece_shape="square")
        second_post = mock.Mock(owner=blue, owner_piece_shape="square")

        self.assertEqual(
            pointless_movement_penalty(
                [(first_post, blue, "square")],
                [first_post],
            ),
            -1000,
        )
        self.assertEqual(
            pointless_movement_penalty(
                [(first_post, blue, "square"), (second_post, blue, "square")],
                [second_post, first_post],
            ),
            -1000,
        )
        first_post.owner_piece_shape = "circle"
        second_post.owner_piece_shape = "square"
        self.assertEqual(
            pointless_movement_penalty(
                [(first_post, blue, "square"), (second_post, blue, "circle")],
                [second_post, first_post],
            ),
            0,
        )
        first_post.owner = green
        first_post.owner_piece_shape = "square"
        second_post.owner = blue
        self.assertEqual(
            pointless_movement_penalty(
                [(first_post, blue, "square"), (second_post, green, "square")],
                [second_post, first_post],
            ),
            0,
        )
        third_post = mock.Mock(owner=blue, owner_piece_shape="square")
        self.assertEqual(
            pointless_movement_penalty(
                [(first_post, blue, "square")],
                [third_post],
            ),
            0,
        )

    def test_pointless_movement_penalty_treats_land_route_posts_as_equivalent(self):
        blue = object()
        land_route = mock.Mock(required_circles=0)
        other_route = mock.Mock(required_circles=0)
        maritime_route = mock.Mock(required_circles=1)
        land_origin = mock.Mock(owner=None, owner_piece_shape=None)
        land_destination = mock.Mock(owner=blue, owner_piece_shape="square")
        other_destination = mock.Mock(owner=blue, owner_piece_shape="square")
        maritime_origin = mock.Mock(owner=None, owner_piece_shape=None)
        maritime_destination = mock.Mock(owner=blue, owner_piece_shape="square")
        post_routes = {
            land_origin: land_route,
            land_destination: land_route,
            other_destination: other_route,
            maritime_origin: maritime_route,
            maritime_destination: maritime_route,
        }

        self.assertEqual(
            pointless_movement_penalty(
                [(land_origin, blue, "square")],
                [land_destination],
                post_routes,
            ),
            -1000,
        )
        self.assertEqual(
            pointless_movement_penalty(
                [(land_origin, blue, "square")],
                [other_destination],
                post_routes,
            ),
            0,
        )
        self.assertEqual(
            pointless_movement_penalty(
                [(maritime_origin, blue, "square")],
                [maritime_destination],
                post_routes,
            ),
            0,
        )

    def test_completed_move_rewards_only_net_new_claimable_routes(self):
        self.assertEqual(completed_route_move_reward({1}, {1, 2}), 50)
        self.assertEqual(completed_route_move_reward({1, 2}, {1}), -50)
        self.assertEqual(completed_route_move_reward({1}, {2}), 0)

    def test_route_focus_reward_has_a_cooldown_until_claim(self):
        rewarded, reward = move_route_focus_reward((), {4: 2})
        self.assertEqual(rewarded, frozenset({4}))
        self.assertEqual(reward, 10)

        rewarded, reward = move_route_focus_reward(rewarded, {4: 3})
        self.assertEqual(rewarded, frozenset({4}))
        self.assertEqual(reward, 0)

        rewarded, reward = move_route_focus_reward(rewarded - {4}, {4: 2})
        self.assertEqual(rewarded, frozenset({4}))
        self.assertEqual(reward, 10)

    def test_move_claim_combo_requires_the_next_paid_action_to_claim_that_route(self):
        pending, reward = update_move_claim_combo(
            {4},
            action=RouteInteraction(4, 1),
            turn_phase=TurnPhase.ACTIONS,
            action_was_spent=True,
        )
        self.assertEqual(pending, frozenset())
        self.assertEqual(reward, 250)

        pending, reward = update_move_claim_combo(
            {4},
            action=RouteInteraction(3, 1),
            turn_phase=TurnPhase.ACTIONS,
            action_was_spent=True,
        )
        self.assertEqual(pending, frozenset())
        self.assertEqual(reward, 0)

    def test_move_claim_combo_tracks_every_route_filled_by_the_move(self):
        pending, reward = update_move_claim_combo(
            (),
            action=PostInteraction(0, PieceShape.TRADER),
            turn_phase=TurnPhase.ACTIONS,
            action_was_spent=True,
            newly_completed_routes={2, 7},
        )
        self.assertEqual(pending, frozenset({2, 7}))
        self.assertEqual(reward, 0)

        unchanged, reward = update_move_claim_combo(
            pending,
            action=PostInteraction(1, PieceShape.TRADER),
            turn_phase=TurnPhase.MOVE_PIECES,
            action_was_spent=False,
        )
        self.assertEqual(unchanged, pending)
        self.assertEqual(reward, 0)

    def test_move_reward_values(self):
        self.assertEqual(MOVE_ROUTE_FOCUS_REWARD, 10)
        self.assertEqual(MOVE_CLAIM_COMBO_REWARD, 250)

    def test_route_claim_without_an_outcome_is_penalized(self):
        penalty = pointless_route_claim_penalty(
            action=RouteInteraction(4, 0),
            turn_phase=TurnPhase.ACTIONS,
            action_was_spent=True,
            gained_office=False,
            gained_upgrade=False,
            gained_marker=False,
            gained_points=False,
            route_had_permanent_marker=False,
        )
        self.assertEqual(penalty, POINTLESS_ROUTE_CLAIM_PENALTY)

    def test_any_useful_route_claim_outcome_avoids_the_penalty(self):
        outcomes = (
            "gained_office",
            "gained_upgrade",
            "gained_marker",
            "gained_points",
            "route_had_permanent_marker",
        )
        for outcome in outcomes:
            values = {name: name == outcome for name in outcomes}
            with self.subTest(outcome=outcome):
                self.assertEqual(
                    pointless_route_claim_penalty(
                        action=RouteInteraction(4, 0),
                        turn_phase=TurnPhase.ACTIONS,
                        action_was_spent=True,
                        **values,
                    ),
                    0,
                )

    def test_route_building_post_rewards(self):
        self.assertEqual(
            route_building_post_reward(route_already_has_piece=True, is_displacement=False),
            5,
        )
        self.assertEqual(
            route_building_post_reward(route_already_has_piece=True, is_displacement=True),
            3,
        )
        self.assertEqual(
            route_building_post_reward(route_already_has_piece=False, is_displacement=False),
            0,
        )

    def test_intermediate_non_keys_upgrades_receive_reward(self):
        before = ("WHITE", 2, 0, 3)
        after = ("ORANGE", 2, 0, 3)
        self.assertEqual(intermediate_ability_upgrade_reward(before, after), 250)

    def test_first_actions_upgrade_receives_larger_reward(self):
        before = ("WHITE", 2, 0, 3)
        after = ("WHITE", 2, 1, 3)
        self.assertEqual(intermediate_ability_upgrade_reward(before, after), 400)

    def test_final_upgrade_uses_projected_score_instead_of_extra_reward(self):
        before = ("WHITE", 2, 4, 3)
        after = ("WHITE", 2, 5, 3)
        self.assertEqual(intermediate_ability_upgrade_reward(before, after), 0)

    def test_repeated_actions_value_still_counts_as_an_intermediate_upgrade(self):
        before = ("WHITE", 2, 1, 3)
        after = ("WHITE", 2, 2, 3)
        self.assertEqual(intermediate_ability_upgrade_reward(before, after), 250)

    def test_move_penalty_is_applied_only_when_normal_move_finishes(self):
        original = (10.0, 20.0, 30.0)
        self.assertEqual(
            apply_movement_efficiency_penalty(
                original,
                acting_player_index=1,
                movement_capacity=4,
                pieces_moved=2,
                normal_move_completed=True,
            ),
            (10.0, -80.0, 30.0),
        )
        self.assertEqual(
            apply_movement_efficiency_penalty(
                original,
                acting_player_index=1,
                movement_capacity=4,
                pieces_moved=1,
                normal_move_completed=False,
            ),
            original,
        )

    def test_no_replacement_route_penalizes_only_responsible_player(self):
        game = load_game(STATE)
        responsible_player = game.current_player
        game.replace_bonus_marker = 1
        game.pending_bonus_markers = ["SwapOffice"]
        responsible_player.actions_remaining = 0
        responsible_player.ending_turn = True
        for route in game.selected_map.routes:
            if route.has_tradesmen():
                continue
            post = next(post for post in route.posts if not post.is_owned())
            post.claim(responsible_player, post.required_shape or "square")

        callback = mock.Mock()
        trainer = self.trainer()
        with mock.patch("training.self_play.load_game", return_value=game):
            trajectory = trainer.collect_game(STATE, failure_callback=callback)

        responsible_index = game.current_player_index
        expected_rewards = [0.0] * len(game.players)
        expected_rewards[responsible_index] = NO_REPLACEMENT_ROUTE_PENALTY
        self.assertEqual(trajectory.terminal_rewards, tuple(expected_rewards))
        self.assertEqual(trajectory.winner_indices, ())
        self.assertEqual(trajectory.completion_reason, "no_replacement_route")
        self.assertEqual(trainer.progress.completed_games, 0)
        callback.assert_called_once()

    def test_complete_trajectory_uses_acting_players_visible_observation(self):
        game = load_game(STATE)
        expected = ObservationEncoder().build(game)
        expected_mask = training_action_mask(
            game, disable_move_action=True, move_general_stock_threshold=0
        )
        trainer = self.trainer()
        before = tuple(parameter.detach().clone() for parameter in trainer.model.parameters())

        trajectory = trainer.collect_game(STATE)

        self.assertTrue(trajectory.decisions)
        decision = trajectory.decisions[0]
        self.assertEqual(decision.acting_player_index, expected.observer_index)
        self.assertTrue(torch.equal(decision.observation, expected.features))
        self.assertTrue(torch.equal(decision.legal_action_mask, expected_mask))
        self.assertEqual(decision.policy_tier, trajectory.seat_tiers[decision.acting_player_index])
        self.assertIn(decision.policy_tier, {1, 2, 3, 4, 5})
        self.assertTrue(
            all(
                torch.equal(old, current)
                for old, current in zip(before, trainer.model.parameters())
            )
        )
        expected_move_ratio = (
            trajectory.move_action_count / trajectory.spent_action_count
            if trajectory.spent_action_count
            else None
        )
        self.assertEqual(trajectory.move_ratio, expected_move_ratio)
        self.assertLessEqual(
            trajectory.move_claim_conversions,
            trajectory.moves_creating_claimable_route,
        )
        expected_conversion_rate = (
            trajectory.move_claim_conversions / trajectory.moves_creating_claimable_route
            if trajectory.moves_creating_claimable_route
            else None
        )
        self.assertEqual(
            trajectory.move_claim_conversion_rate,
            expected_conversion_rate,
        )

    def test_evaluation_keeps_tiers_but_disables_epsilon(self):
        trainer = self.trainer()
        selected_tiers = []
        select_action = trainer._select_action

        def capture_tier(scores, legal_indices, tier, equivalent_groups=None):
            selected_tiers.append(tier)
            return select_action(scores, legal_indices, tier, equivalent_groups)

        trainer._select_action = capture_tier
        trainer.collect_game(STATE, evaluation=True)

        self.assertTrue(selected_tiers)
        self.assertTrue(all(tier.epsilon == 0 for tier in selected_tiers))
        self.assertTrue(all(tier.top_k in (2, 5, 10, 15, 20) for tier in selected_tiers))

    def test_update_changes_shared_model_only_after_completed_game(self):
        trainer = self.trainer()
        trajectory = trainer.collect_game(STATE)
        before = tuple(parameter.detach().clone() for parameter in trainer.model.parameters())

        loss = trainer.update_model((trajectory,))

        self.assertGreater(loss, 0)
        self.assertTrue(
            any(
                not torch.equal(old, current)
                for old, current in zip(before, trainer.model.parameters())
            )
        )
        self.assertEqual(trainer.progress.completed_games, 1)
        self.assertEqual(trainer.progress.training_updates, 1)

    def test_long_game_receives_four_bounded_updates(self):
        trainer = self.trainer()
        trainer.config = replace(trainer.config, decision_batch_size=10)
        trajectory = trainer.collect_game(STATE)

        trainer.update_model((trajectory,))

        self.assertEqual(trainer.progress.training_updates, 4)

    def test_long_game_samples_are_bounded_and_non_overlapping(self):
        trainer = self.trainer()
        trainer.config = replace(trainer.config, decision_batch_size=4)
        decisions = [
            training_decision(
                index,
                0,
                (0,),
                0,
                movement_workflow_id=1 if index in (1, 2, 3) else None,
            )
            for index in range(8)
        ]
        mark_movement_workflow_target(decisions, 1, -1500)

        batches = trainer._training_batches(decisions)
        selected_actions = [decision.action_index for batch in batches for decision in batch]

        self.assertEqual(tuple(map(len, batches)), (4, 4))
        self.assertEqual(len(selected_actions), len(set(selected_actions)))
        self.assertTrue({1, 3, 7}.issubset(selected_actions))
        self.assertTrue(
            any(
                {1, 2, 3}.issubset(decision.action_index for decision in batch) for batch in batches
            )
        )

    def test_very_long_game_uses_four_bounded_updates(self):
        trainer = self.trainer()
        trainer.config = replace(trainer.config, decision_batch_size=4)
        decisions = tuple(training_decision(index, 0, (0,), 0) for index in range(20))

        batches = trainer._training_batches(decisions)

        self.assertEqual(len(batches), 4)
        self.assertEqual(sum(map(len, batches)), 16)
        self.assertTrue(all(len(batch) <= 4 for batch in batches))

    def test_epsilon_explores_all_legal_actions(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 5.0, 2.0])
        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.1
        trainer.rng.randrange.return_value = 1

        selection = trainer._select_action(scores, [1, 2], PolicyTier(3, 10, 0.2))

        self.assertEqual(selection.action_index, 2)
        self.assertTrue(selection.used_epsilon)
        self.assertEqual(selection.model_rank, 2)
        trainer.rng.randrange.assert_called_once_with(2)
        trainer.rng.choices.assert_not_called()

    def test_inverse_sqrt_rank_weights_and_normalization(self):
        raw = inverse_sqrt_rank_weights(5)
        normalized = normalized_rank_weights(5)

        self.assertEqual(raw, tuple(1.0 / math.sqrt(rank) for rank in range(1, 6)))
        self.assertAlmostEqual(sum(normalized), 1.0)
        self.assertAlmostEqual(normalized[0], 0.3094, places=4)
        self.assertAlmostEqual(normalized[4], 0.1384, places=4)

    def test_top_two_uses_normalized_rank_weighting(self):
        weights = normalized_rank_weights(2)

        self.assertAlmostEqual(weights[0], 0.5858, places=4)
        self.assertAlmostEqual(weights[1], 0.4142, places=4)

    def test_top_k_selects_with_inverse_sqrt_rank_weighting(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 5.0, 2.0, 7.0])
        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.9
        trainer.rng.choices.return_value = [1]

        selection = trainer._select_action(scores, [1, 2, 3], PolicyTier(1, 2, 0.05))

        self.assertEqual(selection.action_index, 1)
        self.assertFalse(selection.used_epsilon)
        self.assertEqual(selection.model_rank, 2)
        self.assertEqual(selection.legal_action_count, 3)
        trainer.rng.choices.assert_called_once_with(
            range(2),
            weights=normalized_rank_weights(2),
            k=1,
        )

    def test_top_k_with_tied_scores_stays_within_legal_pool(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 5.0, 5.0, 5.0])
        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.9
        trainer.rng.choices.return_value = [0]

        selection = trainer._select_action(scores, [1, 2, 3], PolicyTier(1, 2, 0.05))

        self.assertIn(selection.action_index, {1, 2, 3})
        trainer.rng.choices.assert_called_once_with(
            range(2),
            weights=normalized_rank_weights(2),
            k=1,
        )

    def test_effective_k_and_fully_random_tier(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 5.0, 2.0])
        trainer.rng = mock.Mock()
        trainer.rng.reset_mock()
        trainer.rng.random.return_value = 0.9
        trainer.rng.choices.return_value = [1]

        selection = trainer._select_action(scores, [1, 2], PolicyTier(3, 10, 0.2))
        self.assertEqual(selection.action_index, 2)
        trainer.rng.choices.assert_called_once_with(
            range(2),
            weights=normalized_rank_weights(2),
            k=1,
        )

        trainer.rng.reset_mock()
        trainer.rng.random.return_value = 0.2
        trainer.rng.randrange.return_value = 0
        selection = trainer._select_action(scores, [1, 2], PolicyTier(5, None, 1.0))
        self.assertEqual(selection.action_index, 1)
        self.assertTrue(selection.used_epsilon)
        trainer.rng.randrange.assert_called_once_with(2)
        trainer.rng.choices.assert_not_called()

    def test_single_legal_action_is_always_selected(self):
        trainer = self.trainer()
        scores = torch.tensor([10.0, -5.0])

        selection = trainer._select_action(scores, [1], PolicyTier(1, 2, 0.05))

        self.assertEqual(selection.action_index, 1)
        self.assertEqual(selection.model_rank, 1)
        self.assertEqual(selection.legal_action_count, 1)

    def test_action_limit_returns_neutral_incomplete_trajectory_for_learning(self):
        trainer = self.trainer()
        trainer.config = replace(trainer.config, max_actions=1)

        trajectory = trainer.collect_game(STATE)

        self.assertEqual(trajectory.completion_reason, "action_limit")
        self.assertEqual(trajectory.terminal_rewards, (0.0, 0.0, 0.0))
        self.assertEqual(len(trajectory.decisions), 1)
        self.assertEqual(trainer.progress.completed_games, 0)

    def test_workflow_selection_uses_weighted_top_three_and_random_exploration(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 40.0, 20.0, 30.0, 10.0])
        legal = [1, 2, 3, 4]

        for roll, expected_index, expected_rank in (
            (0.10, 1, 1),
            (0.50, 3, 2),
            (0.70, 2, 3),
        ):
            trainer.rng = mock.Mock()
            trainer.rng.random.return_value = roll
            selection = trainer._select_workflow_action(scores, legal)
            self.assertEqual(selection.action_index, expected_index)
            self.assertEqual(selection.model_rank, expected_rank)
            self.assertFalse(selection.used_epsilon)

        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.90
        trainer.rng.randrange.return_value = 3
        selection = trainer._select_workflow_action(scores, legal)
        self.assertEqual(selection.action_index, 4)
        self.assertEqual(selection.model_rank, 4)
        self.assertTrue(selection.used_epsilon)

    def test_workflow_selection_handles_forced_and_two_choice_steps(self):
        trainer = self.trainer()
        scores = torch.tensor([0.0, 10.0, 20.0])

        forced = trainer._select_workflow_action(scores, [2])
        self.assertEqual(forced.action_index, 2)
        self.assertFalse(forced.used_epsilon)

        trainer.rng = mock.Mock()
        trainer.rng.random.side_effect = (0.59, 0.60)
        best = trainer._select_workflow_action(scores, [1, 2])
        second = trainer._select_workflow_action(scores, [1, 2])
        self.assertEqual(best.action_index, 2)
        self.assertEqual(second.action_index, 1)

    def test_normal_move_exploration_separates_pickups_from_route_destinations(self):
        player = object()
        land_route = mock.Mock(
            required_circles=0,
            posts=[mock.Mock(owner=player), mock.Mock(owner=None), mock.Mock(owner=None)],
        )
        maritime_route = mock.Mock(
            required_circles=1,
            posts=[mock.Mock(owner=None) for _ in range(2)],
        )
        game = mock.Mock()
        game.current_player = player
        game.selected_map.routes = [land_route, maritime_route]
        trader_indices = [
            DEFAULT_ACTION_CODEC.encode(PostInteraction(slot, PieceShape.TRADER))
            for slot in range(5)
        ]
        merchant_indices = [
            DEFAULT_ACTION_CODEC.encode(PostInteraction(slot, PieceShape.MERCHANT))
            for slot in range(3)
        ]

        categories = move_workflow_exploration_categories(
            game,
            trader_indices[:3] + merchant_indices + trader_indices[3:],
        )

        self.assertEqual(
            categories,
            (
                ((trader_indices[0], merchant_indices[0]),),
                (
                    (trader_indices[1], trader_indices[2]),
                    (merchant_indices[1], merchant_indices[2]),
                    (trader_indices[3],),
                    (trader_indices[4],),
                ),
            ),
        )

    def test_action_phase_groups_equivalent_empty_land_posts(self):
        player = object()
        land_route = mock.Mock(
            required_circles=0,
            posts=[mock.Mock(owner=None), mock.Mock(owner=None), mock.Mock(owner=player)],
        )
        maritime_route = mock.Mock(
            required_circles=1,
            posts=[mock.Mock(owner=None), mock.Mock(owner=None)],
        )
        game = mock.Mock()
        game.selected_map.routes = [land_route, maritime_route]
        trader_indices = [
            DEFAULT_ACTION_CODEC.encode(PostInteraction(slot, PieceShape.TRADER))
            for slot in range(5)
        ]

        groups = action_phase_selection_groups(game, trader_indices)

        self.assertEqual(
            groups,
            (
                (trader_indices[0], trader_indices[1]),
                (trader_indices[2],),
                (trader_indices[3],),
                (trader_indices[4],),
            ),
        )

    def test_action_phase_groups_matching_displacements_and_move_pickups(self):
        current_player = object()
        opponent_one = object()
        opponent_two = object()
        route = mock.Mock(
            required_circles=0,
            posts=[
                mock.Mock(owner=opponent_one, owner_piece_shape="square"),
                mock.Mock(owner=opponent_one, owner_piece_shape="square"),
                mock.Mock(owner=opponent_two, owner_piece_shape="square"),
                mock.Mock(owner=current_player, owner_piece_shape="square"),
                mock.Mock(owner=current_player, owner_piece_shape="square"),
                mock.Mock(owner=current_player, owner_piece_shape="circle"),
            ],
        )
        game = mock.Mock()
        game.current_player = current_player
        game.selected_map.routes = [route]
        trader_indices = [
            DEFAULT_ACTION_CODEC.encode(PostInteraction(slot, PieceShape.TRADER))
            for slot in range(6)
        ]

        groups = action_phase_selection_groups(game, trader_indices)

        self.assertEqual(
            groups,
            (
                (trader_indices[0], trader_indices[1]),
                (trader_indices[2],),
                (trader_indices[3], trader_indices[4]),
                (trader_indices[5],),
            ),
        )

    def test_move_workflow_groups_same_shape_pickups_on_one_land_route(self):
        player = object()
        route = mock.Mock(
            required_circles=0,
            posts=[
                mock.Mock(owner=player, owner_piece_shape="square"),
                mock.Mock(owner=player, owner_piece_shape="square"),
                mock.Mock(owner=player, owner_piece_shape="circle"),
            ],
        )
        game = mock.Mock()
        game.current_player = player
        game.selected_map.routes = [route]
        indices = [
            DEFAULT_ACTION_CODEC.encode(PostInteraction(slot, PieceShape.TRADER))
            for slot in range(3)
        ]

        categories = move_workflow_exploration_categories(game, indices)

        self.assertEqual(categories, (((indices[0], indices[1]), (indices[2],)),))

    def test_move3_groups_matching_opponent_pieces_but_not_different_owners(self):
        current_player = object()
        opponent_one = object()
        opponent_two = object()
        route = mock.Mock(
            required_circles=0,
            posts=[
                mock.Mock(owner=opponent_one, owner_piece_shape="square"),
                mock.Mock(owner=opponent_one, owner_piece_shape="square"),
                mock.Mock(owner=opponent_one, owner_piece_shape="circle"),
                mock.Mock(owner=opponent_two, owner_piece_shape="square"),
            ],
        )
        game = mock.Mock()
        game.current_player = current_player
        game.selected_map.routes = [route]
        indices = [
            DEFAULT_ACTION_CODEC.encode(PostInteraction(slot, PieceShape.TRADER))
            for slot in range(4)
        ]

        categories = move_workflow_exploration_categories(
            game,
            indices,
            opponent_pickups=True,
        )

        self.assertEqual(
            categories,
            (((indices[0], indices[1]), (indices[2],), (indices[3],)),),
        )

    def test_ranked_action_selection_collapses_equivalent_placements(self):
        trainer = self.trainer()
        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.90
        trainer.rng.choices.return_value = [0]
        trainer.rng.randrange.return_value = 1
        scores = torch.tensor([10.0, 20.0, 30.0, 5.0])

        selection = trainer._select_action(
            scores,
            [0, 1, 2, 3],
            PolicyTier(1, 1, 0.05),
            ((0, 1, 2), (3,)),
        )

        self.assertEqual(selection.action_index, 1)
        self.assertEqual(selection.model_rank, 1)
        self.assertEqual(selection.legal_action_count, 2)
        self.assertEqual(selection.equivalent_action_indices, (0, 1, 2))
        trainer.rng.choices.assert_called_once_with(
            range(1),
            weights=(1.0,),
            k=1,
        )

    def test_workflow_exploration_randomizes_the_post_inside_a_chosen_route(self):
        trainer = self.trainer()
        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.90
        trainer.rng.randrange.side_effect = (1, 0, 1)
        scores = torch.tensor([10.0, 20.0, 30.0, 5.0, 1.0])

        selection = trainer._select_workflow_action(
            scores,
            [0, 1, 2, 3, 4],
            (((3,),), ((0, 1, 2), (4,))),
        )

        self.assertEqual(selection.action_index, 1)
        self.assertTrue(selection.used_epsilon)
        self.assertEqual(selection.model_rank, 1)
        self.assertEqual(selection.legal_action_count, 3)
        self.assertEqual(selection.equivalent_action_indices, (0, 1, 2))

    def test_ranked_workflow_choice_collapses_equivalent_posts(self):
        trainer = self.trainer()
        trainer.rng = mock.Mock()
        trainer.rng.random.return_value = 0.10
        trainer.rng.randrange.return_value = 1
        scores = torch.tensor([10.0, 20.0, 30.0, 5.0])

        selection = trainer._select_workflow_action(
            scores,
            [0, 1, 2, 3],
            (((3,),), ((0, 1, 2),)),
        )

        self.assertEqual(selection.action_index, 1)
        self.assertFalse(selection.used_epsilon)
        self.assertEqual(selection.model_rank, 1)
        self.assertEqual(selection.legal_action_count, 2)
        self.assertEqual(selection.equivalent_action_indices, (0, 1, 2))

    def test_default_tier_subsets_and_random_assignment(self):
        trainer = self.trainer(seed=99)

        self.assertEqual(set(tier.number for tier in trainer._assign_tiers(3)), {1, 3, 5})
        self.assertEqual(set(tier.number for tier in trainer._assign_tiers(4)), {1, 2, 4, 5})
        assignments = [tuple(tier.number for tier in trainer._assign_tiers(5)) for _ in range(8)]
        self.assertTrue(all(set(assignment) == {1, 2, 3, 4, 5} for assignment in assignments))
        self.assertGreater(len(set(assignments)), 1)

    def test_evaluation_tiers_rotate_through_seats_without_epsilon(self):
        trainer = self.trainer()

        assignments = [trainer._assign_evaluation_tiers(3, rotation) for rotation in range(3)]

        self.assertEqual(
            [tuple(tier.number for tier in assignment) for assignment in assignments],
            [(1, 3, 5), (3, 5, 1), (5, 1, 3)],
        )
        self.assertTrue(all(tier.epsilon == 0 for assignment in assignments for tier in assignment))

    def test_fixed_seed_reproduces_trajectory(self):
        first = self.trainer(seed=99).collect_game(STATE)
        second = self.trainer(seed=99).collect_game(STATE)

        self.assertEqual(first.action_trace, second.action_trace)
        self.assertEqual(first.seat_tiers, second.seat_tiers)
        self.assertEqual(first.terminal_rewards, second.terminal_rewards)
        self.assertEqual(
            [decision.reward_to_go for decision in first.decisions],
            [decision.reward_to_go for decision in second.decisions],
        )
        self.assertEqual(first.final_scores, second.final_scores)

    def test_discounted_reward_to_go_is_separate_for_each_player(self):
        decisions = (
            training_decision(1, 0, (100, 0), 100, turn=1),
            training_decision(2, 1, (0, 200), 200, turn=2),
            training_decision(3, 0, (-50, 0), -50, turn=3),
            training_decision(4, 1, (0, 0), 0, turn=4),
        )

        completed = assign_reward_to_go(decisions, (1000, 0), gamma=0.5)

        self.assertEqual([item.reward_to_go for item in completed], [575, 200, 950, 0])

    def test_decisions_in_same_player_turn_do_not_decay_each_other(self):
        decisions = (
            training_decision(1, 0, (100, 0), 100, turn=1),
            training_decision(2, 0, (200, 0), 200, turn=1),
            training_decision(3, 0, (300, 0), 300, turn=1),
            training_decision(4, 0, (400, 0), 400, turn=3),
        )

        completed = assign_reward_to_go(decisions, (1000, 0), gamma=0.5)

        self.assertEqual(
            [item.reward_to_go for item in completed],
            [1300, 1200, 1000, 1400],
        )

    def test_other_players_decisions_do_not_discount_winner_reward(self):
        decisions = (
            training_decision(1, 0, (0, 0, 0), 0, turn=1),
            training_decision(2, 1, (0, 0, 0), 0, turn=2),
            training_decision(3, 2, (0, 0, 0), 0, turn=3),
        )

        completed = assign_reward_to_go(decisions, (3000, 0, 0), gamma=0.5)

        self.assertEqual(completed[0].reward_to_go, 3000)

    def test_local_movement_target_replaces_terminal_credit_only_for_its_workflow(self):
        decisions = [
            training_decision(1, 0, (0,), 0, turn=1),
            training_decision(2, 0, (0,), 0, turn=1, movement_workflow_id=7),
            training_decision(3, 0, (0,), 0, turn=1, movement_workflow_id=7),
            training_decision(4, 0, (0,), 0, turn=1),
        ]
        mark_movement_workflow_target(decisions, 7, -1500)

        completed = assign_training_targets(decisions, (5000,), gamma=0.99)

        self.assertEqual(
            [decision.reward_to_go for decision in completed],
            [5000, -1500, -1500, 5000],
        )
        self.assertEqual([decision.immediate_reward for decision in completed], [0, 0, 0, 0])

    def test_normal_move_without_claim_keeps_rewards_but_not_terminal_credit(self):
        decisions = (
            training_decision(
                1,
                0,
                (10,),
                10,
                turn=1,
                movement_workflow_id=7,
                receives_terminal_credit=False,
            ),
            training_decision(2, 0, (100,), 100, turn=1),
        )

        completed = assign_training_targets(decisions, (5000,), gamma=0.99)

        self.assertEqual([decision.reward_to_go for decision in completed], [110, 5100])

    def test_immediate_claim_restores_terminal_credit_to_contributing_moves(self):
        decisions = [
            training_decision(
                1,
                0,
                (0,),
                0,
                turn=1,
                movement_workflow_id=7,
                receives_terminal_credit=False,
            ),
            training_decision(
                2,
                0,
                (0,),
                0,
                turn=1,
                movement_workflow_id=8,
                receives_terminal_credit=False,
            ),
        ]
        workflow_ids = credited_movement_workflows(
            ((7, frozenset({4})), (8, frozenset({4, 6}))),
            {4},
            4,
        )
        for workflow_id in workflow_ids:
            grant_movement_workflow_terminal_credit(decisions, workflow_id)

        completed = assign_training_targets(decisions, (5000,), gamma=0.99)

        self.assertEqual(workflow_ids, (7, 8))
        self.assertEqual([decision.reward_to_go for decision in completed], [5000, 5000])

    def test_claim_credits_only_moves_that_contributed_to_that_completed_route(self):
        workflow_ids = credited_movement_workflows(
            ((7, frozenset({4})), (8, frozenset({6}))),
            {4, 6},
            4,
        )

        self.assertEqual(workflow_ids, (7,))
        self.assertEqual(
            credited_movement_workflows(((7, frozenset({4})),), set(), 4),
            (),
        )

    def test_all_move_turn_targets_only_its_move_workflows(self):
        decisions = [
            training_decision(1, 0, (0,), 0, turn=1, movement_workflow_id=7),
            training_decision(2, 0, (0,), 0, turn=1, movement_workflow_id=7),
            training_decision(3, 0, (0,), 0, turn=1, movement_workflow_id=8),
            training_decision(4, 0, (0,), 0, turn=1),
        ]

        applied = apply_all_move_turn_target(decisions, (7, 8), spent_actions=2)
        completed = assign_training_targets(decisions, (5000,), gamma=0.99)

        self.assertTrue(applied)
        self.assertEqual(
            [decision.reward_to_go for decision in completed],
            [
                ALL_MOVE_TURN_LOCAL_TARGET,
                ALL_MOVE_TURN_LOCAL_TARGET,
                ALL_MOVE_TURN_LOCAL_TARGET,
                5000,
            ],
        )

    def test_mixed_action_turn_does_not_receive_all_move_target(self):
        decisions = [
            training_decision(1, 0, (0,), 0, movement_workflow_id=7),
            training_decision(2, 0, (0,), 0),
        ]

        applied = apply_all_move_turn_target(decisions, (7,), spent_actions=2)

        self.assertFalse(applied)
        self.assertTrue(all(decision.local_training_target is None for decision in decisions))

    def test_all_move_turn_does_not_weaken_a_stronger_movement_target(self):
        decisions = [
            training_decision(1, 0, (0,), 0, movement_workflow_id=7),
            training_decision(2, 0, (0,), 0, movement_workflow_id=8),
        ]
        mark_movement_workflow_target(decisions, 7, -1500)

        apply_all_move_turn_target(decisions, (7, 8), spent_actions=2)

        self.assertEqual(decisions[0].local_training_target, -1500)
        self.assertEqual(decisions[1].local_training_target, ALL_MOVE_TURN_LOCAL_TARGET)

    def test_terminal_reward_is_winner_only_and_trigger_bonus_requires_winner(self):
        game = mock.Mock()
        game.players = [mock.Mock(final_score=40), mock.Mock(final_score=35)]

        self.assertEqual(calculate_terminal_rewards(game, (0,), 1), (4000, 0))
        self.assertEqual(calculate_terminal_rewards(game, (0,), 0), (4150, 0))

        losing_decision = training_decision(1, 1, (0, 200), 200)
        completed = assign_reward_to_go((losing_decision,), (4000, 0), gamma=0.99)
        self.assertEqual(completed[0].reward_to_go, 200)

    def test_checkpoint_resumes_model_optimizer_progress_and_rng(self):
        trainer = self.trainer()
        trainer.train((STATE,), episodes=1, batch_size=1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            curriculum_state = {"stage_index": 2, "configuration_signature": "test"}
            trainer.save_checkpoint(checkpoint, (STATE,), curriculum_state=curriculum_state)
            restored = SelfPlayTrainer.from_checkpoint(checkpoint)

            self.assertEqual(restored.progress.completed_games, 1)
            self.assertEqual(restored.progress.training_updates, 1)
            self.assertEqual(restored.progress.checkpoint_saves, 1)
            self.assertEqual(restored.progress.checkpoint_loads, 1)
            self.assertEqual(restored.config.gamma, 0.99)
            self.assertEqual(restored.config.learning_rate, 0.0001)
            self.assertTrue(all(group["lr"] == 0.0001 for group in restored.optimizer.param_groups))
            self.assertEqual(restored.config.income_penalty_scale, 100)
            self.assertEqual(restored.config.tier_top_k, (2, 5, 10, 15, 20))
            self.assertEqual(restored.config.tier_epsilons, (0.05, 0.10, 0.20, 0.35, 0.35))
            self.assertEqual(restored.config.three_player_tiers, (1, 3, 5))
            self.assertEqual(restored.rng.getstate(), trainer.rng.getstate())
            self.assertEqual(restored.curriculum_state, curriculum_state)
            self.assertTrue(
                all(
                    torch.equal(first, second)
                    for first, second in zip(
                        trainer.model.parameters(), restored.model.parameters()
                    )
                )
            )
            restored.train((STATE,), episodes=1, batch_size=1)
            restored.save_checkpoint(checkpoint, (STATE,))
            self.assertEqual(
                SelfPlayTrainer.from_checkpoint(checkpoint).curriculum_state,
                curriculum_state,
            )

    def test_checkpoint_explicitly_migrates_observation_v1_weights_and_optimizer(self):
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            trainer.save_checkpoint(checkpoint, (STATE,))
            contents = torch.load(checkpoint, map_location="cpu")
            contents["observation_schema_version"] = 1
            contents["observation_schema_fingerprint"] = LEGACY_OBSERVATION_SCHEMA_V1_FINGERPRINT
            torch.save(contents, checkpoint)

            restored = SelfPlayTrainer.from_checkpoint(checkpoint)
            self.assertTrue(restored.model.migrated_observation_schema)
            self.assertEqual(
                restored.optimizer.state_dict()["param_groups"],
                trainer.optimizer.state_dict()["param_groups"],
            )

    def test_tier_metrics_are_recorded_by_tier_not_seat(self):
        trainer = self.trainer()
        trajectory = trainer.collect_game(STATE)
        tier = trajectory.decisions[0].policy_tier

        metrics = trainer.tier_metrics()[tier]

        self.assertEqual(metrics["games"], 1)
        self.assertEqual(metrics["wins"], 1)
        self.assertEqual(metrics["win_rate"], 1)
        tier_decisions = sum(decision.policy_tier == tier for decision in trajectory.decisions)
        self.assertGreaterEqual(metrics["average_selected_rank"], 1)
        self.assertEqual(
            metrics["epsilon_selections"] + metrics["top_k_selections"], tier_decisions
        )

    def test_only_selected_output_receives_direct_training_gradient(self):
        class IndependentOutputs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.zeros(ACTION_SPACE_SIZE))

            def forward(self, observations):
                return self.values.unsqueeze(0).expand(len(observations), -1)

        model = IndependentOutputs().to(device)
        trainer = SelfPlayTrainer(model=model, config=TrainingConfig())
        decision = replace(training_decision(7, 0, (100,), 100), reward_to_go=100.0)
        trajectory = mock.Mock(decisions=(decision,))

        trainer.update_model((trajectory,))

        self.assertNotEqual(float(model.values[7]), 0)
        unchanged = torch.cat((model.values[:7], model.values[8:]))
        self.assertTrue(torch.equal(unchanged, torch.zeros_like(unchanged)))

    def test_equivalent_route_outputs_share_the_training_target(self):
        class IndependentOutputs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.zeros(ACTION_SPACE_SIZE))

            def forward(self, observations):
                return self.values.unsqueeze(0).expand(len(observations), -1)

        model = IndependentOutputs().to(device)
        trainer = SelfPlayTrainer(model=model, config=TrainingConfig())
        with torch.no_grad():
            model.values[7:10] = torch.tensor([50.0, 100.0, 150.0], device=device)
        decision = replace(
            training_decision(
                7,
                0,
                (100,),
                100,
                equivalent_action_indices=(7, 8, 9),
            ),
            reward_to_go=100.0,
        )
        trajectory = mock.Mock(decisions=(decision,))

        trainer.update_model((trajectory,))

        self.assertGreater(float(model.values[7]), 50.0)
        self.assertEqual(float(model.values[8]), 100.0)
        self.assertLess(float(model.values[9]), 150.0)
        unchanged = torch.cat((model.values[:7], model.values[10:]))
        self.assertTrue(torch.equal(unchanged, torch.zeros_like(unchanged)))

    def test_training_clips_each_minibatch_gradient(self):
        trainer = self.trainer()
        trajectory = trainer.collect_game(STATE)

        with mock.patch("torch.nn.utils.clip_grad_norm_") as clip:
            trainer.update_model((trajectory,))

        self.assertEqual(clip.call_count, trainer.progress.training_updates)
        self.assertTrue(
            all(call.args[1] == trainer.config.max_gradient_norm for call in clip.call_args_list)
        )

    def test_checkpoint_rejects_incompatible_action_schema(self):
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            trainer.save_checkpoint(checkpoint, (STATE,))
            contents = torch.load(checkpoint, map_location="cpu")
            contents["action_schema_version"] = ACTION_SCHEMA_VERSION + 1
            torch.save(contents, checkpoint)

            with self.assertRaisesRegex(ValueError, "action schema"):
                SelfPlayTrainer.from_checkpoint(checkpoint)

    def test_training_checkpoint_is_usable_for_inference(self):
        trainer = self.trainer()
        trainer.train((STATE,), episodes=1, batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            trainer.save_checkpoint(checkpoint, (STATE,))

            inference_model = HansaNN(model_file=checkpoint)

        self.assertFalse(inference_model.training)


if __name__ == "__main__":
    unittest.main()
