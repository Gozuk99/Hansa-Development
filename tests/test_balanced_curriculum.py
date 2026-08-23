import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from game.invariants import validate_game
from game.loaded_state_validation import validate_loaded_game
from training.balanced_curriculum import (
    CONFIGURATIONS,
    EARLY_ROUTE_SCAFFOLD_RATE,
    MATURITY_CYCLE,
    MATURITY_PROFILES,
    BalancedCurriculumRunner,
    _scenario_condition_label,
    _select_focus,
    _select_optional_modules,
    _uses_early_route_scaffold,
)
from training.balanced_state_generator import (
    BalancedGenerationRequest,
    BonusMarkerSetup,
    EndingCondition,
    RegionalFocus,
    StartingPosition,
    StrategicFocus,
    _scaffold_early_routes,
    generate_balanced_state,
    player_has_starting_score_source,
    starting_scores_have_valid_sources,
)
from map_data.map_attributes import Map
from training.curriculum import CurriculumConfig
from training.targeted_state_generator import _fill_prepared_route


class BalancedCurriculumTests(unittest.TestCase):
    @staticmethod
    def _runner(training_generation_number=0):
        runner = object.__new__(BalancedCurriculumRunner)
        runner.config = SimpleNamespace(seed=124)
        runner.training_generation_number = training_generation_number
        return runner

    def test_each_nine_game_block_contains_every_map_player_combination(self):
        runner = self._runner()

        for block in range(3):
            observed = set()
            for offset in range(9):
                runner.training_generation_number = block * 9 + offset
                observed.add(runner._configuration_for_game())
            self.assertEqual(observed, set(CONFIGURATIONS))

    def test_three_player_regional_options_exclude_scotland_and_isle_of_man(self):
        request = BalancedGenerationRequest(
            seed=1,
            map_num=3,
            player_count=3,
            ending_condition=EndingCondition.NEAR_SCORE,
            regional_focus=RegionalFocus.SCOTLAND,
        )
        with self.assertRaisesRegex(ValueError, "Three-player"):
            generate_balanced_state(request, max_attempts=1)

    def test_near_score_states_start_at_sixteen_or_seventeen(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=321,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_SCORE,
            )
        )

        self.assertTrue(all(player.score in (16, 17) for player in generated.game.players))
        self.assertIn(17, [player.score for player in generated.game.players])
        projected_scores = generated.game.projected_scores()
        self.assertLessEqual(max(projected_scores) - min(projected_scores), 3)

    def test_map_two_five_player_ordinary_near_score_generates_efficiently(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=1_109_906_515,
                map_num=2,
                player_count=5,
                ending_condition=EndingCondition.NEAR_SCORE,
                score_range=(6, 11),
                development_range=(5, 7),
                strategic_focus=StrategicFocus.NONE,
                prepared_routes_one_short=True,
                bonus_markers_remaining=3,
                completed_cities_below_limit=5,
                round_range=(6, 10),
            ),
            max_attempts=100,
        )

        self.assertEqual(generated.attempt_seed, 1_109_906_515)
        self.assertTrue(all(6 <= player.score <= 11 for player in generated.game.players))
        projected_scores = generated.game.projected_scores()
        self.assertLessEqual(max(projected_scores) - min(projected_scores), 3)

    def test_training_bonus_marker_state_keeps_one_marker_in_supply(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=322,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_BONUS_MARKERS,
                bonus_markers_remaining=1,
            )
        )

        self.assertEqual(len(generated.game.selected_map.bonus_marker_pool), 1)
        self.assertEqual(
            sum(route.bonus_marker is not None for route in generated.game.selected_map.routes),
            3,
        )
        self.assertEqual(generated.game.replace_bonus_marker, 0)
        self.assertEqual(generated.game.pending_bonus_markers, [])

    def test_prepared_bonus_marker_route_is_one_post_short(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=324,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_BONUS_MARKERS,
                bonus_markers_remaining=1,
                prepared_routes_one_short=True,
            )
        )

        self.assertTrue(
            any(
                route.bonus_marker is not None
                and sum(post.owner is None for post in route.posts) == 1
                for route in generated.game.selected_map.routes
            )
        )

    def test_prepared_route_randomizes_open_post_and_unrestricted_piece_shapes(self):
        open_indices = set()
        observed_shapes = set()
        for seed in range(20):
            posts = []
            for _index in range(4):
                post = mock.Mock(required_shape=None, owner=None, owner_piece_shape=None)
                post.is_owned.return_value = False
                posts.append(post)
            route = SimpleNamespace(posts=posts)
            prepared, _missing = _fill_prepared_route(
                route,
                object(),
                {"square": 10, "circle": 10},
                None,
                True,
                random.Random(seed),
            )
            self.assertTrue(prepared)
            open_indices.add(next(index for index, post in enumerate(posts) if post.owner is None))
            observed_shapes.update(
                post.owner_piece_shape for post in posts if post.owner_piece_shape is not None
            )

        self.assertGreater(len(open_indices), 1)
        self.assertEqual(observed_shapes, {"square", "circle"})

    def test_partial_prepared_routes_leave_exactly_one_post_open(self):
        player = object()
        for route_size in (2, 3, 4):
            with self.subTest(route_size=route_size):
                posts = []
                for _index in range(route_size):
                    post = mock.Mock(required_shape=None, owner=None, owner_piece_shape=None)
                    post.is_owned.return_value = False
                    posts.append(post)
                prepared, missing_shape = _fill_prepared_route(
                    SimpleNamespace(posts=posts),
                    player,
                    {"square": 10, "circle": 10},
                    None,
                    True,
                    random.Random(route_size),
                )

                self.assertTrue(prepared)
                self.assertIn(missing_shape, ("square", "circle"))
                self.assertEqual(sum(post.owner is player for post in posts), route_size - 1)
                self.assertEqual(sum(post.owner is None for post in posts), 1)

    def test_automatic_focus_selection_never_adds_a_blocker(self):
        blocked = {
            StrategicFocus.BLOCKED_EAST_WEST,
            StrategicFocus.BLOCKED_DUAL_EAST_WEST,
        }
        for seed in range(1_000):
            focus, _regional = _select_focus(random.Random(seed), 3, 5, EndingCondition.NEAR_SCORE)
            self.assertNotIn(focus, blocked)

    def test_training_completed_city_state_stays_two_cities_below_limit(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=323,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_COMPLETED_CITIES,
                completed_cities_below_limit=2,
            )
        )

        self.assertEqual(
            generated.game.current_full_cities_count,
            generated.game.selected_map.max_full_cities - 2,
        )

    def test_active_maturity_cycle_has_requested_training_distribution(self):
        runner = self._runner()
        observed = Counter(
            runner._maturity_for_game(game_number).name
            for game_number in range(len(MATURITY_CYCLE))
        )

        self.assertEqual(
            observed,
            {
                "mixed": 64,
                "early_mixed": 48,
                "early": 24,
                "mid": 9,
                "late": 9,
                "end": 6,
            },
        )
        total = sum(profile.weight for profile in MATURITY_PROFILES)
        self.assertEqual(total, 160)
        self.assertEqual(observed["mixed"] / total, 0.40)
        self.assertEqual(observed["early_mixed"] / total, 0.30)
        self.assertEqual(observed["early"] / total, 0.15)
        self.assertEqual((observed["mid"] + observed["late"] + observed["end"]) / total, 0.15)
        self.assertEqual(
            (observed["mid"], observed["late"], observed["end"]),
            (9, 9, 6),
        )
        self.assertEqual(
            runner._stage_label(None),
            "early_early_mixed_mid_late_end_mixed_game",
        )

    def test_early_route_scaffold_split_is_reproducible_and_near_seventy_percent(self):
        first = [_uses_early_route_scaffold(seed) for seed in range(10_000)]
        second = [_uses_early_route_scaffold(seed) for seed in range(10_000)]

        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first) / len(first), EARLY_ROUTE_SCAFFOLD_RATE, delta=0.02)

    def test_scaffolded_early_states_fill_two_unique_legal_routes_per_player(self):
        for map_num, players in ((1, 3), (2, 4), (3, 5)):
            with self.subTest(map_num=map_num, players=players):
                generated = generate_balanced_state(
                    BalancedGenerationRequest(
                        seed=9_000 + map_num * 10 + players,
                        map_num=map_num,
                        player_count=players,
                        ending_condition=EndingCondition.NEAR_SCORE,
                        score_range=(0, 5),
                        bonus_markers_remaining=9,
                        completed_cities_below_limit=7,
                        development_range=(2, 4),
                        prepare_ending_condition=False,
                        round_range=(2, 5),
                        early_route_scaffold=True,
                    )
                )
                game = generated.game
                route_ids = generated.scaffolded_route_ids_by_seat
                route_lengths = generated.scaffolded_route_lengths_by_seat
                flattened = [route_id for player_routes in route_ids for route_id in player_routes]

                self.assertTrue(generated.early_route_scaffold)
                self.assertEqual(len(flattened), players * 2)
                self.assertEqual(len(set(flattened)), players * 2)
                for seat, player in enumerate(game.players):
                    self.assertEqual(len(route_ids[seat]), 2)
                    self.assertEqual(len(route_lengths[seat]), 2)
                    self.assertEqual(
                        sum(route.is_controlled_by(player) for route in game.selected_map.routes),
                        2,
                    )
                    for route_id, route_length in zip(route_ids[seat], route_lengths[seat]):
                        route = game.selected_map.routes[route_id]
                        self.assertEqual(len(route.posts), route_length)
                        self.assertTrue(route.is_controlled_by(player))
                        self.assertTrue(
                            all(
                                post.required_shape in (None, post.owner_piece_shape)
                                for post in route.posts
                            )
                        )
                validate_game(game)
                self.assertTrue(validate_loaded_game(game))

    def test_unscaffolded_early_state_keeps_routes_sparse(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=9_100,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_SCORE,
                score_range=(0, 5),
                bonus_markers_remaining=9,
                completed_cities_below_limit=7,
                development_range=(2, 4),
                prepare_ending_condition=False,
                round_range=(2, 5),
                early_route_scaffold=False,
            )
        )

        self.assertFalse(generated.early_route_scaffold)
        self.assertEqual(generated.scaffolded_route_ids_by_seat, ())
        self.assertFalse(
            any(
                route.is_controlled_by(player)
                for player in generated.game.players
                for route in generated.game.selected_map.routes
            )
        )

    def test_scaffold_route_sampling_considers_all_empty_routes_without_quality_filtering(self):
        players = [object() for _index in range(3)]
        routes = []
        for index in range(12):
            posts = []
            for _post_index in range(2 + index % 3):
                post = mock.Mock(required_shape=None, owner=None, owner_piece_shape=None)
                post.is_owned.return_value = False
                posts.append(post)
            route = mock.Mock(
                posts=posts,
                bonus_marker=(object() if index % 2 else None),
                has_bonus_marker=bool(index % 2),
            )
            routes.append(route)
        game = SimpleNamespace(
            players=players,
            selected_map=SimpleNamespace(routes=routes),
        )
        pools = {player: {"square": 26, "circle": 4} for player in players}
        rng = mock.Mock(wraps=random.Random(99))

        result = _scaffold_early_routes(game, pools, rng)

        self.assertIsNotNone(result)
        sampled_population, sampled_count = rng.sample.call_args_list[0].args
        self.assertEqual(sampled_population, routes)
        self.assertEqual(sampled_count, 6)

    def test_mixed_development_uses_shuffled_legal_role_targets(self):
        role_targets = {
            3: {"low": (2, 3), "medium": (6, 5), "high": (10, 7)},
            4: {"low": (2, 3), "low_mid": (5, 4), "high_mid": (7, 6), "high": (10, 7)},
            5: {
                "very_low": (2, 3),
                "low": (4, 4),
                "medium": (6, 5),
                "high": (8, 6),
                "very_high": (10, 7),
            },
        }
        observed_orders = set()
        for players in (3, 4, 5):
            generated = generate_balanced_state(
                BalancedGenerationRequest(
                    seed=4_000 + players,
                    map_num=1,
                    player_count=players,
                    ending_condition=EndingCondition.NEAR_SCORE,
                    score_range=(0, 12),
                    development_range=(2, 8),
                    prepare_ending_condition=False,
                    mixed_development=True,
                    round_range=(6, 12),
                )
            )
            roles = generated.development_roles_by_seat
            scores = generated.starting_scores_by_seat
            development = generated.starting_development_by_seat
            observed_orders.add(roles)
            self.assertEqual(set(roles), set(role_targets[players]))
            self.assertLessEqual(max(scores) - min(scores), 8)
            self.assertLessEqual(max(development) - min(development), 4)
            for role, score, developed in zip(roles, scores, development):
                target_score, target_development = role_targets[players][role]
                self.assertLessEqual(abs(score - target_score), 1)
                self.assertLessEqual(abs(developed - target_development), 1)
            self.assertTrue(starting_scores_have_valid_sources(generated.game))
        self.assertGreater(len(observed_orders), 1)

    def test_early_mixed_uses_modest_shuffled_development_and_partial_routes(self):
        observed_high_seats = set()
        for map_num, players in ((1, 3), (2, 4), (3, 5)):
            with self.subTest(map_num=map_num, players=players):
                generated = generate_balanced_state(
                    BalancedGenerationRequest(
                        seed=6_000 + map_num * 10 + players,
                        map_num=map_num,
                        player_count=players,
                        ending_condition=EndingCondition.NEAR_SCORE,
                        score_range=(2, 6),
                        development_range=(3, 5),
                        bonus_markers_remaining=7,
                        completed_cities_below_limit=6,
                        prepare_ending_condition=False,
                        round_range=(3, 6),
                        early_mixed_development=True,
                    )
                )
                game = generated.game
                scores = generated.starting_scores_by_seat
                development = generated.starting_development_by_seat
                roles = generated.development_roles_by_seat
                observed_high_seats.add(roles.index("high"))

                self.assertTrue(all(2 <= score <= 6 for score in scores))
                self.assertTrue(all(3 <= value <= 5 for value in development))
                self.assertLessEqual(max(scores) - min(scores), 4)
                self.assertLessEqual(max(development) - min(development), 2)
                self.assertTrue(starting_scores_have_valid_sources(game))
                self.assertFalse(generated.early_route_scaffold)
                self.assertEqual(generated.scaffolded_route_ids_by_seat, ())
                self.assertTrue(
                    all(
                        any(not post.is_owned() for post in route.posts)
                        for route in game.selected_map.routes
                    )
                )
                for player in game.players:
                    occupied_routes = [
                        route
                        for route in game.selected_map.routes
                        if any(post.owner is player for post in route.posts)
                    ]
                    self.assertGreaterEqual(len(occupied_routes), 2)
                    self.assertTrue(
                        any(
                            sum(post.owner is player for post in route.posts) >= 2
                            for route in occupied_routes
                        )
                    )
                validate_game(game)
                self.assertTrue(validate_loaded_game(game))

        self.assertGreater(len(observed_high_seats), 1)

    def test_early_mixed_is_weaker_than_existing_mixed_development(self):
        early_mixed = generate_balanced_state(
            BalancedGenerationRequest(
                seed=6_100,
                map_num=1,
                player_count=5,
                ending_condition=EndingCondition.NEAR_SCORE,
                score_range=(2, 6),
                development_range=(3, 5),
                bonus_markers_remaining=7,
                completed_cities_below_limit=6,
                prepare_ending_condition=False,
                round_range=(3, 6),
                early_mixed_development=True,
            )
        )
        mixed = generate_balanced_state(
            BalancedGenerationRequest(
                seed=6_100,
                map_num=1,
                player_count=5,
                ending_condition=EndingCondition.NEAR_SCORE,
                score_range=(0, 12),
                development_range=(2, 8),
                bonus_markers_remaining=5,
                completed_cities_below_limit=5,
                prepare_ending_condition=False,
                round_range=(6, 12),
                mixed_development=True,
            )
        )

        self.assertLess(
            sum(early_mixed.starting_scores_by_seat),
            sum(mixed.starting_scores_by_seat),
        )
        self.assertLess(
            sum(early_mixed.starting_development_by_seat),
            sum(mixed.starting_development_by_seat),
        )

    def test_positive_starting_score_requires_city_control_or_spent_prestige_circle(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=4_100,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_SCORE,
                score_range=(0, 0),
                development_range=(2, 4),
                prepare_ending_condition=False,
                round_range=(2, 5),
            )
        )
        game = generated.game
        player = game.players[0]
        player.score = 3
        for city in game.selected_map.cities:
            for office in city.offices:
                if office.controller is player:
                    office.controller = None
        for circle in game.selected_map.specialprestigepoints.circle_data:
            if circle["owner"] is player:
                circle["owner"] = None
        player.bonus_markers.append("SwapOffice")
        player.actions_index = min(player.actions_index + 1, 5)
        self.assertFalse(player_has_starting_score_source(game, player))
        self.assertFalse(starting_scores_have_valid_sources(game))

        player.score = 0
        self.assertTrue(starting_scores_have_valid_sources(game))
        player.score = 3
        city = next(
            city
            for city in game.selected_map.cities
            if city.offices and all(office.controller is None for office in city.offices)
        )
        city.offices[0].controller = player
        self.assertTrue(player_has_starting_score_source(game, player))
        city.offices[0].controller = None

        game.selected_map.specialprestigepoints.circle_data[0]["owner"] = player
        self.assertTrue(player_has_starting_score_source(game, player))
        self.assertTrue(starting_scores_have_valid_sources(game))

    def test_early_and_early_mixed_training_receive_the_extended_action_limit(self):
        runner = self._runner()
        stage = SimpleNamespace(name="mixed", action_limit=10_000)

        self.assertEqual(
            runner._training_action_limit(stage, SimpleNamespace(scenario="early")),
            15_000,
        )
        self.assertEqual(
            runner._training_action_limit(stage, SimpleNamespace(scenario="early_mixed")),
            15_000,
        )
        for maturity in ("mixed", "mid", "late", "end"):
            self.assertEqual(
                runner._training_action_limit(stage, SimpleNamespace(scenario=maturity)),
                10_000,
            )

    def test_early_evaluation_keeps_the_historical_action_limit(self):
        runner = self._runner()
        stage = SimpleNamespace(name="mixed", action_limit=10_000)

        self.assertEqual(
            runner._evaluation_action_limit(stage, SimpleNamespace(scenario="early")),
            10_000,
        )

    def test_optional_training_modules_follow_requested_probabilities(self):
        outcomes = Counter()
        mission_games = 0
        sample_size = 10_000
        for seed in range(sample_size):
            missions, _favour, marker_setup = _select_optional_modules(random.Random(seed), 1)
            mission_games += int(missions)
            outcomes[marker_setup] += 1

        self.assertAlmostEqual(mission_games / sample_size, 0.40, delta=0.02)
        self.assertAlmostEqual(outcomes[BonusMarkerSetup.DEFAULT] / sample_size, 0.50, delta=0.02)
        self.assertAlmostEqual(
            outcomes[BonusMarkerSetup.ALL_PROMOS] / sample_size, 0.25, delta=0.02
        )
        self.assertAlmostEqual(outcomes[BonusMarkerSetup.MIXED] / sample_size, 0.25, delta=0.02)
        self.assertTrue(
            all(not _select_optional_modules(random.Random(seed), 2)[0] for seed in range(100))
        )

    def test_all_promo_setup_contains_every_promo_and_shuffled_defaults(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=951,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_SCORE,
                bonus_marker_setup=BonusMarkerSetup.ALL_PROMOS,
            )
        )

        supply = generated.game.configuration.promo_markers
        counts = Counter(supply)
        self.assertEqual(len(supply), 12)
        for marker, count in Map.PROMO_BONUS_MARKERS.items():
            self.assertEqual(counts[marker], count)
        self.assertEqual(
            sum(counts[marker] for marker in Map.STANDARD_BONUS_MARKER_SUPPLY),
            6,
        )

    def test_mixed_promo_setup_contains_both_marker_families(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=952,
                map_num=2,
                player_count=4,
                ending_condition=EndingCondition.NEAR_SCORE,
                bonus_marker_setup=BonusMarkerSetup.MIXED,
            )
        )

        supply = generated.game.configuration.promo_markers
        counts = Counter(supply)
        promo_count = sum(counts[marker] for marker in Map.PROMO_BONUS_MARKERS)
        self.assertEqual(len(supply), 12)
        self.assertGreater(promo_count, 0)
        self.assertLess(promo_count, sum(Map.PROMO_BONUS_MARKERS.values()))

    def test_early_scenario_has_no_end_condition_label(self):
        early = next(profile for profile in MATURITY_PROFILES if profile.name == "early")
        early_mixed = next(
            profile for profile in MATURITY_PROFILES if profile.name == "early_mixed"
        )
        mixed = next(profile for profile in MATURITY_PROFILES if profile.name == "mixed")
        mid = next(profile for profile in MATURITY_PROFILES if profile.name == "mid")
        late = next(profile for profile in MATURITY_PROFILES if profile.name == "late")

        self.assertTrue(
            all(
                _scenario_condition_label(early, condition) is None for condition in EndingCondition
            )
        )
        self.assertTrue(
            all(
                _scenario_condition_label(early_mixed, condition) is None
                for condition in EndingCondition
            )
        )
        self.assertTrue(
            all(
                _scenario_condition_label(mixed, condition) is None for condition in EndingCondition
            )
        )
        self.assertEqual(
            [_scenario_condition_label(mid, condition) for condition in EndingCondition],
            ["score_focus", "bonus_marker_focus", "completed_city_focus"],
        )
        self.assertEqual(
            _scenario_condition_label(late, EndingCondition.NEAR_BONUS_MARKERS),
            "near_bonus_markers",
        )

    def test_completed_city_focus_does_not_select_dual_east_west(self):
        for seed in range(100):
            focus, _regional = _select_focus(
                random.Random(seed), 3, 5, EndingCondition.NEAR_COMPLETED_CITIES
            )
            self.assertNotIn(
                focus,
                (
                    StrategicFocus.DUAL_EAST_WEST,
                    StrategicFocus.BLOCKED_DUAL_EAST_WEST,
                ),
            )

    def test_bonus_marker_focus_avoids_incompatible_combinations(self):
        for seed in range(100):
            focus, regional = _select_focus(
                random.Random(seed), 3, 5, EndingCondition.NEAR_BONUS_MARKERS
            )
            self.assertIsNot(focus, StrategicFocus.BLOCKED_DUAL_EAST_WEST)
            self.assertIsNot(regional, RegionalFocus.ISLE_OF_MAN)

    def test_regional_dual_east_west_focus_drops_only_the_blocker(self):
        saw_dual_regional = False
        for seed in range(1_000):
            focus, regional = _select_focus(random.Random(seed), 3, 5, EndingCondition.NEAR_SCORE)
            if regional is not None:
                self.assertIsNot(focus, StrategicFocus.BLOCKED_DUAL_EAST_WEST)
                saw_dual_regional |= focus is StrategicFocus.DUAL_EAST_WEST

        self.assertTrue(saw_dual_regional)

    def test_three_player_focus_does_not_combine_dual_east_west_with_blocker(self):
        saw_dual = False
        for seed in range(1_000):
            focus, _regional = _select_focus(random.Random(seed), 2, 3, EndingCondition.NEAR_SCORE)
            self.assertIsNot(focus, StrategicFocus.BLOCKED_DUAL_EAST_WEST)
            saw_dual |= focus is StrategicFocus.DUAL_EAST_WEST

        self.assertTrue(saw_dual)

    def test_maturity_profiles_progress_from_fresh_to_end_game(self):
        profiles = {profile.name: profile for profile in MATURITY_PROFILES}

        self.assertEqual(set(profiles), {"early", "early_mixed", "mid", "late", "end", "mixed"})
        self.assertEqual(profiles["early"].score_range, (0, 5))
        self.assertEqual(profiles["early"].development_range, (2, 4))
        self.assertEqual(profiles["early"].bonus_markers_remaining, 9)
        self.assertEqual(profiles["early"].round_range, (2, 5))
        self.assertEqual(profiles["early"].starting_position_label, "")
        self.assertEqual(profiles["mid"].score_range, (6, 11))
        self.assertEqual(profiles["mid"].round_range, (6, 10))
        self.assertEqual(profiles["mid"].starting_position_label, "")
        self.assertEqual(profiles["late"].bonus_markers_remaining, 2)
        self.assertEqual(profiles["late"].round_range, (11, 15))
        self.assertEqual(profiles["late"].starting_position_label, "")
        self.assertEqual(profiles["end"].score_range, (16, 17))
        self.assertEqual(profiles["end"].completed_cities_below_limit, 2)
        self.assertEqual(profiles["end"].round_range, (16, 20))
        self.assertIs(
            profiles["end"].starting_position,
            StartingPosition.TWO_DECISIONS_BEFORE,
        )
        self.assertEqual(
            profiles["end"].starting_position_label,
            "two_decisions_before",
        )
        self.assertEqual(profiles["early_mixed"].score_range, (2, 6))
        self.assertEqual(profiles["early_mixed"].development_range, (3, 5))
        self.assertEqual(profiles["early_mixed"].bonus_markers_remaining, 7)
        self.assertEqual(profiles["early_mixed"].round_range, (3, 6))
        self.assertEqual(profiles["early_mixed"].starting_position_label, "")
        self.assertEqual(profiles["mixed"].score_range, (0, 12))
        self.assertEqual(profiles["mixed"].development_range, (2, 8))
        self.assertEqual(profiles["mixed"].starting_position_label, "")

    def test_early_state_applies_early_scores_and_development(self):
        with mock.patch(
            "training.balanced_state_generator._prepare_ending_condition",
            side_effect=AssertionError("early generation prepared an ending"),
        ):
            generated = generate_balanced_state(
                BalancedGenerationRequest(
                    seed=901,
                    map_num=1,
                    player_count=3,
                    ending_condition=EndingCondition.NEAR_BONUS_MARKERS,
                    score_range=(0, 5),
                    development_range=(2, 4),
                    bonus_markers_remaining=9,
                    completed_cities_below_limit=7,
                    prepared_routes_one_short=True,
                    prepare_ending_condition=False,
                    round_range=(2, 5),
                )
            )

        self.assertTrue(all(0 <= player.score <= 5 for player in generated.game.players))
        self.assertEqual(len(generated.game.selected_map.bonus_marker_pool), 9)
        self.assertEqual(
            sum(route.bonus_marker is not None for route in generated.game.selected_map.routes),
            3,
        )
        minimum, maximum = generated.development_range
        self.assertEqual((minimum, maximum), (2, 4))
        self.assertEqual(generated.game.current_full_cities_count, 0)
        self.assertTrue(validate_game(generated.game))
        self.assertTrue(validate_loaded_game(generated.game))
        self.assertTrue(2 <= generated.game.round_number <= 5)
        self.assertEqual(
            generated.game.turn_number,
            (generated.game.round_number - 1) * 3 + generated.game.current_player_index + 1,
        )

    def test_curriculum_early_request_does_not_prepare_an_ending(self):
        early_number = next(
            game_number
            for game_number in range(len(MATURITY_CYCLE))
            if self._runner()._maturity_for_game(game_number).name == "early"
        )
        runner = self._runner(early_number)
        runner._latest_descriptor = None
        generated = SimpleNamespace()

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.hansa"
            metadata_path = Path(directory) / "state.json"
            with (
                mock.patch(
                    "training.balanced_curriculum.generate_balanced_state",
                    return_value=generated,
                ) as generate,
                mock.patch(
                    "training.balanced_curriculum.save_balanced_state",
                    return_value=(state_path, metadata_path),
                ) as save,
            ):
                descriptor = runner._generate_state(
                    SimpleNamespace(full_game=False),
                    123,
                    Path(directory),
                )

        request = generate.call_args.args[0]
        self.assertFalse(request.prepare_ending_condition)
        self.assertFalse(request.prepared_routes_one_short)
        self.assertEqual(request.round_range, (2, 5))
        self.assertEqual(request.strategic_focus, StrategicFocus.NONE)
        self.assertIsNone(request.regional_focus)
        self.assertEqual(request.early_route_scaffold, _uses_early_route_scaffold(123))
        self.assertEqual(descriptor.early_route_scaffold, request.early_route_scaffold)
        self.assertEqual(descriptor.scenario, "early")
        self.assertEqual(descriptor.starting_position, "")
        self.assertEqual(save.call_args.kwargs["scenario_directory"], "early")

    def test_curriculum_early_mixed_request_is_distinct_and_unprepared(self):
        generation_number = next(
            game_number
            for game_number in range(len(MATURITY_CYCLE))
            if self._runner()._maturity_for_game(game_number).name == "early_mixed"
        )
        runner = self._runner(generation_number)
        runner._latest_descriptor = None
        generated = SimpleNamespace(
            starting_scores_by_seat=(2, 4, 6),
            starting_development_by_seat=(3, 4, 5),
            development_roles_by_seat=("low", "medium", "high"),
        )

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.hansa"
            metadata_path = Path(directory) / "state.json"
            with (
                mock.patch(
                    "training.balanced_curriculum.generate_balanced_state",
                    return_value=generated,
                ) as generate,
                mock.patch(
                    "training.balanced_curriculum.save_balanced_state",
                    return_value=(state_path, metadata_path),
                ) as save,
            ):
                descriptor = runner._generate_state(
                    SimpleNamespace(full_game=False),
                    123,
                    Path(directory),
                )

        request = generate.call_args.args[0]
        self.assertTrue(request.early_mixed_development)
        self.assertFalse(request.mixed_development)
        self.assertFalse(request.prepare_ending_condition)
        self.assertFalse(request.prepared_routes_one_short)
        self.assertFalse(request.early_route_scaffold)
        self.assertEqual(request.score_range, (2, 6))
        self.assertEqual(request.development_range, (3, 5))
        self.assertEqual(request.round_range, (3, 6))
        self.assertEqual(descriptor.scenario, "early_mixed")
        self.assertEqual(descriptor.starting_scores_by_seat, (2, 4, 6))
        self.assertEqual(descriptor.starting_development_by_seat, (3, 4, 5))
        self.assertEqual(descriptor.development_roles_by_seat, ("low", "medium", "high"))
        self.assertEqual(save.call_args.kwargs["scenario_directory"], "early_mixed")

    def test_non_early_curriculum_requests_keep_ending_preparation(self):
        for maturity_name in ("mid", "late", "end"):
            generation_number = next(
                number
                for number in range(len(MATURITY_CYCLE))
                if self._runner()._maturity_for_game(number).name == maturity_name
            )
            runner = self._runner(generation_number)
            runner._latest_descriptor = None
            with (
                tempfile.TemporaryDirectory() as directory,
                mock.patch(
                    "training.balanced_curriculum.generate_balanced_state",
                    return_value=SimpleNamespace(),
                ) as generate,
                mock.patch(
                    "training.balanced_curriculum.save_balanced_state",
                    return_value=(Path(directory) / "state.hansa", Path(directory) / "state.json"),
                ),
            ):
                runner._generate_state(SimpleNamespace(full_game=False), 123, Path(directory))

            request = generate.call_args.args[0]
            profile = next(item for item in MATURITY_PROFILES if item.name == maturity_name)
            self.assertTrue(request.prepare_ending_condition)
            self.assertTrue(request.prepared_routes_one_short)
            self.assertEqual(request.round_range, profile.round_range)
            self.assertFalse(request.early_route_scaffold)

    def test_east_west_and_isle_of_man_can_be_generated_together(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=124,
                map_num=3,
                player_count=4,
                ending_condition=EndingCondition.NEAR_SCORE,
                strategic_focus=StrategicFocus.EAST_WEST,
                regional_focus=RegionalFocus.ISLE_OF_MAN,
                use_emperors_favour=True,
                use_promo_markers=True,
            )
        )

        self.assertEqual(generated.game.map_num, 3)
        self.assertTrue(any(value.startswith("east_west:") for value in generated.focus_variants))
        self.assertTrue(
            any(value.startswith("britannia_isle_of_man:") for value in generated.focus_variants)
        )

    def test_near_score_isle_of_man_focus_generates_without_exhausting_attempts(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=0,
                map_num=3,
                player_count=5,
                ending_condition=EndingCondition.NEAR_SCORE,
                score_range=(16, 17),
                regional_focus=RegionalFocus.ISLE_OF_MAN,
                prepared_routes_one_short=True,
                development_range=(7, 9),
            ),
            max_attempts=100,
        )

        self.assertIn(
            "britannia_isle_of_man:Wales+Scotland",
            generated.focus_variants,
        )

    def test_dual_east_west_prepares_two_safe_competitors(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=410,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_SCORE,
                strategic_focus=StrategicFocus.DUAL_EAST_WEST,
            )
        )

        self.assertEqual(generated.focus_variants, ("east_west", "east_west_rival"))
        self.assertGreaterEqual(sum(player.score <= 8 for player in generated.game.players), 2)

    def test_blocked_east_west_prepares_an_opponent_on_the_final_route(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=701,
                map_num=1,
                player_count=4,
                ending_condition=EndingCondition.NEAR_SCORE,
                strategic_focus=StrategicFocus.BLOCKED_EAST_WEST,
                prepared_routes_one_short=True,
            )
        )

        contested_routes = [
            route
            for route in generated.game.selected_map.routes
            if len({post.owner for post in route.posts if post.owner is not None}) > 1
        ]
        self.assertTrue(contested_routes)
        self.assertIn("east_west_blocked", generated.focus_variants)

    def test_special_prestige_focus_has_a_legal_player_and_circle_route(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=411,
                map_num=2,
                player_count=4,
                ending_condition=EndingCondition.NEAR_BONUS_MARKERS,
                strategic_focus=StrategicFocus.SPECIAL_PRESTIGE,
                bonus_markers_remaining=1,
            )
        )
        game = generated.game
        prestige = game.selected_map.specialprestigepoints
        special_city = next(
            city
            for city in game.selected_map.cities
            if "SpecialPrestigePoints" in city.upgrade_city_type
        )

        self.assertTrue(
            any(
                route.is_controlled_by(player)
                and route.contains_a_circle()
                and prestige.can_claim_prestige(player)
                for player in game.players
                for route in special_city.routes
            )
        )

    def test_network_keys_focus_builds_three_to_seven_office_network(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=803,
                map_num=3,
                player_count=4,
                ending_condition=EndingCondition.NEAR_SCORE,
                strategic_focus=StrategicFocus.NETWORK_KEYS,
                prepared_routes_one_short=True,
            )
        )
        game = generated.game
        _network, office_count, _keys, key_value = generated.focus_variants[0].split("_")
        candidates = [
            player
            for player in game.players
            if player.keys == int(key_value)
            and game.calculate_largest_network(player) == int(office_count)
        ]

        self.assertTrue(candidates)
        self.assertIn(int(office_count), range(3, 8))
        self.assertIn(int(key_value), (2, 3, 4))
        self.assertTrue(
            any(
                sum(post.owner is None for post in route.posts) == 1
                and all(post.owner in (None, player) for post in route.posts)
                for player in candidates
                for route in game.selected_map.routes
            )
        )

    def test_close_finish_state_starts_with_a_claimable_route(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=412,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_COMPLETED_CITIES,
                completed_cities_below_limit=1,
                starting_position=StartingPosition.TWO_DECISIONS_BEFORE,
            )
        )
        game = generated.game
        prepared = game.players[generated.prepared_player_index]

        self.assertIs(game.current_player, prepared)
        self.assertTrue(any(route.is_controlled_by(prepared) for route in game.selected_map.routes))

    def test_configuration_shuffle_is_reproducible(self):
        expected = list(CONFIGURATIONS)
        random.Random(124).shuffle(expected)
        runner = self._runner()
        self.assertEqual(runner._configuration_for_game(), expected[0])

    def test_evaluation_generation_does_not_advance_training_sequence(self):
        runner = self._runner(7)
        runner._latest_descriptor = None
        stage = SimpleNamespace(full_game=True)

        with tempfile.TemporaryDirectory() as directory:
            runner._generate_state(stage, 55, Path(directory), map_num=2, player_count=3)

        self.assertEqual(runner.training_generation_number, 7)

    def test_training_generation_counter_is_saved_in_curriculum_state(self):
        runner = self._runner(12)
        runner.stage_index = 0
        runner.batch_number = 0
        runner.game_number = 99
        runner.rolling_losses = []
        runner.config = CurriculumConfig()

        state = runner._curriculum_state()

        self.assertEqual(state["training_generation_number"], 12)


if __name__ == "__main__":
    unittest.main()
