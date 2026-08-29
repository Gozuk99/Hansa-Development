import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from game.invariants import validate_game
from game.loaded_state_validation import validate_loaded_game
from game.persistence import load_game
from training.balanced_curriculum import (
    CONFIGURATIONS,
    MATURITY_CYCLE,
    MATURITY_PROFILES,
    BalancedCurriculumRunner,
    _scenario_condition_label,
    _select_focus,
    _select_fresh_optional_modules,
    _select_optional_modules,
)
from training.balanced_state_generator import (
    BalancedGenerationRequest,
    BonusMarkerSetup,
    EndingCondition,
    RegionalFocus,
    StartingPosition,
    StrategicFocus,
    generate_balanced_state,
    player_has_starting_score_source,
    starting_scores_have_valid_sources,
)
from map_data.map_attributes import Map
from training.curriculum import (
    DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS,
    CurriculumConfig,
)
from training.targeted_state_generator import _fill_prepared_route


class BalancedCurriculumTests(unittest.TestCase):
    @staticmethod
    def _runner(training_generation_number=0):
        runner = object.__new__(BalancedCurriculumRunner)
        runner.config = SimpleNamespace(
            seed=124,
            zero_epsilon_training_fractions=DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS,
        )
        runner.game_number = training_generation_number
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
                "fresh": 10,
                "mid": 3,
                "late": 4,
                "end": 3,
            },
        )
        total = sum(profile.weight for profile in MATURITY_PROFILES)
        self.assertEqual(total, 20)
        self.assertEqual(observed["fresh"] / total, 0.50)
        self.assertEqual(observed["early"] / total, 0.00)
        self.assertEqual(observed["mid"] / total, 0.15)
        self.assertEqual(observed["late"] / total, 0.20)
        self.assertEqual(observed["end"] / total, 0.15)
        self.assertEqual(
            (observed["mid"], observed["late"], observed["end"]),
            (3, 4, 3),
        )
        self.assertEqual(
            runner._stage_label(None),
            "fresh_mid_late_end_game",
        )

    def test_zero_epsilon_selection_uses_deterministic_maturity_schedule(self):
        runner = self._runner()
        expected_rates = {
            "fresh": 0.05,
            "early": 0.05,
            "mid": 0.05,
            "late": 0.50,
            "end": 1.00,
        }
        for maturity, expected_rate in expected_rates.items():
            first = [runner._training_exploration_mode(maturity, game) for game in range(10_000)]
            second = [
                self._runner()._training_exploration_mode(maturity, game) for game in range(10_000)
            ]
            self.assertEqual(first, second)
            observed_rate = first.count("zero_epsilon") / len(first)
            self.assertAlmostEqual(observed_rate, expected_rate, delta=0.02)

        late = [runner._training_exploration_mode("late", game) for game in range(100)]
        self.assertIn("normal", late)
        self.assertIn("zero_epsilon", late)

    def test_fallback_evaluation_maturity_distribution_remains_unchanged(self):
        runner = self._runner()
        observed = Counter(
            runner._evaluation_maturity_for_seed(seed).name for seed in range(10_000)
        )

        self.assertEqual(set(observed), {"early", "mid", "late", "end"})
        self.assertEqual(observed, {"early": 4_000, "mid": 3_000, "late": 2_000, "end": 1_000})

    def test_fresh_optional_modules_have_requested_seeded_distribution(self):
        selections = [
            _select_fresh_optional_modules(random.Random(seed), 1) for seed in range(12_000)
        ]
        mission_rate = sum(missions for missions, _favour, _markers in selections) / len(selections)
        favour_rate = sum(favour for _missions, favour, _markers in selections) / len(selections)
        marker_counts = Counter(markers for _missions, _favour, markers in selections)

        self.assertTrue(0.48 < mission_rate < 0.52)
        self.assertTrue(0.48 < favour_rate < 0.52)
        for marker_setup in BonusMarkerSetup:
            self.assertTrue(0.31 < marker_counts[marker_setup] / len(selections) < 0.35)
        self.assertTrue(
            all(
                not _select_fresh_optional_modules(random.Random(seed), map_num)[0]
                for map_num in (2, 3)
                for seed in range(100)
            )
        )

    def test_fresh_curriculum_uses_canonical_untouched_setup_for_every_configuration(self):
        fresh_number = next(
            number
            for number in range(len(MATURITY_CYCLE))
            if self._runner()._maturity_for_game(number).name == "fresh"
        )
        observed_marker_routes = set()
        for map_num, player_count in CONFIGURATIONS:
            with self.subTest(map_num=map_num, player_count=player_count):
                runner = self._runner(fresh_number)
                runner._latest_descriptor = None
                with (
                    tempfile.TemporaryDirectory() as directory,
                    mock.patch.object(
                        runner,
                        "_configuration_for_game",
                        return_value=(map_num, player_count),
                    ),
                ):
                    descriptor = runner._generate_state(
                        SimpleNamespace(full_game=False),
                        20_000 + map_num * 100 + player_count,
                        Path(directory),
                    )
                    game = load_game(descriptor.path)

                self.assertEqual(descriptor.scenario, "fresh")
                self.assertEqual(descriptor.starting_position, "")
                self.assertEqual(game.turn_number, 1)
                self.assertEqual(game.round_number, 1)
                self.assertEqual(game.current_player_index, 0)
                self.assertFalse(game.game_end)
                self.assertTrue(all(player.score == 0 for player in game.players))
                self.assertTrue(
                    all(
                        (
                            player.keys,
                            player.privilege,
                            player.book,
                            player.actions,
                            player.bank,
                        )
                        == (1, "WHITE", 2, 2, 3)
                        for player in game.players
                    )
                )
                self.assertTrue(
                    all(
                        office.controller is None
                        for city in game.selected_map.cities
                        for office in city.offices
                    )
                )
                self.assertFalse(any(route.has_tradesmen() for route in game.selected_map.routes))
                self.assertTrue(
                    all(
                        not player.bonus_markers and not player.used_bonus_markers
                        for player in game.players
                    )
                )
                self.assertEqual(len(game.selected_map.bonus_marker_pool), 12)
                self.assertEqual(
                    {
                        route.bonus_marker.type
                        for route in game.selected_map.routes
                        if route.bonus_marker is not None
                    },
                    {"Move3", "SwapOffice", "PlaceAdjacent"},
                )
                self.assertEqual(descriptor.mission_cards_enabled, game.use_mission_cards)
                self.assertEqual(descriptor.emperors_favour_enabled, game.use_emperors_favour)
                self.assertEqual(descriptor.emperors_favour_tiles, tuple(game.tile_pool))
                self.assertEqual(
                    descriptor.bonus_marker_draw_supply,
                    tuple(game.selected_map.bonus_marker_pool),
                )
                self.assertTrue(validate_game(game))
                self.assertTrue(validate_loaded_game(game))
                observed_marker_routes.add(descriptor.starting_bonus_marker_routes)

        self.assertGreater(len(observed_marker_routes), 1)

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

    def test_only_early_training_receives_the_extended_action_limit(self):
        runner = self._runner()
        stage = SimpleNamespace(name="mixed", action_limit=10_000)

        self.assertEqual(
            runner._training_action_limit(stage, SimpleNamespace(scenario="early")),
            15_000,
        )
        for maturity in ("fresh", "mid", "late", "end"):
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
        mid = next(profile for profile in MATURITY_PROFILES if profile.name == "mid")
        late = next(profile for profile in MATURITY_PROFILES if profile.name == "late")

        self.assertTrue(
            all(
                _scenario_condition_label(early, condition) is None for condition in EndingCondition
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

        self.assertEqual(set(profiles), {"fresh", "early", "mid", "late", "end"})
        self.assertTrue(profiles["fresh"].fresh)
        self.assertEqual(profiles["fresh"].round_range, (1, 1))
        self.assertEqual(profiles["fresh"].starting_position_label, "")
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
        self.assertFalse(any(route.has_tradesmen() for route in generated.game.selected_map.routes))
        self.assertTrue(validate_game(generated.game))
        self.assertTrue(validate_loaded_game(generated.game))
        self.assertTrue(2 <= generated.game.round_number <= 5)
        self.assertEqual(
            generated.game.turn_number,
            (generated.game.round_number - 1) * 3 + generated.game.current_player_index + 1,
        )

    def test_curriculum_early_request_does_not_prepare_an_ending(self):
        runner = self._runner()
        early = next(profile for profile in MATURITY_PROFILES if profile.name == "early")
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
                with mock.patch.object(runner, "_maturity_for_game", return_value=early):
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
        self.assertEqual(descriptor.scenario, "early")
        self.assertEqual(descriptor.starting_position, "")
        self.assertEqual(save.call_args.kwargs["scenario_directory"], "early")

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
