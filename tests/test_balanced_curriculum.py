import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from training.balanced_curriculum import (
    CONFIGURATIONS,
    MATURITY_CYCLE,
    MATURITY_PROFILES,
    BalancedCurriculumRunner,
    _select_focus,
)
from training.balanced_state_generator import (
    BalancedGenerationRequest,
    EndingCondition,
    RegionalFocus,
    StartingPosition,
    StrategicFocus,
    generate_balanced_state,
)
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

    def test_active_maturity_cycle_focuses_on_late_and_end_games(self):
        runner = self._runner()
        observed = Counter(
            runner._maturity_for_game(game_number).name
            for game_number in range(len(MATURITY_CYCLE))
        )

        self.assertEqual(
            observed,
            {"late": 4, "end": 1},
        )
        self.assertEqual(sum(profile.weight for profile in MATURITY_PROFILES), 5)

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

        self.assertEqual(set(profiles), {"late", "end"})
        self.assertEqual(profiles["late"].bonus_markers_remaining, 2)
        self.assertEqual(profiles["end"].completed_cities_below_limit, 2)
        self.assertIs(
            profiles["end"].starting_position,
            StartingPosition.TWO_DECISIONS_BEFORE,
        )

    def test_early_state_applies_early_scores_and_development(self):
        generated = generate_balanced_state(
            BalancedGenerationRequest(
                seed=901,
                map_num=1,
                player_count=3,
                ending_condition=EndingCondition.NEAR_BONUS_MARKERS,
                score_range=(0, 5),
                development_range=(2, 4),
                bonus_markers_remaining=5,
                completed_cities_below_limit=7,
                prepared_routes_one_short=True,
            )
        )

        self.assertTrue(all(0 <= player.score <= 5 for player in generated.game.players))
        self.assertEqual(len(generated.game.selected_map.bonus_marker_pool), 5)
        minimum, maximum = generated.development_range
        self.assertEqual((minimum, maximum), (2, 4))

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

    def test_close_finish_state_requires_placement_before_route_completion(self):
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
        self.assertTrue(
            any(
                sum(post.owner is None for post in route.posts) == 1
                and all(post.owner in (None, prepared) for post in route.posts)
                for route in game.selected_map.routes
            )
        )

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
