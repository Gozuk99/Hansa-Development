import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from game.action_validation import state_fingerprint, validate_action_state
from game.invariants import validate_game
from game.loaded_state_validation import validate_loaded_game
from game.persistence import load_game
from game.structured_actions import RouteInteraction
from map_data.constants import (
    BANK_MAX_VALUES,
    BOOK_OF_KNOWLEDGE_MAX_VALUES,
    PRIVILEGE_COLORS,
)
from training.balanced_state_generator import EndingCondition
from training.targeted_state_generator import (
    EndGameScenario,
    GenerationRequest,
    generate_state,
    save_generated_state,
)
from tools.generate_training_states import EVALUATION_SPECS, parse_args


class TargetedStateGeneratorTests(unittest.TestCase):
    def test_final_development_uses_one_shared_range_and_offices_are_legal(self):
        generated = generate_state(
            GenerationRequest(
                seed=97531,
                scenario=EndGameScenario.NEAR_BONUS_MARKERS,
                map_num=2,
                player_count=5,
            )
        )
        game = generated.game

        self.assertIn(generated.development_range, ((7, 9), (9, 11), (11, 13)))
        minimum, maximum = generated.development_range
        for player in game.players:
            upgrades = (
                player.keys_index
                + PRIVILEGE_COLORS.index(player.privilege)
                + BOOK_OF_KNOWLEDGE_MAX_VALUES.index(player.book)
                + player.actions_index
                + BANK_MAX_VALUES.index(player.bank)
            )
            offices = sum(
                office.controller is player
                for city in game.selected_map.cities
                for office in city.offices
            )
            self.assertGreaterEqual(upgrades + offices, minimum)
            self.assertLessEqual(upgrades + offices, maximum)

        for city in game.selected_map.cities:
            found_open = False
            for office in city.offices:
                if office.controller is None:
                    found_open = True
                    continue
                self.assertFalse(found_open)
                self.assertGreaterEqual(
                    PRIVILEGE_COLORS.index(office.controller.privilege),
                    PRIVILEGE_COLORS.index(office.printed_privilege or "WHITE"),
                )

    def test_evaluation_suite_covers_maps_players_endings_and_optional_rules(self):
        self.assertEqual(len(EVALUATION_SPECS), 27)
        configurations = Counter((spec.map_num, spec.player_count) for spec in EVALUATION_SPECS)
        self.assertEqual(set(configurations.values()), {3})
        self.assertEqual(Counter(spec.map_num for spec in EVALUATION_SPECS), {1: 9, 2: 9, 3: 9})
        self.assertEqual(
            Counter(spec.player_count for spec in EVALUATION_SPECS), {3: 9, 4: 9, 5: 9}
        )
        self.assertTrue(any(spec.mission_cards for spec in EVALUATION_SPECS))
        self.assertTrue(any(spec.emperors_favour for spec in EVALUATION_SPECS))
        self.assertTrue(any(spec.promo_markers for spec in EVALUATION_SPECS))
        self.assertEqual(
            Counter(spec.ending_condition for spec in EVALUATION_SPECS),
            {ending: 9 for ending in EndingCondition},
        )
        self.assertEqual(sum(spec.east_west for spec in EVALUATION_SPECS), 9)
        self.assertTrue(any(spec.east_west and spec.regional_focus for spec in EVALUATION_SPECS))
        self.assertEqual(sum(spec.immediate_finish for spec in EVALUATION_SPECS), 1)
        self.assertTrue(
            all(not spec.mission_cards or spec.map_num == 1 for spec in EVALUATION_SPECS)
        )
        manifest = json.loads(
            Path("training_data/generated/evaluation/manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue(
            all(
                "\\" not in entry[field]
                for entry in manifest
                for field in ("save_file", "metadata_file")
            )
        )
        self.assertTrue(parse_args(["--eval"]).eval)

    def test_every_targeted_scenario_generates_a_playable_state(self):
        cases = (
            [
                (scenario, map_num, player_count)
                for scenario in (
                    EndGameScenario.NEAR_SCORE,
                    EndGameScenario.NEAR_BONUS_MARKERS,
                    EndGameScenario.NEAR_COMPLETED_CITIES,
                    EndGameScenario.EAST_WEST,
                )
                for map_num in (1, 2, 3)
                for player_count in (3, 4, 5)
            ]
            + [(EndGameScenario.BRITANNIA_WALES, 3, player_count) for player_count in (3, 4, 5)]
            + [
                (scenario, 3, player_count)
                for scenario in (
                    EndGameScenario.BRITANNIA_SCOTLAND,
                    EndGameScenario.BRITANNIA_ISLE_OF_MAN,
                )
                for player_count in (4, 5)
            ]
        )
        for scenario_number, (scenario, map_num, player_count) in enumerate(cases):
            with self.subTest(
                scenario=scenario.value,
                map_num=map_num,
                player_count=player_count,
            ):
                generated = generate_state(
                    GenerationRequest(
                        seed=(10_000 + map_num * 100 + player_count * 10 + scenario_number),
                        scenario=scenario,
                        map_num=map_num,
                        player_count=player_count,
                        immediate_finish=True,
                        prepared_route_full=(
                            True
                            if scenario
                            in (
                                EndGameScenario.EAST_WEST,
                                EndGameScenario.BRITANNIA_WALES,
                                EndGameScenario.BRITANNIA_SCOTLAND,
                                EndGameScenario.BRITANNIA_ISLE_OF_MAN,
                            )
                            else None
                        ),
                    )
                )
                game = generated.game

                self.assertTrue(validate_game(game))
                self.assertTrue(validate_loaded_game(game))
                self.assertFalse(game.game_end)
                self.assertTrue(game.get_legal_actions())
                projected_scores = game.projected_scores()
                self.assertLessEqual(max(projected_scores) - min(projected_scores), 3)
                self.assertGreater(validate_action_state(game, quiet=True).legal_action_count, 0)
                self.assertEqual(
                    game.turn_number,
                    (game.round_number - 1) * player_count + game.current_player_index + 1,
                )
                self.assertTrue(
                    all(
                        player is game.current_player or player.actions_remaining == 0
                        for player in game.players
                    )
                )

                if scenario is EndGameScenario.NEAR_SCORE:
                    self.assertTrue(all(player.score in (17, 18) for player in game.players))
                    self.assertEqual(game.current_player.score, 18)
                    self.assertTrue(
                        all(
                            any(
                                office.controller is player
                                for city in game.selected_map.cities
                                for office in city.offices
                            )
                            for player in game.players
                        )
                    )
                    scoring_routes = [
                        route_index
                        for route_index, route in enumerate(game.selected_map.routes)
                        if route.is_controlled_by(game.current_player)
                        and all(
                            city.determine_controller() is game.current_player
                            for city in route.cities
                        )
                    ]
                    self.assertTrue(scoring_routes)
                    claim = RouteInteraction(scoring_routes[0], 0)
                    self.assertIn(claim, game.get_legal_actions())
                    game.apply_structured_action(claim)
                    self.assertTrue(game.game_end)
                    self.assertGreaterEqual(game.current_player.score, 20)
                elif scenario is EndGameScenario.NEAR_BONUS_MARKERS:
                    scores = [player.score for player in game.players]
                    self.assertLessEqual(max(scores) - min(scores), 1)
                    self.assertFalse(game.selected_map.bonus_marker_pool)
                    marker_routes = [
                        (route_index, route)
                        for route_index, route in enumerate(game.selected_map.routes)
                        if route.bonus_marker is not None
                        and route.is_controlled_by(game.current_player)
                    ]
                    self.assertTrue(marker_routes)
                    route_index, _route = marker_routes[0]
                    claim = RouteInteraction(route_index, 0)
                    self.assertIn(claim, game.get_legal_actions())
                    game.apply_structured_action(claim)
                    self.assertTrue(game.bonus_pool_exhausted_during_claim)
                    self.assertTrue(game.game_end)
                elif scenario is EndGameScenario.NEAR_COMPLETED_CITIES:
                    scores = [player.score for player in game.players]
                    self.assertLessEqual(max(scores) - min(scores), 1)
                    self.assertEqual(
                        game.current_full_cities_count,
                        game.selected_map.max_full_cities - 1,
                    )
                    controlled_routes = [
                        route
                        for route in game.selected_map.routes
                        if route.is_controlled_by(game.current_player)
                    ]
                    self.assertTrue(controlled_routes)
                    office_actions = [
                        action
                        for action in game.get_legal_actions()
                        if isinstance(action, RouteInteraction)
                        and action.interaction_slot in (1, 2)
                        and sum(
                            office.controller is None
                            for office in game.selected_map.routes[action.route_slot]
                            .cities[action.interaction_slot - 1]
                            .offices
                        )
                        == 1
                    ]
                    self.assertTrue(office_actions)
                    game.apply_structured_action(office_actions[0])
                    self.assertEqual(
                        game.current_full_cities_count,
                        game.selected_map.max_full_cities,
                    )
                    self.assertTrue(game.game_end or game.game_end_pending_immediate_resolution)
                elif scenario is EndGameScenario.EAST_WEST:
                    self.assertFalse(
                        game.has_east_west_connection(*game.selected_map.east_west_cities)
                    )
                    self.assertTrue(self._has_completing_east_west_office(game))
                else:
                    self.assertEqual(game.map_num, 3)
                    before = game.calculate_britannia_region_points()
                    self.assertTrue(
                        self._has_improving_britannia_office(
                            game,
                            before,
                            isle_only=(scenario is EndGameScenario.BRITANNIA_ISLE_OF_MAN),
                        )
                    )

    @staticmethod
    def _prepared_turn(game, player_index=None):
        candidate = copy.deepcopy(game)
        if player_index is None:
            return candidate
        for player in candidate.players:
            player.actions_remaining = 0
            player.ending_turn = False
        candidate.current_player_index = player_index
        candidate.current_player = candidate.players[player_index]
        candidate.active_player = player_index
        candidate.current_player.start_turn(
            extra_actions=int(candidate.OneActionOwner is candidate.current_player)
        )
        if candidate.map_num == 3:
            candidate.current_player.refresh_map3_priv_actions(candidate)
        return candidate

    @classmethod
    def _has_completing_east_west_office(cls, game, player_index=None):
        game = cls._prepared_turn(game, player_index)
        player_index = game.current_player_index
        for action in game.get_legal_actions():
            if not isinstance(action, RouteInteraction) or action.interaction_slot not in (1, 2):
                continue
            candidate = copy.deepcopy(game)
            candidate.apply_structured_action(action)
            if candidate.players[player_index] in candidate.players_who_completed_east_west:
                return True
        return False

    @classmethod
    def _has_improving_britannia_office(cls, game, before, isle_only, player_index=None):
        game = cls._prepared_turn(game, player_index)
        before = game.calculate_britannia_region_points()
        player_index = game.current_player_index
        for action in game.get_legal_actions():
            if not isinstance(action, RouteInteraction) or action.interaction_slot not in (1, 2):
                continue
            route = game.selected_map.routes[action.route_slot]
            city = route.cities[action.interaction_slot - 1]
            if isle_only and city.name != "IsleOfMan":
                continue
            candidate = copy.deepcopy(game)
            candidate.apply_structured_action(action)
            after = candidate.calculate_britannia_region_points()
            if after[candidate.players[player_index]] > before[game.players[player_index]]:
                return True
        return False

    def test_generation_is_deterministic(self):
        request = GenerationRequest(
            seed=8675309,
            scenario=EndGameScenario.NEAR_COMPLETED_CITIES,
            map_num=2,
            player_count=5,
            use_mission_cards=False,
            use_emperors_favour=True,
            use_promo_markers=True,
        )

        first = generate_state(request)
        second = generate_state(request)

        self.assertEqual(first.attempt_seed, second.attempt_seed)
        self.assertEqual(state_fingerprint(first.game), state_fingerprint(second.game))

    def test_default_position_gives_every_other_player_a_turn_before_prepared_player(self):
        generated = generate_state(
            GenerationRequest(
                seed=2468,
                scenario=EndGameScenario.NEAR_SCORE,
                map_num=2,
                player_count=5,
            )
        )
        game = generated.game
        prepared_player = next(
            player
            for player in game.players
            if any(
                route.is_controlled_by(player)
                and all(city.determine_controller() is player for city in route.cities)
                for route in game.selected_map.routes
            )
        )

        self.assertFalse(generated.immediate_finish)
        self.assertEqual(
            game.current_player_index,
            (game.players.index(prepared_player) + 1) % len(game.players),
        )

    def test_one_round_bonus_marker_position_has_only_the_prepared_marker(self):
        generated = generate_state(
            GenerationRequest(
                seed=1357,
                scenario=EndGameScenario.NEAR_BONUS_MARKERS,
                map_num=1,
                player_count=4,
            )
        )
        marker_routes = [
            route for route in generated.game.selected_map.routes if route.bonus_marker is not None
        ]

        self.assertEqual(len(marker_routes), 1)
        owners = {post.owner for post in marker_routes[0].posts}
        self.assertEqual(len(owners), 1)
        self.assertNotIn(None, owners)

    def test_saved_state_is_organized_and_round_trips(self):
        generated = generate_state(
            GenerationRequest(
                seed=42,
                scenario=EndGameScenario.NEAR_SCORE,
                map_num=1,
                player_count=3,
                use_mission_cards=True,
                use_emperors_favour=True,
                use_promo_markers=True,
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_path, metadata_path = save_generated_state(generated, root)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            restored = load_game(save_path)

            self.assertEqual(save_path.parent, root / "near_score" / "map_1" / "3_players")
            self.assertEqual(metadata_path.with_suffix(".hansa"), save_path)
            self.assertEqual(metadata["scenario"], "near_score")
            self.assertEqual(
                metadata["options"],
                {
                    "mission_cards": True,
                    "emperors_favour": True,
                    "promo_bonus_markers": True,
                },
            )
            self.assertEqual(restored.rng.getstate(), generated.game.rng.getstate())
            self.assertEqual(restored.get_legal_actions(), generated.game.get_legal_actions())
            self.assertEqual(restored.ai_action_mask(), generated.game.ai_action_mask())
            self.assertEqual(
                [player.score for player in restored.players],
                [player.score for player in generated.game.players],
            )

    def test_random_request_resolves_to_supported_configuration(self):
        generated = generate_state(GenerationRequest(seed=1234))
        game = generated.game

        self.assertIn(generated.scenario, EndGameScenario)
        self.assertIn(game.map_num, (1, 2, 3))
        self.assertIn(len(game.players), (3, 4, 5))
        self.assertTrue(not game.use_mission_cards or game.map_num == 1)

    def test_current_player_receives_emperors_extra_action(self):
        generated = generate_state(
            GenerationRequest(
                seed=7,
                scenario=EndGameScenario.NEAR_SCORE,
                map_num=1,
                player_count=3,
                use_mission_cards=False,
                use_emperors_favour=True,
                use_promo_markers=False,
            )
        )
        game = generated.game

        expected_extra_action = int(game.OneActionOwner is game.current_player)
        self.assertEqual(
            game.current_player.actions_remaining,
            game.current_player.actions + expected_extra_action,
        )

    def test_near_score_uses_only_seventeen_or_eighteen_points(self):
        generated = generate_state(
            GenerationRequest(
                seed=81,
                scenario=EndGameScenario.NEAR_SCORE,
                score_range=(8, 14),
                immediate_finish=True,
            )
        )

        self.assertTrue(all(player.score in (17, 18) for player in generated.game.players))
        self.assertEqual(generated.score_range, (17, 18))

    def test_near_score_state_prepares_only_one_completed_route(self):
        generated = generate_state(
            GenerationRequest(
                seed=2468,
                scenario=EndGameScenario.NEAR_SCORE,
                map_num=2,
                player_count=5,
                score_range=(10, 17),
                immediate_finish=True,
            )
        )

        occupied_routes = [
            route
            for route in generated.game.selected_map.routes
            if any(post.owner is not None for post in route.posts)
        ]
        self.assertEqual(len(occupied_routes), 1)
        self.assertTrue(occupied_routes[0].is_controlled_by(generated.game.current_player))


if __name__ == "__main__":
    unittest.main()
