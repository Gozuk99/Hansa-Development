import json
from pathlib import Path
import tempfile
import unittest

from game.action_validation import state_fingerprint, validate_action_state
from game.invariants import validate_game
from game.loaded_state_validation import validate_loaded_game
from game.persistence import load_game
from training.targeted_state_generator import (
    EndGameScenario,
    GenerationRequest,
    generate_state,
    save_generated_state,
)


class TargetedStateGeneratorTests(unittest.TestCase):
    def test_every_targeted_scenario_generates_a_playable_state(self):
        for scenario_number, scenario in enumerate(EndGameScenario):
            for map_num in (1, 2, 3):
                for player_count in (3, 4, 5):
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
                            )
                        )
                        game = generated.game

                        self.assertTrue(validate_game(game))
                        self.assertTrue(validate_loaded_game(game))
                        self.assertFalse(game.game_end)
                        self.assertTrue(game.get_legal_actions())
                        self.assertGreater(
                            validate_action_state(game, quiet=True).legal_action_count, 0
                        )
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
                            self.assertTrue(
                                all(17 <= player.score <= 19 for player in game.players)
                            )
                        elif scenario is EndGameScenario.NEAR_BONUS_MARKERS:
                            self.assertIn(len(game.selected_map.bonus_marker_pool), (1, 2))
                        else:
                            self.assertEqual(
                                game.current_full_cities_count,
                                game.selected_map.max_full_cities - 1,
                            )

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

        self.assertIs(game.OneActionOwner, game.current_player)
        self.assertEqual(game.current_player.actions_remaining, game.current_player.actions + 1)


if __name__ == "__main__":
    unittest.main()
