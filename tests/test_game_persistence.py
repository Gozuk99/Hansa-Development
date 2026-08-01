import base64
import json
import random
import hashlib
import pickle
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from drawing.game_window import GameWindow
from drawing.new_game_menu import NewGameMenu
from game.action_schema import ACTION_SCHEMA_VERSION
from game.game_config import GameConfiguration, PlayerControl
from game.persistence import SAVE_FORMAT_VERSION, SaveGameError, load_game, save_game
from game.persistence import default_save_directory
from game.loaded_state_validation import validate_loaded_game
from game.invariants import GameInvariantError
from map_data.map_attributes import BonusMarker, Map


class UnsafeSavedObject:
    def __reduce__(self):
        return eval, ("40 + 2",)


class ExactGamePersistenceTests(unittest.TestCase):
    def configured_game(self):
        configuration = GameConfiguration(
            map_num=1,
            player_count=5,
            player_controls=(PlayerControl.HUMAN,) * 5,
            use_mission_cards=True,
            use_emperors_favour=True,
            seed=124,
        )
        return configuration.create_game()

    def test_default_save_directory_is_user_owned(self):
        with mock.patch.dict("os.environ", {"LOCALAPPDATA": "C:/UserData"}):
            directory = default_save_directory()

        self.assertNotEqual(directory, Path.cwd() / "saves")

    def test_exact_save_round_trip_preserves_state_and_legal_choices(self):
        game = self.configured_game()
        first_action = next(index for index, enabled in enumerate(game.ai_action_mask()) if enabled)
        game.apply_ai_action(first_action)
        before_rng = game.rng.getstate()
        before_actions = game.get_legal_actions()
        before_mask = game.ai_action_mask()

        with tempfile.TemporaryDirectory() as directory:
            filename = save_game(game, Path(directory) / "tricky-position")
            restored = load_game(filename)

        self.assertEqual(filename.suffix, ".hansa")
        self.assertEqual(restored.map_num, game.map_num)
        self.assertEqual(restored.turn_number, game.turn_number)
        self.assertEqual(restored.round_number, game.round_number)
        self.assertEqual(restored.current_player_index, game.current_player_index)
        self.assertEqual(restored.turn_phase, game.turn_phase)
        self.assertEqual(restored.rng.getstate(), before_rng)
        self.assertEqual(restored.get_legal_actions(), before_actions)
        self.assertEqual(restored.ai_action_mask(), before_mask)
        self.assertEqual(restored.configuration, game.configuration)
        self.assertTrue(all(player.control is PlayerControl.HUMAN for player in restored.players))

    def test_save_metadata_identifies_format_schema_and_position(self):
        game = self.configured_game()
        with tempfile.TemporaryDirectory() as directory:
            filename = save_game(game, Path(directory) / "position.hansa")
            metadata = json.loads(filename.read_text(encoding="utf-8"))["metadata"]

        self.assertEqual(metadata["save_format_version"], SAVE_FORMAT_VERSION)
        self.assertEqual(metadata["action_schema_version"], ACTION_SCHEMA_VERSION)
        self.assertEqual(metadata["map_num"], 1)
        self.assertEqual(metadata["player_count"], 5)
        self.assertEqual(metadata["turn_phase"], game.turn_phase.value)

    def test_pending_displacement_round_trip_can_continue(self):
        game = GameConfiguration(map_num=2, seed=124).create_game()
        opponent = game.players[1]
        route = game.selected_map.routes[0]
        route.posts[0].claim(opponent, "square")
        opponent.personal_supply_squares -= 1
        game.apply_action(0)
        self.assertTrue(game.waiting_for_displaced_player)

        with tempfile.TemporaryDirectory() as directory:
            filename = save_game(game, Path(directory) / "during-displacement.hansa")
            restored = load_game(filename)

        self.assertTrue(restored.waiting_for_displaced_player)
        self.assertEqual(
            restored.displaced_player.player.order,
            game.displaced_player.player.order,
        )
        self.assertEqual(
            restored.displaced_player.displaced_shape,
            game.displaced_player.displaced_shape,
        )
        self.assertEqual(
            restored.displaced_player.total_pieces_to_place,
            game.displaced_player.total_pieces_to_place,
        )
        self.assertEqual(restored.get_legal_actions(), game.get_legal_actions())
        self.assertEqual(restored.ai_action_mask(), game.ai_action_mask())

    def test_round_trip_preserves_controller_random_state(self):
        game = self.configured_game()
        controller_rng = random.Random(991)
        controller_rng.random()
        expected_state = controller_rng.getstate()

        with tempfile.TemporaryDirectory() as directory:
            filename = save_game(
                game,
                Path(directory) / "ai-turn.hansa",
                controller_rng_state=expected_state,
            )
            restored = load_game(filename)

        self.assertEqual(restored._saved_controller_rng_state, expected_state)

    def test_load_rejects_damaged_payload(self):
        game = self.configured_game()
        with tempfile.TemporaryDirectory() as directory:
            filename = save_game(game, Path(directory) / "position.hansa")
            document = json.loads(filename.read_text(encoding="utf-8"))
            payload = bytearray(base64.b64decode(document["payload"]))
            payload[-1] ^= 1
            document["payload"] = base64.b64encode(payload).decode("ascii")
            filename.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(SaveGameError, "damaged or has been modified"):
                load_game(filename)

    def test_load_rejects_incompatible_save_or_action_schema(self):
        game = self.configured_game()
        with tempfile.TemporaryDirectory() as directory:
            filename = save_game(game, Path(directory) / "position.hansa")
            original = json.loads(filename.read_text(encoding="utf-8"))

            wrong_format = {**original, "metadata": dict(original["metadata"])}
            wrong_format["metadata"]["save_format_version"] = SAVE_FORMAT_VERSION + 1
            filename.write_text(json.dumps(wrong_format), encoding="utf-8")
            with self.assertRaisesRegex(SaveGameError, "incompatible save format"):
                load_game(filename)

            wrong_schema = {**original, "metadata": dict(original["metadata"])}
            wrong_schema["metadata"]["action_schema_version"] = ACTION_SCHEMA_VERSION + 1
            filename.write_text(json.dumps(wrong_schema), encoding="utf-8")
            with self.assertRaisesRegex(SaveGameError, "incompatible action schema"):
                load_game(filename)

    def test_load_rejects_executable_pickle_content(self):
        game = self.configured_game()
        with tempfile.TemporaryDirectory() as directory:
            filename = save_game(game, Path(directory) / "position.hansa")
            document = json.loads(filename.read_text(encoding="utf-8"))
            payload = pickle.dumps(UnsafeSavedObject())
            document["payload"] = base64.b64encode(payload).decode("ascii")
            document["metadata"]["payload_sha256"] = hashlib.sha256(payload).hexdigest()
            filename.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(SaveGameError, "forbidden type"):
                load_game(filename)

    def test_load_rejects_non_object_metadata_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "bad.hansa"
            filename.write_text(
                json.dumps({"metadata": [], "payload": "ignored"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SaveGameError, "metadata must be a JSON object"):
                load_game(filename)

    def test_menu_loads_selected_saved_game(self):
        loaded_game = self.configured_game()
        menu = object.__new__(NewGameMenu)
        menu.error = ""
        menu.result = None
        menu.running = True

        with (
            mock.patch("drawing.new_game_menu.choose_load_file", return_value=Path("save.hansa")),
            mock.patch("drawing.new_game_menu.load_game", return_value=loaded_game),
        ):
            menu._load_saved_game()

        self.assertIs(menu.result, loaded_game)
        self.assertFalse(menu.running)

    def test_menu_reports_file_picker_failure(self):
        menu = object.__new__(NewGameMenu)
        menu.error = ""
        menu.result = None
        menu.running = True

        with mock.patch(
            "drawing.new_game_menu.choose_load_file",
            side_effect=RuntimeError("file picker unavailable"),
        ):
            menu._load_saved_game()

        self.assertEqual(menu.error, "file picker unavailable")
        self.assertTrue(menu.running)

    def test_game_window_save_uses_selected_filename(self):
        game = self.configured_game()
        window = GameWindow.__new__(GameWindow)
        window.game = game
        window.rng = random.Random(124)
        window.save_status = ""
        expected = Path("chosen.hansa")

        with (
            mock.patch("drawing.game_window.choose_save_file", return_value=expected),
            mock.patch("drawing.game_window.save_game", return_value=expected) as save,
        ):
            window.save_current_game()

        save.assert_called_once_with(
            game,
            expected,
            controller_rng_state=window.rng.getstate(),
        )
        self.assertEqual(window.save_status, "Saved: chosen.hansa")

    def test_game_window_reports_file_picker_failure(self):
        window = GameWindow.__new__(GameWindow)
        window.game = self.configured_game()
        window.rng = random.Random(124)
        window.save_status = ""

        with mock.patch(
            "drawing.game_window.choose_save_file",
            side_effect=RuntimeError("file picker unavailable"),
        ):
            window.save_current_game()

        self.assertEqual(window.save_status, "Save failed: file picker unavailable")

    def test_saved_state_validation_rejects_duplicate_player_identity(self):
        game = self.configured_game()
        game.players[1].color = game.players[0].color

        with self.assertRaisesRegex(GameInvariantError, "player colors repeat"):
            validate_loaded_game(game)

    def test_saved_state_validation_rejects_ability_index_mismatch(self):
        game = self.configured_game()
        game.players[0].keys = 4

        with self.assertRaisesRegex(GameInvariantError, "Keys value/index disagree"):
            validate_loaded_game(game)

    def test_saved_state_validation_rejects_disabled_optional_content(self):
        game = GameConfiguration(map_num=1, seed=124).create_game()
        game.players[0].mission_card = ["Arnheim", "Stendal", "Minden"]

        with self.assertRaisesRegex(GameInvariantError, "Mission Card exists while disabled"):
            validate_loaded_game(game)

    def test_saved_state_validation_rejects_bonus_marker_count_changes(self):
        game = self.configured_game()
        game.selected_map.bonus_marker_pool.pop()

        with self.assertRaisesRegex(GameInvariantError, "must contain 15 markers"):
            validate_loaded_game(game)

    def test_saved_state_validation_rejects_wrong_selected_map(self):
        game = self.configured_game()
        game.map_num = 2

        with self.assertRaisesRegex(GameInvariantError, "selected map does not match"):
            validate_loaded_game(game)

    def test_saved_state_validation_rejects_excess_actions(self):
        game = self.configured_game()
        game.current_player.actions_remaining = 99

        with self.assertRaisesRegex(GameInvariantError, "too many remaining actions"):
            validate_loaded_game(game)

    def test_saved_state_validation_rejects_stale_pending_choice(self):
        game = self.configured_game()
        game.exchange_target_player = game.players[1]

        with self.assertRaisesRegex(GameInvariantError, "stale Exchange Bonus Marker target"):
            validate_loaded_game(game)

    def test_pending_exchange_marker_is_a_valid_saved_workflow(self):
        supply = [
            marker
            for marker, count in Map.STANDARD_BONUS_MARKER_SUPPLY.items()
            for _ in range(count)
        ]
        supply[-1] = "ExchangeBonusMarker"
        game = GameConfiguration(
            map_num=1,
            use_promo_markers=True,
            promo_marker_mode="manual",
            promo_markers=tuple(supply),
            seed=124,
        ).create_game()
        game.selected_map.bonus_marker_pool.remove("ExchangeBonusMarker")
        game.pending_exchange_marker = BonusMarker("ExchangeBonusMarker", game.current_player)
        game.waiting_for_bm_exchange_bm = True

        self.assertTrue(validate_loaded_game(game))

    def test_saved_state_validation_rejects_unfinished_mandatory_game_end(self):
        game = self.configured_game()
        game.current_player.score = 20

        with self.assertRaisesRegex(GameInvariantError, "passed a mandatory ending condition"):
            validate_loaded_game(game)

    def test_saved_state_validation_rejects_final_scoring_in_active_game(self):
        game = self.configured_game()
        game.current_player.final_score = 1
        game.current_player.final_score_breakdown = {"Initial Points": 1}

        with self.assertRaisesRegex(GameInvariantError, "active game contains final scoring"):
            validate_loaded_game(game)

    def test_saved_state_validation_rejects_incorrect_completed_city_count(self):
        game = self.configured_game()
        game.current_full_cities_count = 1

        with self.assertRaisesRegex(GameInvariantError, "completed-city count"):
            validate_loaded_game(game)

    def test_save_rejects_structurally_invalid_engine_state(self):
        game = self.configured_game()
        game.players[1].color = game.players[0].color
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "invalid.hansa"
            with self.assertRaisesRegex(SaveGameError, "player colors repeat"):
                save_game(game, filename)

    def test_save_wraps_validation_errors_for_the_gui(self):
        game = self.configured_game()
        game.players[1].color = game.players[0].color

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SaveGameError, "Cannot save invalid game state"):
                save_game(game, Path(directory) / "invalid.hansa")

    def test_saved_state_validation_tracks_only_current_turn_action_grants(self):
        game = self.configured_game()
        player = game.current_player
        player.grant_actions(3)
        self.assertTrue(validate_loaded_game(game))

        player.actions_granted_this_turn = 0
        with self.assertRaisesRegex(GameInvariantError, "too many remaining actions"):
            validate_loaded_game(game)

    def test_action_grant_tracking_clears_when_turn_advances(self):
        game = self.configured_game()
        player = game.current_player
        player.grant_actions(3)
        while player.actions_remaining:
            player.spend_action()

        game.advance_turn()

        self.assertEqual(player.actions_granted_this_turn, 0)
        self.assertTrue(validate_loaded_game(game))

    def test_saved_state_validation_rejects_missing_mandatory_displaced_piece(self):
        game = GameConfiguration(map_num=2, seed=124).create_game()
        opponent = game.players[1]
        route = game.selected_map.routes[0]
        route.posts[0].claim(opponent, "square")
        opponent.personal_supply_squares -= 1
        game.apply_action(0)
        game.displaced_player.total_pieces_to_place = 0

        with self.assertRaisesRegex(GameInvariantError, "impossible remaining-piece count"):
            validate_loaded_game(game)

    def test_saved_state_validation_rejects_excess_displacement_pieces(self):
        game = GameConfiguration(map_num=2, seed=124).create_game()
        opponent = game.players[1]
        route = game.selected_map.routes[0]
        route.posts[0].claim(opponent, "square")
        opponent.personal_supply_squares -= 1
        game.apply_action(0)
        game.displaced_player.total_pieces_to_place = 99

        with self.assertRaisesRegex(GameInvariantError, "impossible remaining-piece count"):
            validate_loaded_game(game)

    def test_displaced_piece_progress_reduces_maximum_remaining_count(self):
        game = GameConfiguration(map_num=2, seed=124).create_game()
        opponent = game.players[1]
        route = game.selected_map.routes[0]
        route.posts[0].claim(opponent, "square")
        opponent.personal_supply_squares -= 1
        game.apply_action(0)
        target = next(
            post
            for candidate_route in game.selected_map.routes
            for post in candidate_route.posts
            if post.owner is None and post.required_shape in (None, "square")
        )
        target.claim(opponent, "square")
        game.displaced_player.played_displaced_shape = True

        with self.assertRaisesRegex(GameInvariantError, "impossible remaining-piece count"):
            validate_loaded_game(game)


if __name__ == "__main__":
    unittest.main()
