import hashlib
import json
import sys
import unittest

import torch

from ai.observation_encoder import ObservationEncoder
from game.game_actions import refresh_displacement_targets
from game.game_runner import create_headless_game
from game.structured_actions import SupplyInteraction, TileInteraction
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


class ObservationEncoderTests(unittest.TestCase):
    def game(self, map_num=1, players=3, **options):
        return create_headless_game(map_num, players, seed=124, **options)

    def observation(self, game):
        return ObservationEncoder().build(game)

    def test_structural_cache_preserves_observation_contract_after_dynamic_changes(self):
        expected = {
            1: (
                "83f10dacc5381d20972c260b735da3299e3f498883bbcf43efecd8b32de7b233",
                "b3efcebca0a0ab461649655916945e524097a38bf7d921fa53365a793c53796c",
            ),
            2: (
                "b97e894bf6b3c295a250e35e4a1bc90e34ee98e3e7d2773b6b885d00e6f38e56",
                "203fb75f0dd734e1ba290a4e603f481383aa8e2ddcd5cc5cbdda50b41bd47791",
            ),
            3: (
                "facc801e38406057ede4e8331fdee335ba36b5faf61435854dcfcce29ecda009",
                "d513289b9d55fcaac4e88fc270385c4f42e714c9803ab2131df73a3089fb8cab",
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
        self.assertEqual(workflow_start + encoder.WORKFLOW_SIZE, encoder.FEATURE_SIZE)

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
