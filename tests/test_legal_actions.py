import unittest

from ai.action_options import masking_out_invalid_actions
from game.action_codec import DEFAULT_ACTION_CODEC
from game.game_actions import refresh_displacement_targets
from game.game_runner import create_headless_game, legal_action_indices
from game.invariants import validate_game
from game.legal_actions import to_legacy_index
from game.structured_actions import (
    BonusMarkerInteraction,
    PlayerInteraction,
    SupplyInteraction,
)
from game.turn_state import TurnPhase
from map_data.map_attributes import BonusMarker


class LegalActionTests(unittest.TestCase):
    def assert_structured_matches_legacy(self, game):
        actions = game.get_legal_actions()
        self.assertEqual(len(actions), len(set(actions)))
        self.assertEqual(actions, game.get_legal_actions())
        ai_mask = game.ai_action_mask()
        self.assertEqual(len(ai_mask), 768)
        self.assertEqual(sum(ai_mask), len(actions))
        self.assertEqual(
            {DEFAULT_ACTION_CODEC.encode(action) for action in actions},
            {index for index, enabled in enumerate(ai_mask) if enabled},
        )
        self.assertEqual(
            {to_legacy_index(game, action) for action in actions},
            set(masking_out_invalid_actions(game).nonzero().flatten().tolist()),
        )
        if game.turn_phase != TurnPhase.GAME_OVER:
            self.assertTrue(actions)

    def test_fresh_game_has_legal_actions(self):
        game = create_headless_game(map_num=2, num_players=3, seed=124)
        actions = legal_action_indices(game)
        self.assertTrue(actions)
        self.assertTrue(all(0 <= action < 768 for action in actions))
        self.assertTrue(validate_game(game))
        self.assert_structured_matches_legacy(game)

    def test_supported_fresh_states_are_deterministic_and_complete(self):
        for map_num in (1, 2, 3):
            for players in (3, 4, 5):
                with self.subTest(map_num=map_num, players=players):
                    game = create_headless_game(map_num, players, seed=124)
                    self.assert_structured_matches_legacy(game)

    def test_terminal_state_has_no_legal_interactions(self):
        game = create_headless_game(2, 3, seed=124)
        game.game_end = True
        self.assertEqual(game.turn_phase, TurnPhase.GAME_OVER)
        self.assertEqual(game.get_legal_actions(), [])
        self.assertEqual(legal_action_indices(game), ())

    def test_ai_index_decodes_and_executes_through_engine(self):
        game = create_headless_game(2, 3, seed=124)
        index = legal_action_indices(game)[0]
        DEFAULT_ACTION_CODEC.decode(index)
        game.apply_ai_action(index)
        self.assertTrue(validate_game(game))

        with self.assertRaisesRegex(RuntimeError, "reserved"):
            game.apply_ai_action(767)

    def test_displacement_exposes_posts_optional_supply_and_finish(self):
        game = create_headless_game(2, 3, seed=124)
        opponent = game.players[1]
        route = next(
            route
            for route in game.selected_map.routes
            if {city.name for city in route.cities} == {"Malmo", "Visby"}
        )
        opponent.general_stock_squares = 1
        opponent.general_stock_circles = 1
        game.original_route_of_displacement = route
        game.waiting_for_displaced_player = True
        game.displaced_player.populate_displaced_player(game, opponent, "circle")
        refresh_displacement_targets(game)

        actions = game.get_legal_actions()
        self.assertTrue(any(isinstance(action, SupplyInteraction) for action in actions))
        self.assert_structured_matches_legacy(game)

        game.displaced_player.played_displaced_shape = True
        opponent.holding_pieces.clear()
        actions = game.get_legal_actions()
        self.assertIn(618, {to_legacy_index(game, action) for action in actions})

    def test_exchange_bonus_marker_has_both_structured_stages(self):
        game = create_headless_game(1, 3, seed=124)
        player, opponent = game.players[:2]
        player.bonus_markers = [BonusMarker("ExchangeBonusMarker", owner=player)]
        opponent.used_bonus_markers = [BonusMarker("Move3", owner=opponent)]

        activation = [
            action
            for action in game.get_legal_actions()
            if isinstance(action, BonusMarkerInteraction) and to_legacy_index(game, action) == 532
        ]
        self.assertEqual(len(activation), 1)
        game.apply_action(532)

        targets = [
            action for action in game.get_legal_actions() if isinstance(action, PlayerInteraction)
        ]
        self.assertEqual(targets, [PlayerInteraction(1)])
        game.apply_action(584)

        used_markers = [
            action
            for action in game.get_legal_actions()
            if isinstance(action, BonusMarkerInteraction)
        ]
        self.assertEqual(len(used_markers), 1)
        self.assertEqual(to_legacy_index(game, used_markers[0]), 528)
        self.assert_structured_matches_legacy(game)

    def test_exchange_last_opponent_and_marker_type_fit_reserved_family(self):
        game = create_headless_game(1, 5, seed=124)
        target = game.players[-1]
        target.used_bonus_markers = [BonusMarker("BlockTradeRoute", owner=target)]
        game.waiting_for_bm_exchange_bm = True
        game.exchange_target_player = target

        actions = [
            action
            for action in game.get_legal_actions()
            if isinstance(action, BonusMarkerInteraction)
        ]
        self.assertEqual(actions, [BonusMarkerInteraction(40)])
        self.assertEqual(DEFAULT_ACTION_CODEC.encode(actions[0]), 632)
        self.assertEqual(to_legacy_index(game, actions[0]), 534)


if __name__ == "__main__":
    unittest.main()
