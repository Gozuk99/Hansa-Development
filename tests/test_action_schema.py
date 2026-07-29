import unittest
from pathlib import Path

from game.action_schema import (
    ABILITY_UPGRADE,
    ACTION_RANGES,
    ACTION_SCHEMA_VERSION,
    ACTION_SPACE_SIZE,
    CITY_TARGET,
    OFFICE_PAIR,
    PLAYER_TARGET,
    POST_TRADER,
    ROUTE_SELECT,
    validate_action_schema,
)
from game.game_runner import create_headless_game


class ActionSchemaTests(unittest.TestCase):
    EXPECTED_REGISTRY = (
        ("post_trader", 0, 121, "SelectPost", False),
        ("post_merchant", 121, 121, "SelectPost", False),
        ("post_reserved", 242, 14, None, True),
        ("route_select", 256, 40, "SelectRoute", False),
        ("route_outcome", 296, 5, "SelectRouteOutcome", False),
        ("route_endpoint", 301, 2, "SelectRouteEndpoint", False),
        ("route_upgrade_slot", 303, 2, "SelectCityUpgradeSlot", False),
        ("prestige_value", 305, 4, "SelectPrestigeValue", False),
        ("piece_shape", 309, 2, "SelectPieceShape", False),
        ("route_reserved", 311, 265, None, True),
        ("income_merchant_count", 576, 5, "SelectIncome", False),
        ("exact_two_merchant_count", 581, 3, "SelectTwoPieceMix", False),
        ("tribute_merchant_count", 584, 3, "SelectTributeIncome", False),
        ("piece_choice_reserved", 587, 21, None, True),
        ("bonus_marker_activate", 608, 9, "ActivateBonusMarker", False),
        ("bonus_marker_take_used", 617, 8, "SelectUsedBonusMarker", False),
        ("bonus_marker_reserved", 625, 15, None, True),
        ("tile_buy", 640, 6, "BuyEmperorsFavour", False),
        ("tile_payment", 646, 8, "SelectBonusMarkerPayment", False),
        ("income_favour_response", 654, 3, "RespondToIncomeFavour", False),
        ("tile_reserved", 657, 15, None, True),
        ("player_target", 672, 5, "SelectPlayer", False),
        ("city_target", 677, 30, "SelectCity", False),
        ("office_pair", 707, 7, "SelectOfficePair", False),
        ("ability_upgrade", 714, 5, "SelectAbility", False),
        ("choice_reserved", 719, 1, None, True),
        ("displacement_source", 720, 3, "SelectDisplacementSource", False),
        ("displacement_piece_kind", 723, 2, "SelectDisplacementPiece", False),
        ("displacement_reserved", 725, 27, None, True),
        ("finish_move_pickup", 752, 1, "FinishMovePickup", False),
        ("finish_displacement", 753, 1, "FinishDisplacement", False),
        (
            "decline_displacement_optional",
            754,
            1,
            "DeclineDisplacementOptionalPieces",
            False,
        ),
        ("end_turn", 755, 1, "EndTurn", False),
        ("forgo_bonus_marker", 756, 1, "ForgoBonusMarker", False),
        (
            "confirm_bonus_marker_replacement",
            757,
            1,
            "ConfirmBonusMarkerReplacement",
            False,
        ),
        ("control_reserved", 758, 10, None, True),
    )

    def test_schema_version_and_size(self):
        self.assertEqual(ACTION_SCHEMA_VERSION, 1)
        self.assertEqual(ACTION_SPACE_SIZE, 768)

    def test_ranges_are_contiguous_non_overlapping_and_complete(self):
        validate_action_schema()
        indices = [
            index
            for action_range in ACTION_RANGES
            for index in range(action_range.start, action_range.stop)
        ]
        self.assertEqual(indices, list(range(ACTION_SPACE_SIZE)))

    def test_active_ranges_have_one_semantic_family(self):
        for action_range in ACTION_RANGES:
            with self.subTest(action_range=action_range.name):
                if action_range.reserved:
                    self.assertIsNone(action_range.structured_action)
                else:
                    self.assertIsNotNone(action_range.structured_action)

    def test_registry_matches_authoritative_allocation(self):
        actual = tuple(
            (
                action_range.name,
                action_range.start,
                action_range.capacity,
                action_range.structured_action,
                action_range.reserved,
            )
            for action_range in ACTION_RANGES
        )
        self.assertEqual(actual, self.EXPECTED_REGISTRY)
        self.assertEqual(
            sum(
                action_range.capacity for action_range in ACTION_RANGES if not action_range.reserved
            ),
            400,
        )

    def test_documented_allocation_matches_registry(self):
        document = (Path(__file__).resolve().parents[1] / "docs" / "action-schema-v1.md").read_text(
            encoding="utf-8"
        )
        for name, start, capacity, structured_action, reserved in self.EXPECTED_REGISTRY:
            with self.subTest(action_range=name):
                end = start + capacity - 1
                displayed_range = str(start) if capacity == 1 else f"{start}–{end}"
                status = "Reserved" if reserved else "Active"
                action = "—" if reserved else f"`{structured_action}`"
                row = (
                    f"| `{name.upper()}` | `{displayed_range}` | {capacity} | {status} | {action} |"
                )
                self.assertIn(row, document)

    def test_reserved_ranges_are_explicit(self):
        reserved_indices = {
            index
            for action_range in ACTION_RANGES
            if action_range.reserved
            for index in range(action_range.start, action_range.stop)
        }
        expected_reserved_indices = set(range(242, 256))
        expected_reserved_indices.update(range(311, 576))
        expected_reserved_indices.update(range(587, 608))
        expected_reserved_indices.update(range(625, 640))
        expected_reserved_indices.update(range(657, 672))
        expected_reserved_indices.add(719)
        expected_reserved_indices.update(range(725, 752))
        expected_reserved_indices.update(range(758, 768))
        self.assertEqual(reserved_indices, expected_reserved_indices)
        self.assertEqual(len(reserved_indices), 368)

    def test_capacities_cover_every_supported_map_configuration(self):
        for map_num in (1, 2, 3):
            for num_players in (3, 4, 5):
                with self.subTest(map_num=map_num, num_players=num_players):
                    game = create_headless_game(map_num, num_players, seed=1)
                    selected_map = game.selected_map
                    post_count = sum(len(route.posts) for route in selected_map.routes)
                    max_office_pairs = max(len(city.offices) - 1 for city in selected_map.cities)
                    self.assertLessEqual(post_count, POST_TRADER.capacity)
                    self.assertLessEqual(len(selected_map.routes), ROUTE_SELECT.capacity)
                    self.assertLessEqual(len(selected_map.cities), CITY_TARGET.capacity)
                    self.assertLessEqual(len(game.players), PLAYER_TARGET.capacity)
                    self.assertLessEqual(len(selected_map.upgrade_cities), ABILITY_UPGRADE.capacity)
                    self.assertLessEqual(max_office_pairs, OFFICE_PAIR.capacity)


if __name__ == "__main__":
    unittest.main()
