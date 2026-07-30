import unittest
from pathlib import Path

from game.action_schema import (
    ACTION_RANGES,
    ACTION_SCHEMA_VERSION,
    ACTION_SPACE_SIZE,
    validate_action_schema,
)
from map_data.constants import OUTPUT_SIZE


class ActionSchemaTests(unittest.TestCase):
    EXPECTED = (
        ("post", 0, 256, 242, "PostInteraction"),
        ("route", 256, 320, 280, "RouteInteraction"),
        ("income", 576, 16, 5, "IncomeInteraction"),
        ("bonus_marker", 592, 48, 41, "BonusMarkerInteraction"),
        ("tile", 640, 16, 8, "TileInteraction"),
        ("city", 656, 64, 52, "CityInteraction"),
        ("ability", 720, 8, 5, "AbilityInteraction"),
        ("supply", 728, 2, 1, "SupplyInteraction"),
        ("player", 730, 6, 5, "PlayerInteraction"),
        ("control", 736, 8, 2, "ControlInteraction"),
        ("expansion", 744, 24, 0, None),
    )

    def test_version_size_and_registry(self):
        self.assertEqual(ACTION_SCHEMA_VERSION, 2)
        self.assertEqual(ACTION_SPACE_SIZE, 768)
        self.assertEqual(OUTPUT_SIZE, ACTION_SPACE_SIZE)
        self.assertEqual(
            tuple(
                (
                    item.name,
                    item.start,
                    item.capacity,
                    item.active_capacity,
                    item.interaction_type,
                )
                for item in ACTION_RANGES
            ),
            self.EXPECTED,
        )

    def test_ranges_cover_schema_once(self):
        validate_action_schema()
        indices = [
            index
            for action_range in ACTION_RANGES
            for index in range(action_range.start, action_range.stop)
        ]
        self.assertEqual(indices, list(range(ACTION_SPACE_SIZE)))

    def test_padding_is_distributed_inside_families(self):
        active = sum(item.active_capacity for item in ACTION_RANGES)
        reserved = sum(item.reserved_capacity for item in ACTION_RANGES)
        self.assertEqual(active, 641)
        self.assertEqual(reserved, 127)
        for item in ACTION_RANGES[:-1]:
            with self.subTest(family=item.name):
                self.assertGreater(item.reserved_capacity, 0)

    def test_documented_allocation_matches_registry(self):
        document = (Path(__file__).resolve().parents[1] / "docs" / "action-schema-v2.md").read_text(
            encoding="utf-8"
        )
        for name, start, capacity, active, _interaction in self.EXPECTED:
            end = start + capacity - 1
            with self.subTest(name=name):
                self.assertIn(
                    f"| `{name.upper()}` | `{start}–{end}` | {capacity} | "
                    f"{active} | {capacity - active} |",
                    document,
                )


if __name__ == "__main__":
    unittest.main()
