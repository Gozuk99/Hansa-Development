import unittest
from pathlib import Path

from game.action_schema import (
    ACTION_RANGES,
    ACTION_SCHEMA_VERSION,
    ACTION_SPACE_SIZE,
    RESERVED,
    validate_action_schema,
)


class ActionSchemaTests(unittest.TestCase):
    EXPECTED = (
        ("position", 0, 352, "position", False),
        ("route", 352, 160, "route", False),
        ("city", 512, 46, "city", False),
        ("income", 558, 5, "income", False),
        ("exact_two", 563, 3, "exact_two", False),
        ("tribute_income", 566, 3, "tribute_income", False),
        ("bonus_marker", 569, 9, "bonus_marker", False),
        ("used_bonus_marker", 578, 32, "used_bonus_marker", False),
        ("tile", 610, 8, "tile", False),
        ("ability", 618, 5, "ability", False),
        ("control", 623, 3, "control", False),
        ("reserved", 626, 142, None, True),
    )

    def test_version_size_and_registry(self):
        self.assertEqual(ACTION_SCHEMA_VERSION, 2)
        self.assertEqual(ACTION_SPACE_SIZE, 768)
        self.assertEqual(
            tuple(
                (
                    item.name,
                    item.start,
                    item.capacity,
                    item.family,
                    item.reserved,
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

    def test_exact_active_and_reserved_capacity(self):
        active = sum(item.capacity for item in ACTION_RANGES if not item.reserved)
        reserved = sum(item.capacity for item in ACTION_RANGES if item.reserved)
        self.assertEqual(active, 626)
        self.assertEqual(reserved, 142)
        self.assertEqual((RESERVED.start, RESERVED.stop), (626, 768))

    def test_documented_allocation_matches_registry(self):
        document = (Path(__file__).resolve().parents[1] / "docs" / "action-schema-v2.md").read_text(
            encoding="utf-8"
        )
        for name, start, capacity, _family, _reserved in self.EXPECTED:
            end = start + capacity - 1
            with self.subTest(name=name):
                self.assertIn(f"| `{name.upper()}` | `{start}–{end}` | {capacity} |", document)


if __name__ == "__main__":
    unittest.main()
