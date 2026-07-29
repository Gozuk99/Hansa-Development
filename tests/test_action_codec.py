import unittest
from dataclasses import replace

from game.action_codec import (
    DEFAULT_ACTION_CODEC,
    DEFAULT_ACTION_FAMILIES,
    ActionCodec,
    ActionCodecValidationError,
    ActionIndexOutOfRangeError,
    InvalidStructuredActionError,
    ReservedActionIndexError,
    UnknownActionError,
)
from game.action_schema import ACTION_RANGES, ACTION_SPACE_SIZE
from game.structured_actions import (
    ActivateBonusMarker,
    BonusMarkerType,
    GameAction,
    PieceShape,
    SelectIncome,
    SelectPost,
    SelectUsedBonusMarker,
)


class UnsupportedAction(GameAction):
    pass


class ActionCodecTests(unittest.TestCase):
    def test_every_active_index_round_trips_and_has_description(self):
        active_indices = {
            index
            for action_range in ACTION_RANGES
            if not action_range.reserved
            for index in range(action_range.start, action_range.stop)
        }
        self.assertEqual(len(active_indices), 400)
        for index in sorted(active_indices):
            with self.subTest(index=index):
                action = DEFAULT_ACTION_CODEC.decode(index)
                self.assertEqual(DEFAULT_ACTION_CODEC.encode(action), index)
                self.assertTrue(DEFAULT_ACTION_CODEC.describe(index))

    def test_every_reserved_index_is_identified_and_rejected(self):
        reserved_indices = {
            index
            for action_range in ACTION_RANGES
            if action_range.reserved
            for index in range(action_range.start, action_range.stop)
        }
        self.assertEqual(len(reserved_indices), 368)
        for index in sorted(reserved_indices):
            with self.subTest(index=index):
                self.assertTrue(DEFAULT_ACTION_CODEC.is_reserved(index))
                with self.assertRaisesRegex(ReservedActionIndexError, "reserved"):
                    DEFAULT_ACTION_CODEC.decode(index)
                with self.assertRaises(ReservedActionIndexError):
                    DEFAULT_ACTION_CODEC.describe(index)

    def test_known_actions_encode_to_documented_indices(self):
        cases = (
            (SelectPost(0, PieceShape.TRADER), 0),
            (SelectPost(120, PieceShape.MERCHANT), 241),
            (SelectIncome(4), 580),
            (
                ActivateBonusMarker(BonusMarkerType.PLACE_ADJACENT),
                616,
            ),
            (
                SelectUsedBonusMarker(BonusMarkerType.BLOCK_TRADE_ROUTE),
                624,
            ),
        )
        for action, expected_index in cases:
            with self.subTest(action=action):
                self.assertEqual(DEFAULT_ACTION_CODEC.encode(action), expected_index)
                self.assertEqual(DEFAULT_ACTION_CODEC.decode(expected_index), action)

    def test_out_of_range_and_non_integer_indices_fail_clearly(self):
        for index in (-1, ACTION_SPACE_SIZE, True, 1.5, "1"):
            with self.subTest(index=index):
                with self.assertRaises(ActionIndexOutOfRangeError):
                    DEFAULT_ACTION_CODEC.decode(index)
                with self.assertRaises(ActionIndexOutOfRangeError):
                    DEFAULT_ACTION_CODEC.is_reserved(index)

    def test_unknown_action_type_fails_clearly(self):
        with self.assertRaisesRegex(UnknownActionError, "UnsupportedAction"):
            DEFAULT_ACTION_CODEC.encode(UnsupportedAction())

    def test_invalid_structured_action_fields_fail_clearly(self):
        invalid_actions = (
            SelectPost(-1, PieceShape.TRADER),
            SelectPost(0, "trader"),
            SelectIncome(5),
            SelectIncome(True),
            SelectUsedBonusMarker(BonusMarkerType.PLACE_ADJACENT),
        )
        for action in invalid_actions:
            with self.subTest(action=action):
                with self.assertRaises(InvalidStructuredActionError):
                    DEFAULT_ACTION_CODEC.encode(action)

    def test_duplicate_range_registration_fails_validation(self):
        duplicate = replace(DEFAULT_ACTION_FAMILIES[0], name="duplicate_post_trader")
        with self.assertRaisesRegex(ActionCodecValidationError, "overlaps"):
            ActionCodec((*DEFAULT_ACTION_FAMILIES, duplicate))

    def test_missing_registration_fails_validation(self):
        with self.assertRaisesRegex(ActionCodecValidationError, "missing"):
            ActionCodec(DEFAULT_ACTION_FAMILIES[:-1])

    def test_schema_action_type_mismatch_fails_validation(self):
        mismatched = replace(
            DEFAULT_ACTION_FAMILIES[0],
            action_type=ActivateBonusMarker,
        )
        with self.assertRaisesRegex(ActionCodecValidationError, "schema requires"):
            ActionCodec((mismatched, *DEFAULT_ACTION_FAMILIES[1:]))


if __name__ == "__main__":
    unittest.main()
