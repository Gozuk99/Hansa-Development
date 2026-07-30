import unittest
from dataclasses import replace

from game.action_codec import (
    DEFAULT_ACTION_CODEC,
    DEFAULT_INTERACTION_FAMILIES,
    ActionCodec,
    ActionCodecValidationError,
    ActionIndexOutOfRangeError,
    InteractionFamily,
    InvalidInteractionError,
    ReservedActionIndexError,
    UnknownActionError,
)
from game.action_schema import ACTION_SPACE_SIZE, POST
from game.structured_actions import (
    ControlInteraction,
    GameAction,
    PieceShape,
    PostInteraction,
    RouteInteraction,
)


class UnsupportedAction(GameAction):
    pass


class ActionCodecTests(unittest.TestCase):
    def test_every_active_interaction_round_trips(self):
        for family in DEFAULT_INTERACTION_FAMILIES:
            for index in range(
                family.action_range.start,
                family.action_range.active_stop,
            ):
                with self.subTest(index=index):
                    action = DEFAULT_ACTION_CODEC.decode(index)
                    self.assertEqual(DEFAULT_ACTION_CODEC.encode(action), index)
                    self.assertTrue(DEFAULT_ACTION_CODEC.describe(index))

    def test_post_shape_layout_matches_original_mask(self):
        self.assertEqual(
            DEFAULT_ACTION_CODEC.encode(PostInteraction(0, PieceShape.TRADER)),
            0,
        )
        self.assertEqual(
            DEFAULT_ACTION_CODEC.encode(PostInteraction(120, PieceShape.TRADER)),
            120,
        )
        self.assertEqual(
            DEFAULT_ACTION_CODEC.encode(PostInteraction(0, PieceShape.MERCHANT)),
            121,
        )
        self.assertEqual(
            DEFAULT_ACTION_CODEC.encode(PostInteraction(120, PieceShape.MERCHANT)),
            241,
        )

    def test_route_layout_preserves_distinct_physical_interactions(self):
        route_start = 256
        self.assertEqual(
            DEFAULT_ACTION_CODEC.encode(RouteInteraction(3, 0)),
            route_start + 3,
        )
        self.assertEqual(
            DEFAULT_ACTION_CODEC.encode(RouteInteraction(3, 1)),
            route_start + 40 + 3 * 2,
        )
        self.assertEqual(
            DEFAULT_ACTION_CODEC.encode(RouteInteraction(3, 2)),
            route_start + 40 + 3 * 2 + 1,
        )
        self.assertEqual(
            DEFAULT_ACTION_CODEC.encode(RouteInteraction(3, 3)),
            route_start + 120 + 3 * 4,
        )
        self.assertEqual(
            DEFAULT_ACTION_CODEC.encode(RouteInteraction(3, 6)),
            route_start + 120 + 3 * 4 + 3,
        )

    def test_finish_and_end_turn_are_distinct_controls(self):
        finish = DEFAULT_ACTION_CODEC.encode(ControlInteraction(0))
        end_turn = DEFAULT_ACTION_CODEC.encode(ControlInteraction(1))
        self.assertNotEqual(finish, end_turn)

    def test_padding_is_reserved(self):
        with self.assertRaises(ReservedActionIndexError):
            DEFAULT_ACTION_CODEC.decode(POST.active_stop)
        with self.assertRaises(ReservedActionIndexError):
            DEFAULT_ACTION_CODEC.decode(ACTION_SPACE_SIZE - 1)

    def test_mask_enables_exactly_the_encoded_actions(self):
        actions = (
            PostInteraction(4, PieceShape.TRADER),
            RouteInteraction(2, 1),
            ControlInteraction(1),
        )
        mask = DEFAULT_ACTION_CODEC.create_mask(actions)
        self.assertEqual(len(mask), ACTION_SPACE_SIZE)
        self.assertEqual(sum(mask), len(actions))
        self.assertEqual(
            {index for index, enabled in enumerate(mask) if enabled},
            {DEFAULT_ACTION_CODEC.encode(action) for action in actions},
        )

    def test_invalid_indices_fail(self):
        for index in (-1, ACTION_SPACE_SIZE, True, 1.5):
            with self.subTest(index=index):
                with self.assertRaises(ActionIndexOutOfRangeError):
                    DEFAULT_ACTION_CODEC.decode(index)

    def test_unknown_and_invalid_interactions_fail(self):
        with self.assertRaises(UnknownActionError):
            DEFAULT_ACTION_CODEC.encode(UnsupportedAction())
        with self.assertRaises(InvalidInteractionError):
            DEFAULT_ACTION_CODEC.encode(PostInteraction(121, PieceShape.TRADER))

    def test_duplicate_range_and_type_registration_fail(self):
        duplicate_range = replace(
            DEFAULT_INTERACTION_FAMILIES[0],
            name="duplicate",
        )
        with self.assertRaises(ActionCodecValidationError):
            ActionCodec((*DEFAULT_INTERACTION_FAMILIES, duplicate_range))

        original = DEFAULT_INTERACTION_FAMILIES[2]
        duplicate_type = InteractionFamily(
            original.name,
            DEFAULT_INTERACTION_FAMILIES[1].action_type,
            original.action_range,
            original.encode_local,
            original.decode_local,
            original.validate_action,
            original.describe_action,
        )
        with self.assertRaises(ActionCodecValidationError):
            ActionCodec(
                (
                    DEFAULT_INTERACTION_FAMILIES[0],
                    DEFAULT_INTERACTION_FAMILIES[1],
                    duplicate_type,
                    *DEFAULT_INTERACTION_FAMILIES[3:],
                )
            )


if __name__ == "__main__":
    unittest.main()
