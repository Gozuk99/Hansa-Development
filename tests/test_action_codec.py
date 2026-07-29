import unittest
from dataclasses import replace

from game.action_codec import (
    DEFAULT_ACTION_CODEC,
    DEFAULT_ACTION_FAMILIES,
    ActionCodec,
    ActionCodecContext,
    ActionCodecValidationError,
    ActionFamily,
    ActionFamilyCapacityError,
    ActionIndexOutOfRangeError,
    DuplicateActionError,
    InactiveActionIndexError,
    ReservedActionIndexError,
    UnknownActionError,
)
from game.action_schema import ACTION_SPACE_SIZE, POSITION
from game.structured_actions import (
    ClaimRouteOffice,
    CompleteRouteForPoints,
    EndTurn,
    GameAction,
    PieceShape,
    PlaceDisplacedPiece,
    PlaceFromPersonalSupply,
    PlaceOptionalDisplacementPiece,
    SwapAdjacentOffices,
)


class UnsupportedAction(GameAction):
    pass


class ActionCodecTests(unittest.TestCase):
    def setUp(self):
        self.actions = (
            PlaceFromPersonalSupply("post-a", PieceShape.TRADER),
            CompleteRouteForPoints("route-a"),
            ClaimRouteOffice("route-a", "city-b"),
            SwapAdjacentOffices("city-a", "office-1", "office-2"),
            EndTurn(),
        )
        self.context = ActionCodecContext.from_actions(7, self.actions)

    def test_every_legal_action_round_trips(self):
        decision = DEFAULT_ACTION_CODEC.build_decision(self.context)
        self.assertEqual(sum(decision.mask), len(self.actions))
        for action in self.actions:
            with self.subTest(action=action):
                index = DEFAULT_ACTION_CODEC.encode(action, self.context)
                self.assertTrue(decision.mask[index])
                self.assertEqual(DEFAULT_ACTION_CODEC.decode(index, self.context), action)
                self.assertTrue(DEFAULT_ACTION_CODEC.describe(index, self.context))

    def test_mapping_is_deterministic_for_action_order(self):
        reversed_context = ActionCodecContext.from_actions(
            7,
            reversed(self.actions),
            self.actions,
        )
        first = DEFAULT_ACTION_CODEC.build_decision(self.context)
        second = DEFAULT_ACTION_CODEC.build_decision(reversed_context)
        self.assertEqual(first.index_by_action, second.index_by_action)
        self.assertEqual(first.action_by_index, second.action_by_index)

    def test_index_does_not_shift_when_catalogued_action_becomes_illegal(self):
        full = ActionCodecContext.from_actions(7, self.actions, self.actions)
        reduced = ActionCodecContext.from_actions(8, self.actions[1:], self.actions)
        first = DEFAULT_ACTION_CODEC.build_decision(full)
        second = DEFAULT_ACTION_CODEC.build_decision(reduced)
        for action in self.actions[1:]:
            self.assertEqual(first.index_by_action[action], second.index_by_action[action])

    def test_complete_displacement_choices_remain_distinct(self):
        destination = "post-z"
        actions = (
            PlaceDisplacedPiece(destination),
            PlaceOptionalDisplacementPiece(PieceShape.MERCHANT, destination),
        )
        context = ActionCodecContext.from_actions(8, actions)
        indices = {DEFAULT_ACTION_CODEC.encode(action, context) for action in actions}
        self.assertEqual(len(indices), 2)

    def test_reserved_and_inactive_indices_fail_distinctly(self):
        with self.assertRaises(ReservedActionIndexError):
            DEFAULT_ACTION_CODEC.decode(626, self.context)
        with self.assertRaises(InactiveActionIndexError):
            DEFAULT_ACTION_CODEC.decode(POSITION.end, self.context)

    def test_invalid_indices_fail(self):
        for index in (-1, ACTION_SPACE_SIZE, True, 1.5):
            with self.subTest(index=index):
                with self.assertRaises(ActionIndexOutOfRangeError):
                    DEFAULT_ACTION_CODEC.decode(index, self.context)

    def test_unknown_and_illegal_actions_fail(self):
        with self.assertRaises(UnknownActionError):
            DEFAULT_ACTION_CODEC.encode(UnsupportedAction(), self.context)
        with self.assertRaises(UnknownActionError):
            DEFAULT_ACTION_CODEC.encode(
                CompleteRouteForPoints("other-route"),
                self.context,
            )

    def test_duplicate_legal_actions_fail(self):
        action = EndTurn()
        context = ActionCodecContext.from_actions(1, (action, action))
        with self.assertRaises(DuplicateActionError):
            DEFAULT_ACTION_CODEC.build_decision(context)

    def test_legal_action_must_appear_in_catalogue(self):
        context = ActionCodecContext.from_actions(
            1,
            (EndTurn(),),
            (CompleteRouteForPoints("route-a"),),
        )
        with self.assertRaises(UnknownActionError):
            DEFAULT_ACTION_CODEC.build_decision(context)

    def test_family_capacity_is_enforced(self):
        actions = tuple(PlaceDisplacedPiece(f"post-{index}") for index in range(353))
        context = ActionCodecContext.from_actions(1, actions)
        with self.assertRaises(ActionFamilyCapacityError):
            DEFAULT_ACTION_CODEC.build_decision(context)

    def test_duplicate_range_and_type_registration_fail(self):
        duplicate_range = replace(
            DEFAULT_ACTION_FAMILIES[0],
            name="duplicate",
        )
        with self.assertRaises(ActionCodecValidationError):
            ActionCodec((*DEFAULT_ACTION_FAMILIES, duplicate_range))

        duplicate_type = ActionFamily(
            DEFAULT_ACTION_FAMILIES[1].name,
            DEFAULT_ACTION_FAMILIES[1].action_range,
            (PlaceFromPersonalSupply,),
        )
        with self.assertRaises(ActionCodecValidationError):
            ActionCodec(
                (
                    DEFAULT_ACTION_FAMILIES[0],
                    duplicate_type,
                    *DEFAULT_ACTION_FAMILIES[2:],
                )
            )


if __name__ == "__main__":
    unittest.main()
