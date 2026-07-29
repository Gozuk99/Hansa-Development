"""Authoritative index allocation for Hansa Teutonica action schema version 1.

Milestone 2 defines and validates this registry without activating it in the
game engine. Later milestones will add structured actions and the codec that
uses these ranges.
"""

from dataclasses import dataclass


ACTION_SPACE_SIZE = 768
ACTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActionRange:
    """One contiguous range with a single stable semantic family."""

    name: str
    start: int
    capacity: int
    structured_action: str | None
    reserved: bool = False

    @property
    def stop(self) -> int:
        return self.start + self.capacity

    @property
    def end(self) -> int:
        return self.stop - 1


POST_TRADER = ActionRange("post_trader", 0, 121, "SelectPost")
POST_MERCHANT = ActionRange("post_merchant", 121, 121, "SelectPost")
POST_RESERVED = ActionRange("post_reserved", 242, 14, None, reserved=True)

ROUTE_SELECT = ActionRange("route_select", 256, 40, "SelectRoute")
ROUTE_OUTCOME = ActionRange("route_outcome", 296, 5, "SelectRouteOutcome")
ROUTE_ENDPOINT = ActionRange("route_endpoint", 301, 2, "SelectRouteEndpoint")
ROUTE_UPGRADE_SLOT = ActionRange("route_upgrade_slot", 303, 2, "SelectCityUpgradeSlot")
PRESTIGE_VALUE = ActionRange("prestige_value", 305, 4, "SelectPrestigeValue")
PIECE_SHAPE = ActionRange("piece_shape", 309, 2, "SelectPieceShape")
ROUTE_RESERVED = ActionRange("route_reserved", 311, 265, None, reserved=True)

INCOME_MERCHANT_COUNT = ActionRange("income_merchant_count", 576, 5, "SelectIncome")
EXACT_TWO_MERCHANT_COUNT = ActionRange("exact_two_merchant_count", 581, 3, "SelectTwoPieceMix")
TRIBUTE_MERCHANT_COUNT = ActionRange("tribute_merchant_count", 584, 3, "SelectTributeIncome")
PIECE_CHOICE_RESERVED = ActionRange("piece_choice_reserved", 587, 21, None, reserved=True)

BONUS_MARKER_ACTIVATE = ActionRange("bonus_marker_activate", 608, 9, "ActivateBonusMarker")
BONUS_MARKER_TAKE_USED = ActionRange("bonus_marker_take_used", 617, 8, "SelectUsedBonusMarker")
BONUS_MARKER_RESERVED = ActionRange("bonus_marker_reserved", 625, 15, None, reserved=True)

TILE_BUY = ActionRange("tile_buy", 640, 6, "BuyEmperorsFavour")
TILE_PAYMENT = ActionRange("tile_payment", 646, 8, "SelectBonusMarkerPayment")
INCOME_FAVOUR_RESPONSE = ActionRange("income_favour_response", 654, 3, "RespondToIncomeFavour")
TILE_RESERVED = ActionRange("tile_reserved", 657, 15, None, reserved=True)

PLAYER_TARGET = ActionRange("player_target", 672, 5, "SelectPlayer")
CITY_TARGET = ActionRange("city_target", 677, 30, "SelectCity")
OFFICE_PAIR = ActionRange("office_pair", 707, 7, "SelectOfficePair")
ABILITY_UPGRADE = ActionRange("ability_upgrade", 714, 5, "SelectAbility")
CHOICE_RESERVED = ActionRange("choice_reserved", 719, 1, None, reserved=True)

DISPLACEMENT_SOURCE = ActionRange("displacement_source", 720, 3, "SelectDisplacementSource")
DISPLACEMENT_PIECE_KIND = ActionRange("displacement_piece_kind", 723, 2, "SelectDisplacementPiece")
DISPLACEMENT_RESERVED = ActionRange("displacement_reserved", 725, 27, None, reserved=True)

FINISH_MOVE_PICKUP = ActionRange("finish_move_pickup", 752, 1, "FinishMovePickup")
FINISH_DISPLACEMENT = ActionRange("finish_displacement", 753, 1, "FinishDisplacement")
DECLINE_DISPLACEMENT_OPTIONAL = ActionRange(
    "decline_displacement_optional", 754, 1, "DeclineDisplacementOptionalPieces"
)
END_TURN = ActionRange("end_turn", 755, 1, "EndTurn")
FORGO_BONUS_MARKER = ActionRange("forgo_bonus_marker", 756, 1, "ForgoBonusMarker")
CONFIRM_BONUS_MARKER_REPLACEMENT = ActionRange(
    "confirm_bonus_marker_replacement",
    757,
    1,
    "ConfirmBonusMarkerReplacement",
)
CONTROL_RESERVED = ActionRange("control_reserved", 758, 10, None, reserved=True)


ACTION_RANGES = (
    POST_TRADER,
    POST_MERCHANT,
    POST_RESERVED,
    ROUTE_SELECT,
    ROUTE_OUTCOME,
    ROUTE_ENDPOINT,
    ROUTE_UPGRADE_SLOT,
    PRESTIGE_VALUE,
    PIECE_SHAPE,
    ROUTE_RESERVED,
    INCOME_MERCHANT_COUNT,
    EXACT_TWO_MERCHANT_COUNT,
    TRIBUTE_MERCHANT_COUNT,
    PIECE_CHOICE_RESERVED,
    BONUS_MARKER_ACTIVATE,
    BONUS_MARKER_TAKE_USED,
    BONUS_MARKER_RESERVED,
    TILE_BUY,
    TILE_PAYMENT,
    INCOME_FAVOUR_RESPONSE,
    TILE_RESERVED,
    PLAYER_TARGET,
    CITY_TARGET,
    OFFICE_PAIR,
    ABILITY_UPGRADE,
    CHOICE_RESERVED,
    DISPLACEMENT_SOURCE,
    DISPLACEMENT_PIECE_KIND,
    DISPLACEMENT_RESERVED,
    FINISH_MOVE_PICKUP,
    FINISH_DISPLACEMENT,
    DECLINE_DISPLACEMENT_OPTIONAL,
    END_TURN,
    FORGO_BONUS_MARKER,
    CONFIRM_BONUS_MARKER_REPLACEMENT,
    CONTROL_RESERVED,
)


def validate_action_schema() -> None:
    """Raise ValueError when the version-1 registry is incomplete or overlaps."""

    expected_start = 0
    names = set()
    for action_range in ACTION_RANGES:
        if action_range.name in names:
            raise ValueError(f"Duplicate action range name: {action_range.name}")
        names.add(action_range.name)
        if action_range.capacity <= 0:
            raise ValueError(f"{action_range.name} must have positive capacity")
        if action_range.start != expected_start:
            raise ValueError(
                f"Expected {action_range.name} to start at {expected_start}, "
                f"got {action_range.start}"
            )
        if action_range.reserved != (action_range.structured_action is None):
            raise ValueError(
                f"{action_range.name} must be either reserved or associated "
                "with one structured action family"
            )
        expected_start = action_range.stop

    if expected_start != ACTION_SPACE_SIZE:
        raise ValueError(f"Action ranges end at {expected_start}, expected {ACTION_SPACE_SIZE}")


validate_action_schema()
