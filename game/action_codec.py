"""Data-driven codec for structured schema-version-1 actions."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

from game.action_schema import (
    ABILITY_UPGRADE,
    ACTION_RANGES,
    ACTION_SPACE_SIZE,
    BONUS_MARKER_ACTIVATE,
    BONUS_MARKER_TAKE_USED,
    CITY_TARGET,
    CONFIRM_BONUS_MARKER_REPLACEMENT,
    DECLINE_DISPLACEMENT_OPTIONAL,
    DISPLACEMENT_PIECE_KIND,
    DISPLACEMENT_SOURCE,
    END_TURN,
    EXACT_TWO_MERCHANT_COUNT,
    FINISH_DISPLACEMENT,
    FINISH_MOVE_PICKUP,
    FORGO_BONUS_MARKER,
    INCOME_FAVOUR_RESPONSE,
    INCOME_MERCHANT_COUNT,
    OFFICE_PAIR,
    PIECE_SHAPE,
    PLAYER_TARGET,
    POST_MERCHANT,
    POST_TRADER,
    PRESTIGE_VALUE,
    ROUTE_ENDPOINT,
    ROUTE_OUTCOME,
    ROUTE_SELECT,
    ROUTE_UPGRADE_SLOT,
    TILE_BUY,
    TILE_PAYMENT,
    TRIBUTE_MERCHANT_COUNT,
    ActionRange,
)
from game.structured_actions import (
    ActivateBonusMarker,
    BonusMarkerType,
    BuyEmperorsFavour,
    ConfirmBonusMarkerReplacement,
    DeclineDisplacementOptionalPieces,
    DisplacementPieceKind,
    DisplacementSource,
    EmperorsFavourType,
    EndTurn,
    FinishDisplacement,
    FinishMovePickup,
    ForgoBonusMarker,
    GameAction,
    IncomeFavourChoice,
    PieceShape,
    RespondToIncomeFavour,
    RouteOutcome,
    SelectAbility,
    SelectBonusMarkerPayment,
    SelectCity,
    SelectCityUpgradeSlot,
    SelectDisplacementPiece,
    SelectDisplacementSource,
    SelectIncome,
    SelectOfficePair,
    SelectPieceShape,
    SelectPlayer,
    SelectPost,
    SelectPrestigeValue,
    SelectRoute,
    SelectRouteEndpoint,
    SelectRouteOutcome,
    SelectTributeIncome,
    SelectTwoPieceMix,
    SelectUsedBonusMarker,
)


class ActionCodecError(ValueError):
    """Base class for clear codec failures."""


class ActionIndexOutOfRangeError(ActionCodecError):
    pass


class ReservedActionIndexError(ActionCodecError):
    pass


class UnknownActionError(ActionCodecError):
    pass


class InvalidStructuredActionError(ActionCodecError):
    pass


class ActionCodecValidationError(ActionCodecError):
    pass


ActionT = TypeVar("ActionT", bound=GameAction)


@dataclass(frozen=True)
class ActionFamily(Generic[ActionT]):
    """One registered action type/range mapping."""

    name: str
    action_type: type[ActionT]
    action_range: ActionRange
    matches: Callable[[ActionT], bool]
    encode_local: Callable[[ActionT], int]
    decode_local: Callable[[int], ActionT]
    validate: Callable[[ActionT], None]
    describe_action: Callable[[ActionT], str]


def _require_exact_int(value, field: str) -> None:
    if type(value) is not int:
        raise InvalidStructuredActionError(f"{field} must be an integer")


def _slot_family(
    *,
    name: str,
    action_type: type[ActionT],
    action_range: ActionRange,
    field: str,
    label: str,
) -> ActionFamily[ActionT]:
    def validate(action):
        value = getattr(action, field)
        _require_exact_int(value, field)
        if not 0 <= value < action_range.capacity:
            raise InvalidStructuredActionError(
                f"{field} must be between 0 and {action_range.capacity - 1}"
            )

    return ActionFamily(
        name=name,
        action_type=action_type,
        action_range=action_range,
        matches=lambda action: True,
        encode_local=lambda action: getattr(action, field),
        decode_local=lambda local: action_type(**{field: local}),
        validate=validate,
        describe_action=lambda action: f"{label} {getattr(action, field)}",
    )


def _enum_family(
    *,
    name: str,
    action_type: type[ActionT],
    action_range: ActionRange,
    field: str,
    values: tuple,
    label: str,
) -> ActionFamily[ActionT]:
    if len(values) != action_range.capacity:
        raise ActionCodecValidationError(
            f"{name} defines {len(values)} values for capacity {action_range.capacity}"
        )

    def validate(action):
        value = getattr(action, field)
        if value not in values or type(value) is not type(values[0]):
            allowed = ", ".join(str(item.value) for item in values)
            raise InvalidStructuredActionError(f"{field} must be one of: {allowed}")

    return ActionFamily(
        name=name,
        action_type=action_type,
        action_range=action_range,
        matches=lambda action: True,
        encode_local=lambda action: values.index(getattr(action, field)),
        decode_local=lambda local: action_type(**{field: values[local]}),
        validate=validate,
        describe_action=lambda action: f"{label}: {getattr(action, field).value}",
    )


def _value_family(
    *,
    name: str,
    action_type: type[ActionT],
    action_range: ActionRange,
    field: str,
    values: tuple[int, ...],
    label: str,
) -> ActionFamily[ActionT]:
    if len(values) != action_range.capacity:
        raise ActionCodecValidationError(
            f"{name} defines {len(values)} values for capacity {action_range.capacity}"
        )

    def validate(action):
        value = getattr(action, field)
        _require_exact_int(value, field)
        if value not in values:
            raise InvalidStructuredActionError(f"{field} must be one of: {values}")

    return ActionFamily(
        name=name,
        action_type=action_type,
        action_range=action_range,
        matches=lambda action: True,
        encode_local=lambda action: values.index(getattr(action, field)),
        decode_local=lambda local: action_type(**{field: values[local]}),
        validate=validate,
        describe_action=lambda action: f"{label}: {getattr(action, field)}",
    )


def _singleton_family(
    *,
    name: str,
    action_type: type[ActionT],
    action_range: ActionRange,
    label: str,
) -> ActionFamily[ActionT]:
    if action_range.capacity != 1:
        raise ActionCodecValidationError(f"{name} must use a one-entry range")
    return ActionFamily(
        name=name,
        action_type=action_type,
        action_range=action_range,
        matches=lambda action: True,
        encode_local=lambda action: 0,
        decode_local=lambda local: action_type(),
        validate=lambda action: None,
        describe_action=lambda action: label,
    )


def _post_family(action_range: ActionRange, shape: PieceShape) -> ActionFamily[SelectPost]:
    def validate(action):
        _require_exact_int(action.post_slot, "post_slot")
        if not 0 <= action.post_slot < action_range.capacity:
            raise InvalidStructuredActionError(
                f"post_slot must be between 0 and {action_range.capacity - 1}"
            )
        if type(action.shape) is not PieceShape:
            raise InvalidStructuredActionError("shape must be a PieceShape")

    return ActionFamily(
        name=action_range.name,
        action_type=SelectPost,
        action_range=action_range,
        matches=lambda action: action.shape is shape,
        encode_local=lambda action: action.post_slot,
        decode_local=lambda local: SelectPost(local, shape),
        validate=validate,
        describe_action=lambda action: f"Post {action.post_slot}: {shape.value}",
    )


EXCHANGEABLE_MARKERS = (
    BonusMarkerType.SWAP_OFFICE,
    BonusMarkerType.MOVE_THREE,
    BonusMarkerType.UPGRADE_ABILITY,
    BonusMarkerType.THREE_ACTIONS,
    BonusMarkerType.FOUR_ACTIONS,
    BonusMarkerType.EXCHANGE_BONUS_MARKER,
    BonusMarkerType.TRIBUTE_TRADING_POST,
    BonusMarkerType.BLOCK_TRADE_ROUTE,
)

DEFAULT_ACTION_FAMILIES = (
    _post_family(POST_TRADER, PieceShape.TRADER),
    _post_family(POST_MERCHANT, PieceShape.MERCHANT),
    _slot_family(
        name=ROUTE_SELECT.name,
        action_type=SelectRoute,
        action_range=ROUTE_SELECT,
        field="route_slot",
        label="Route",
    ),
    _enum_family(
        name=ROUTE_OUTCOME.name,
        action_type=SelectRouteOutcome,
        action_range=ROUTE_OUTCOME,
        field="outcome",
        values=tuple(RouteOutcome),
        label="Route outcome",
    ),
    _slot_family(
        name=ROUTE_ENDPOINT.name,
        action_type=SelectRouteEndpoint,
        action_range=ROUTE_ENDPOINT,
        field="endpoint_slot",
        label="Route endpoint",
    ),
    _slot_family(
        name=ROUTE_UPGRADE_SLOT.name,
        action_type=SelectCityUpgradeSlot,
        action_range=ROUTE_UPGRADE_SLOT,
        field="upgrade_slot",
        label="City upgrade slot",
    ),
    _value_family(
        name=PRESTIGE_VALUE.name,
        action_type=SelectPrestigeValue,
        action_range=PRESTIGE_VALUE,
        field="value",
        values=(7, 8, 9, 11),
        label="Prestige value",
    ),
    _enum_family(
        name=PIECE_SHAPE.name,
        action_type=SelectPieceShape,
        action_range=PIECE_SHAPE,
        field="shape",
        values=tuple(PieceShape),
        label="Piece shape",
    ),
    _slot_family(
        name=INCOME_MERCHANT_COUNT.name,
        action_type=SelectIncome,
        action_range=INCOME_MERCHANT_COUNT,
        field="merchant_count",
        label="Income Merchant count",
    ),
    _slot_family(
        name=EXACT_TWO_MERCHANT_COUNT.name,
        action_type=SelectTwoPieceMix,
        action_range=EXACT_TWO_MERCHANT_COUNT,
        field="merchant_count",
        label="Two-piece Merchant count",
    ),
    _slot_family(
        name=TRIBUTE_MERCHANT_COUNT.name,
        action_type=SelectTributeIncome,
        action_range=TRIBUTE_MERCHANT_COUNT,
        field="merchant_count",
        label="Tribute Merchant count",
    ),
    _enum_family(
        name=BONUS_MARKER_ACTIVATE.name,
        action_type=ActivateBonusMarker,
        action_range=BONUS_MARKER_ACTIVATE,
        field="marker_type",
        values=tuple(BonusMarkerType),
        label="Activate bonus marker",
    ),
    _enum_family(
        name=BONUS_MARKER_TAKE_USED.name,
        action_type=SelectUsedBonusMarker,
        action_range=BONUS_MARKER_TAKE_USED,
        field="marker_type",
        values=EXCHANGEABLE_MARKERS,
        label="Take used bonus marker",
    ),
    _enum_family(
        name=TILE_BUY.name,
        action_type=BuyEmperorsFavour,
        action_range=TILE_BUY,
        field="tile_type",
        values=tuple(EmperorsFavourType),
        label="Buy Emperor's Favour",
    ),
    _enum_family(
        name=TILE_PAYMENT.name,
        action_type=SelectBonusMarkerPayment,
        action_range=TILE_PAYMENT,
        field="marker_type",
        values=EXCHANGEABLE_MARKERS,
        label="Pay bonus marker",
    ),
    _enum_family(
        name=INCOME_FAVOUR_RESPONSE.name,
        action_type=RespondToIncomeFavour,
        action_range=INCOME_FAVOUR_RESPONSE,
        field="choice",
        values=tuple(IncomeFavourChoice),
        label="Income Favour response",
    ),
    _slot_family(
        name=PLAYER_TARGET.name,
        action_type=SelectPlayer,
        action_range=PLAYER_TARGET,
        field="player_slot",
        label="Player",
    ),
    _slot_family(
        name=CITY_TARGET.name,
        action_type=SelectCity,
        action_range=CITY_TARGET,
        field="city_slot",
        label="City",
    ),
    _slot_family(
        name=OFFICE_PAIR.name,
        action_type=SelectOfficePair,
        action_range=OFFICE_PAIR,
        field="pair_slot",
        label="Office pair",
    ),
    _slot_family(
        name=ABILITY_UPGRADE.name,
        action_type=SelectAbility,
        action_range=ABILITY_UPGRADE,
        field="ability_slot",
        label="Ability",
    ),
    _enum_family(
        name=DISPLACEMENT_SOURCE.name,
        action_type=SelectDisplacementSource,
        action_range=DISPLACEMENT_SOURCE,
        field="source",
        values=tuple(DisplacementSource),
        label="Displacement source",
    ),
    _enum_family(
        name=DISPLACEMENT_PIECE_KIND.name,
        action_type=SelectDisplacementPiece,
        action_range=DISPLACEMENT_PIECE_KIND,
        field="kind",
        values=tuple(DisplacementPieceKind),
        label="Displacement piece",
    ),
    _singleton_family(
        name=FINISH_MOVE_PICKUP.name,
        action_type=FinishMovePickup,
        action_range=FINISH_MOVE_PICKUP,
        label="Finish move pickup",
    ),
    _singleton_family(
        name=FINISH_DISPLACEMENT.name,
        action_type=FinishDisplacement,
        action_range=FINISH_DISPLACEMENT,
        label="Finish displacement",
    ),
    _singleton_family(
        name=DECLINE_DISPLACEMENT_OPTIONAL.name,
        action_type=DeclineDisplacementOptionalPieces,
        action_range=DECLINE_DISPLACEMENT_OPTIONAL,
        label="Decline optional displacement pieces",
    ),
    _singleton_family(
        name=END_TURN.name,
        action_type=EndTurn,
        action_range=END_TURN,
        label="End turn",
    ),
    _singleton_family(
        name=FORGO_BONUS_MARKER.name,
        action_type=ForgoBonusMarker,
        action_range=FORGO_BONUS_MARKER,
        label="Forgo optional bonus marker",
    ),
    _singleton_family(
        name=CONFIRM_BONUS_MARKER_REPLACEMENT.name,
        action_type=ConfirmBonusMarkerReplacement,
        action_range=CONFIRM_BONUS_MARKER_REPLACEMENT,
        label="Confirm bonus-marker replacement",
    ),
)


class ActionCodec:
    """Encode, decode, and describe actions through registered families."""

    def __init__(self, families: Iterable[ActionFamily] = DEFAULT_ACTION_FAMILIES):
        self.families = tuple(families)
        self._families_by_type = defaultdict(list)
        self._family_by_index = [None] * ACTION_SPACE_SIZE
        self._reserved = [False] * ACTION_SPACE_SIZE
        self._build_registry()
        self.validate()

    def _build_registry(self) -> None:
        for action_range in ACTION_RANGES:
            if action_range.reserved:
                for index in range(action_range.start, action_range.stop):
                    self._reserved[index] = True

        for family in self.families:
            self._families_by_type[family.action_type].append(family)
            for index in range(family.action_range.start, family.action_range.stop):
                if self._family_by_index[index] is not None:
                    other = self._family_by_index[index]
                    raise ActionCodecValidationError(
                        f"{family.name} overlaps registered family {other.name} at {index}"
                    )
                self._family_by_index[index] = family

    @staticmethod
    def _validate_index(index: int) -> None:
        if type(index) is not int:
            raise ActionIndexOutOfRangeError("Action index must be an integer")
        if not 0 <= index < ACTION_SPACE_SIZE:
            raise ActionIndexOutOfRangeError(
                f"Action index {index} is outside 0–{ACTION_SPACE_SIZE - 1}"
            )

    def encode(self, action: GameAction) -> int:
        candidates = self._families_by_type.get(type(action), ())
        if not candidates:
            raise UnknownActionError(f"No action family is registered for {type(action).__name__}")
        matches = [family for family in candidates if family.matches(action)]
        if not matches:
            raise InvalidStructuredActionError(
                f"{action!r} does not match a registered {type(action).__name__} family"
            )
        if len(matches) != 1:
            names = ", ".join(family.name for family in matches)
            raise ActionCodecValidationError(
                f"{action!r} matches multiple action families: {names}"
            )
        family = matches[0]
        family.validate(action)
        local_index = family.encode_local(action)
        if not 0 <= local_index < family.action_range.capacity:
            raise ActionCodecValidationError(
                f"{family.name} encoded invalid local index {local_index}"
            )
        return family.action_range.start + local_index

    def decode(self, index: int, state=None) -> GameAction:
        self._validate_index(index)
        if self._reserved[index]:
            raise ReservedActionIndexError(f"Action index {index} is reserved")
        family = self._family_by_index[index]
        if family is None:
            raise UnknownActionError(f"No action family is registered for index {index}")
        return family.decode_local(index - family.action_range.start)

    def describe(self, index: int) -> str:
        action = self.decode(index)
        family = self._family_by_index[index]
        return family.describe_action(action)

    def is_reserved(self, index: int) -> bool:
        self._validate_index(index)
        return self._reserved[index]

    def validate(self) -> None:
        active_ranges = [
            action_range for action_range in ACTION_RANGES if not action_range.reserved
        ]
        registered_ranges = [family.action_range for family in self.families]
        if len(registered_ranges) != len(set(registered_ranges)):
            raise ActionCodecValidationError("An active action range is registered more than once")
        if set(registered_ranges) != set(active_ranges):
            missing = set(active_ranges) - set(registered_ranges)
            extra = set(registered_ranges) - set(active_ranges)
            raise ActionCodecValidationError(
                f"Registered ranges differ from schema; missing={missing}, extra={extra}"
            )
        for family in self.families:
            if family.action_range.structured_action != family.action_type.__name__:
                raise ActionCodecValidationError(
                    f"{family.name} registers {family.action_type.__name__}, but the "
                    f"schema requires {family.action_range.structured_action}"
                )

        decoded_actions = {}
        for index in range(ACTION_SPACE_SIZE):
            if self.is_reserved(index):
                continue
            action = self.decode(index)
            encoded = self.encode(action)
            if encoded != index:
                raise ActionCodecValidationError(f"Round trip changed index {index} into {encoded}")
            if action in decoded_actions:
                raise ActionCodecValidationError(
                    f"Duplicate action mapping for {action!r}: "
                    f"{decoded_actions[action]} and {index}"
                )
            decoded_actions[action] = index
            if not self.describe(index):
                raise ActionCodecValidationError(f"Index {index} has an empty description")


DEFAULT_ACTION_CODEC = ActionCodec()
