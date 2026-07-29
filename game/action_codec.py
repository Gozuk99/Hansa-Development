"""State-aware codec for complete Hansa decisions."""

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Iterable

from game.action_schema import (
    ABILITY,
    ACTION_RANGES,
    ACTION_SPACE_SIZE,
    BONUS_MARKER,
    CITY,
    CONTROL,
    EXACT_TWO,
    INCOME,
    POSITION,
    RESERVED,
    ROUTE,
    TILE,
    TRIBUTE_INCOME,
    USED_BONUS_MARKER,
    ActionRange,
)
from game.structured_actions import (
    ActivateBonusMarker,
    BuyEmperorsFavour,
    ClaimAdditionalTradingPost,
    ClaimGreenCity,
    ClaimRouteOffice,
    ClaimRoutePrestige,
    CompleteRouteForPoints,
    DisplaceOpponent,
    EndTurn,
    ExchangeForUsedBonusMarker,
    FinishDisplacement,
    FinishMovePickup,
    GameAction,
    PickUpDisplacementFallbackPiece,
    PickUpPiece,
    PlaceDisplacedPiece,
    PlaceFromPersonalSupply,
    PlaceHeldDisplacementFallbackPiece,
    PlaceHeldPiece,
    PlaceOptionalDisplacementPiece,
    RespondToIncomeFavour,
    SelectAbility,
    SelectBlockedRoute,
    SelectBonusMarkerPayment,
    SelectBonusMarkerReplacementRoute,
    SelectTributeIncome,
    SelectTributeRoute,
    SelectTwoPieceMix,
    SwapAdjacentOffices,
    TakeIncome,
    UpgradeFromRoute,
)


class ActionCodecError(ValueError):
    pass


class ActionIndexOutOfRangeError(ActionCodecError):
    pass


class ReservedActionIndexError(ActionCodecError):
    pass


class InactiveActionIndexError(ActionCodecError):
    pass


class UnknownActionError(ActionCodecError):
    pass


class DuplicateActionError(ActionCodecError):
    pass


class ActionFamilyCapacityError(ActionCodecError):
    pass


class ActionCodecValidationError(ActionCodecError):
    pass


POSITION_TYPES = (
    PlaceFromPersonalSupply,
    DisplaceOpponent,
    PickUpPiece,
    PlaceHeldPiece,
    PlaceDisplacedPiece,
    PlaceOptionalDisplacementPiece,
    PickUpDisplacementFallbackPiece,
    PlaceHeldDisplacementFallbackPiece,
)
ROUTE_TYPES = (
    CompleteRouteForPoints,
    ClaimRouteOffice,
    UpgradeFromRoute,
    ClaimRoutePrestige,
    ClaimAdditionalTradingPost,
    SelectTributeRoute,
    SelectBlockedRoute,
    SelectBonusMarkerReplacementRoute,
)
CITY_TYPES = (SwapAdjacentOffices, ClaimGreenCity)
TILE_TYPES = (BuyEmperorsFavour, SelectBonusMarkerPayment, RespondToIncomeFavour)
CONTROL_TYPES = (FinishMovePickup, FinishDisplacement, EndTurn)


@dataclass(frozen=True)
class ActionFamily:
    name: str
    action_range: ActionRange
    action_types: tuple[type[GameAction], ...]


DEFAULT_ACTION_FAMILIES = (
    ActionFamily("position", POSITION, POSITION_TYPES),
    ActionFamily("route", ROUTE, ROUTE_TYPES),
    ActionFamily("city", CITY, CITY_TYPES),
    ActionFamily("income", INCOME, (TakeIncome,)),
    ActionFamily("exact_two", EXACT_TWO, (SelectTwoPieceMix,)),
    ActionFamily("tribute_income", TRIBUTE_INCOME, (SelectTributeIncome,)),
    ActionFamily("bonus_marker", BONUS_MARKER, (ActivateBonusMarker,)),
    ActionFamily("used_bonus_marker", USED_BONUS_MARKER, (ExchangeForUsedBonusMarker,)),
    ActionFamily("tile", TILE, TILE_TYPES),
    ActionFamily("ability", ABILITY, (SelectAbility,)),
    ActionFamily("control", CONTROL, CONTROL_TYPES),
)


def _canonical_value(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return tuple(
            (field.name, _canonical_value(getattr(value, field.name))) for field in fields(value)
        )
    if isinstance(value, tuple):
        return tuple(_canonical_value(item) for item in value)
    return value


def _action_sort_key(action: GameAction):
    return (type(action).__name__, _canonical_value(action))


@dataclass(frozen=True)
class ActionCodecContext:
    """Stable family catalogue and legal subset for one state revision."""

    state_revision: int | str
    action_catalogue: tuple[GameAction, ...]
    legal_actions: tuple[GameAction, ...]

    @classmethod
    def from_actions(
        cls,
        state_revision: int | str,
        legal_actions: Iterable[GameAction],
        action_catalogue: Iterable[GameAction] | None = None,
    ) -> "ActionCodecContext":
        legal = tuple(legal_actions)
        catalogue = legal if action_catalogue is None else tuple(action_catalogue)
        return cls(state_revision, catalogue, legal)


@dataclass(frozen=True)
class EncodedDecision:
    state_revision: int | str
    mask: tuple[bool, ...]
    action_by_index: dict[int, GameAction]
    index_by_action: dict[GameAction, int]


class ActionCodec:
    """Translate complete engine-owned decisions through family-local slots."""

    def __init__(self, families: Iterable[ActionFamily] = DEFAULT_ACTION_FAMILIES):
        self.families = tuple(families)
        self._family_by_type: dict[type[GameAction], ActionFamily] = {}
        self._family_by_index: list[ActionFamily | None] = [None] * ACTION_SPACE_SIZE
        self.validate()
        for family in self.families:
            for action_type in family.action_types:
                self._family_by_type[action_type] = family
            for index in range(family.action_range.start, family.action_range.stop):
                self._family_by_index[index] = family

    @staticmethod
    def _validate_index(index: int) -> None:
        if type(index) is not int or not 0 <= index < ACTION_SPACE_SIZE:
            raise ActionIndexOutOfRangeError(
                f"Action index must be an integer from 0 through {ACTION_SPACE_SIZE - 1}"
            )

    def is_reserved(self, index: int) -> bool:
        self._validate_index(index)
        return RESERVED.start <= index < RESERVED.stop

    def _group_catalogue(
        self, context: ActionCodecContext
    ) -> dict[ActionFamily, tuple[GameAction, ...]]:
        if len(context.action_catalogue) != len(set(context.action_catalogue)):
            raise DuplicateActionError("Action catalogue contains duplicate decisions")
        if len(context.legal_actions) != len(set(context.legal_actions)):
            raise DuplicateActionError("Legal action context contains duplicate decisions")
        if not set(context.legal_actions).issubset(context.action_catalogue):
            raise UnknownActionError("Every legal action must appear in the action catalogue")

        grouped: dict[ActionFamily, list[GameAction]] = {family: [] for family in self.families}
        for action in context.action_catalogue:
            family = self._family_by_type.get(type(action))
            if family is None:
                raise UnknownActionError(
                    f"No action family is registered for {type(action).__name__}"
                )
            grouped[family].append(action)

        result = {}
        for family, actions in grouped.items():
            ordered = tuple(sorted(actions, key=_action_sort_key))
            if len(ordered) > family.action_range.capacity:
                raise ActionFamilyCapacityError(
                    f"{family.name} has {len(ordered)} legal decisions but only "
                    f"{family.action_range.capacity} schema slots"
                )
            result[family] = ordered
        return result

    def build_decision(self, context: ActionCodecContext) -> EncodedDecision:
        grouped = self._group_catalogue(context)
        action_by_index = {}
        index_by_action = {}
        legal_actions = set(context.legal_actions)
        for family, actions in grouped.items():
            for local_index, action in enumerate(actions):
                index = family.action_range.start + local_index
                index_by_action[action] = index
                if action in legal_actions:
                    action_by_index[index] = action

        mask = tuple(index in action_by_index for index in range(ACTION_SPACE_SIZE))
        return EncodedDecision(
            context.state_revision,
            mask,
            action_by_index,
            index_by_action,
        )

    def encode(self, action: GameAction, context: ActionCodecContext) -> int:
        if action not in context.legal_actions:
            if type(action) not in self._family_by_type:
                raise UnknownActionError(
                    f"No action family is registered for {type(action).__name__}"
                )
            raise UnknownActionError(f"{action!r} is not legal in this context")
        try:
            return self.build_decision(context).index_by_action[action]
        except KeyError as exc:
            if type(action) not in self._family_by_type:
                raise UnknownActionError(
                    f"No action family is registered for {type(action).__name__}"
                ) from exc
            raise UnknownActionError(f"{action!r} is not legal in this context") from exc

    def decode(self, index: int, context: ActionCodecContext) -> GameAction:
        self._validate_index(index)
        if self.is_reserved(index):
            raise ReservedActionIndexError(f"Action index {index} is reserved")
        try:
            return self.build_decision(context).action_by_index[index]
        except KeyError as exc:
            raise InactiveActionIndexError(
                f"Action index {index} is inactive in state {context.state_revision}"
            ) from exc

    def describe(self, index: int, context: ActionCodecContext) -> str:
        action = self.decode(index, context)
        values = ", ".join(
            f"{field.name}={_canonical_value(getattr(action, field.name))}"
            for field in fields(action)
        )
        return f"{type(action).__name__}({values})"

    def validate(self) -> None:
        active_ranges = tuple(
            action_range for action_range in ACTION_RANGES if not action_range.reserved
        )
        registered_ranges = tuple(family.action_range for family in self.families)
        if len(set(registered_ranges)) != len(registered_ranges):
            raise ActionCodecValidationError("An action range is registered more than once")
        if set(active_ranges) != set(registered_ranges):
            raise ActionCodecValidationError(
                "Registered action-family ranges differ from the active schema"
            )

        claimed_types = set()
        for family in self.families:
            if family.name != family.action_range.family:
                raise ActionCodecValidationError(
                    f"{family.name} does not match schema family {family.action_range.family}"
                )
            if not family.action_types:
                raise ActionCodecValidationError(f"{family.name} has no action types")
            for action_type in family.action_types:
                if action_type in claimed_types:
                    raise ActionCodecValidationError(
                        f"{action_type.__name__} is registered by multiple families"
                    )
                claimed_types.add(action_type)


DEFAULT_ACTION_CODEC = ActionCodec()
