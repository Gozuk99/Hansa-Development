"""Codec for stable Hansa interaction locations."""

from dataclasses import dataclass
from typing import Callable, Iterable

from game.action_schema import (
    ABILITY,
    ACTION_RANGES,
    ACTION_SPACE_SIZE,
    BONUS_MARKER,
    CITY,
    CONTROL,
    INCOME,
    POST,
    PLAYER,
    ROUTE,
    SUPPLY,
    TILE,
    ActionRange,
)
from game.structured_actions import (
    AbilityInteraction,
    BonusMarkerInteraction,
    CityInteraction,
    ControlInteraction,
    GameAction,
    IncomeInteraction,
    PieceShape,
    PostInteraction,
    PlayerInteraction,
    RouteInteraction,
    SupplyInteraction,
    TileInteraction,
)
from map_data.constants import MAX_POSTS, MAX_ROUTES


class ActionCodecError(ValueError):
    pass


class ActionIndexOutOfRangeError(ActionCodecError):
    pass


class ReservedActionIndexError(ActionCodecError):
    pass


class UnknownActionError(ActionCodecError):
    pass


class InvalidInteractionError(ActionCodecError):
    pass


class ActionCodecValidationError(ActionCodecError):
    pass


@dataclass(frozen=True)
class InteractionFamily:
    name: str
    action_type: type[GameAction]
    action_range: ActionRange
    encode_local: Callable[[GameAction], int]
    decode_local: Callable[[int], GameAction]
    validate_action: Callable[[GameAction], None]
    describe_action: Callable[[GameAction], str]


def _require_int(value, field: str) -> None:
    if type(value) is not int:
        raise InvalidInteractionError(f"{field} must be an integer")


def _validate_slot(value, field: str, capacity: int) -> None:
    _require_int(value, field)
    if not 0 <= value < capacity:
        raise InvalidInteractionError(f"{field} must be between 0 and {capacity - 1}")


def _post_family() -> InteractionFamily:
    def validate(action):
        _validate_slot(action.post_slot, "post_slot", MAX_POSTS)
        if type(action.shape) is not PieceShape:
            raise InvalidInteractionError("shape must be a PieceShape")

    def encode(action):
        shape_offset = MAX_POSTS if action.shape is PieceShape.MERCHANT else 0
        return shape_offset + action.post_slot

    def decode(local):
        shape = PieceShape.MERCHANT if local >= MAX_POSTS else PieceShape.TRADER
        return PostInteraction(local % MAX_POSTS, shape)

    return InteractionFamily(
        "post",
        PostInteraction,
        POST,
        encode,
        decode,
        validate,
        lambda action: f"Post {action.post_slot}: {action.shape.value} interaction",
    )


def _route_family() -> InteractionFamily:
    def validate(action):
        _validate_slot(action.route_slot, "route_slot", MAX_ROUTES)
        _validate_slot(action.interaction_slot, "interaction_slot", 7)

    def encode(action):
        slot = action.interaction_slot
        if slot == 0:
            return action.route_slot
        if slot <= 2:
            return MAX_ROUTES + action.route_slot * 2 + (slot - 1)
        return MAX_ROUTES * 3 + action.route_slot * 4 + (slot - 3)

    def decode(local):
        if local < MAX_ROUTES:
            return RouteInteraction(local, 0)
        if local < MAX_ROUTES * 3:
            relative = local - MAX_ROUTES
            route_slot, endpoint = divmod(relative, 2)
            return RouteInteraction(route_slot, endpoint + 1)
        relative = local - MAX_ROUTES * 3
        route_slot, outcome = divmod(relative, 4)
        return RouteInteraction(route_slot, outcome + 3)

    return InteractionFamily(
        "route",
        RouteInteraction,
        ROUTE,
        encode,
        decode,
        validate,
        lambda action: (f"Route {action.route_slot}: interaction {action.interaction_slot}"),
    )


def _linear_family(
    *,
    name: str,
    action_type: type[GameAction],
    action_range: ActionRange,
    field: str,
    label: str,
) -> InteractionFamily:
    def validate(action):
        _validate_slot(
            getattr(action, field),
            field,
            action_range.active_capacity,
        )

    return InteractionFamily(
        name,
        action_type,
        action_range,
        lambda action: getattr(action, field),
        lambda local: action_type(local),
        validate,
        lambda action: f"{label} {getattr(action, field)}",
    )


DEFAULT_INTERACTION_FAMILIES = (
    _post_family(),
    _route_family(),
    _linear_family(
        name="income",
        action_type=IncomeInteraction,
        action_range=INCOME,
        field="merchant_count",
        label="Income interaction",
    ),
    _linear_family(
        name="bonus_marker",
        action_type=BonusMarkerInteraction,
        action_range=BONUS_MARKER,
        field="marker_slot",
        label="Bonus-marker interaction",
    ),
    _linear_family(
        name="tile",
        action_type=TileInteraction,
        action_range=TILE,
        field="tile_slot",
        label="Tile interaction",
    ),
    _linear_family(
        name="city",
        action_type=CityInteraction,
        action_range=CITY,
        field="city_interaction_slot",
        label="City interaction",
    ),
    _linear_family(
        name="ability",
        action_type=AbilityInteraction,
        action_range=ABILITY,
        field="ability_slot",
        label="Ability interaction",
    ),
    _linear_family(
        name="supply",
        action_type=SupplyInteraction,
        action_range=SUPPLY,
        field="supply_slot",
        label="Player-supply interaction",
    ),
    _linear_family(
        name="player",
        action_type=PlayerInteraction,
        action_range=PLAYER,
        field="player_slot",
        label="Player interaction",
    ),
    _linear_family(
        name="control",
        action_type=ControlInteraction,
        action_range=CONTROL,
        field="control_slot",
        label="Control interaction",
    ),
)


class ActionCodec:
    """Translate stable interaction objects without calculating legality."""

    def __init__(
        self,
        families: Iterable[InteractionFamily] = DEFAULT_INTERACTION_FAMILIES,
    ):
        self.families = tuple(families)
        self._family_by_type = {}
        self._family_by_index: list[InteractionFamily | None] = [None] * ACTION_SPACE_SIZE
        self.validate()
        for family in self.families:
            self._family_by_type[family.action_type] = family
            for index in range(
                family.action_range.start,
                family.action_range.active_stop,
            ):
                self._family_by_index[index] = family

    @staticmethod
    def _validate_index(index: int) -> None:
        if type(index) is not int or not 0 <= index < ACTION_SPACE_SIZE:
            raise ActionIndexOutOfRangeError(
                f"Action index must be an integer from 0 through {ACTION_SPACE_SIZE - 1}"
            )

    def is_reserved(self, index: int) -> bool:
        self._validate_index(index)
        return self._family_by_index[index] is None

    def encode(self, action: GameAction) -> int:
        family = self._family_by_type.get(type(action))
        if family is None:
            raise UnknownActionError(
                f"No interaction family is registered for {type(action).__name__}"
            )
        family.validate_action(action)
        local = family.encode_local(action)
        if not 0 <= local < family.action_range.active_capacity:
            raise ActionCodecValidationError(f"{family.name} encoded inactive local slot {local}")
        return family.action_range.start + local

    def decode(self, index: int) -> GameAction:
        self._validate_index(index)
        family = self._family_by_index[index]
        if family is None:
            raise ReservedActionIndexError(f"Action index {index} is reserved")
        return family.decode_local(index - family.action_range.start)

    def describe(self, index: int) -> str:
        action = self.decode(index)
        family = self._family_by_index[index]
        return family.describe_action(action)

    def create_mask(self, legal_actions: Iterable[GameAction]) -> tuple[bool, ...]:
        mask = [False] * ACTION_SPACE_SIZE
        for action in legal_actions:
            index = self.encode(action)
            if mask[index]:
                raise ActionCodecValidationError(
                    f"Multiple legal interactions encoded to index {index}"
                )
            mask[index] = True
        return tuple(mask)

    def validate(self) -> None:
        active_ranges = tuple(
            action_range for action_range in ACTION_RANGES if action_range.active_capacity
        )
        registered_ranges = tuple(family.action_range for family in self.families)
        if len(set(registered_ranges)) != len(registered_ranges):
            raise ActionCodecValidationError("An interaction range is registered more than once")
        if set(active_ranges) != set(registered_ranges):
            raise ActionCodecValidationError("Registered interaction ranges differ from the schema")

        claimed_types = set()
        for family in self.families:
            if family.name != family.action_range.name:
                raise ActionCodecValidationError(
                    f"{family.name} does not match range {family.action_range.name}"
                )
            if family.action_type in claimed_types:
                raise ActionCodecValidationError(
                    f"{family.action_type.__name__} is registered twice"
                )
            claimed_types.add(family.action_type)

        decoded = {}
        for action_range in active_ranges:
            for index in range(action_range.start, action_range.active_stop):
                family = next(item for item in self.families if item.action_range == action_range)
                action = family.decode_local(index - action_range.start)
                family.validate_action(action)
                encoded = action_range.start + family.encode_local(action)
                if encoded != index:
                    raise ActionCodecValidationError(
                        f"Round trip changed index {index} into {encoded}"
                    )
                if action in decoded:
                    raise ActionCodecValidationError(
                        f"Duplicate interaction {action!r} at {decoded[action]} and {index}"
                    )
                decoded[action] = index


DEFAULT_ACTION_CODEC = ActionCodec()
