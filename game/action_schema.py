"""Authoritative allocation for the unactivated atomic Hansa action schema."""

from dataclasses import dataclass


ACTION_SPACE_SIZE = 768
ACTION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ActionRange:
    """One contiguous schema range."""

    name: str
    start: int
    capacity: int
    family: str | None
    reserved: bool = False

    @property
    def stop(self) -> int:
        return self.start + self.capacity

    @property
    def end(self) -> int:
        return self.stop - 1


POSITION = ActionRange("position", 0, 352, "position")
ROUTE = ActionRange("route", 352, 160, "route")
CITY = ActionRange("city", 512, 46, "city")
INCOME = ActionRange("income", 558, 5, "income")
EXACT_TWO = ActionRange("exact_two", 563, 3, "exact_two")
TRIBUTE_INCOME = ActionRange("tribute_income", 566, 3, "tribute_income")
BONUS_MARKER = ActionRange("bonus_marker", 569, 9, "bonus_marker")
USED_BONUS_MARKER = ActionRange("used_bonus_marker", 578, 32, "used_bonus_marker")
TILE = ActionRange("tile", 610, 8, "tile")
ABILITY = ActionRange("ability", 618, 5, "ability")
CONTROL = ActionRange("control", 623, 3, "control")
RESERVED = ActionRange("reserved", 626, 142, None, reserved=True)

ACTION_RANGES = (
    POSITION,
    ROUTE,
    CITY,
    INCOME,
    EXACT_TWO,
    TRIBUTE_INCOME,
    BONUS_MARKER,
    USED_BONUS_MARKER,
    TILE,
    ABILITY,
    CONTROL,
    RESERVED,
)


def validate_action_schema() -> None:
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
        if action_range.reserved != (action_range.family is None):
            raise ValueError(f"{action_range.name} must be reserved or name one action family")
        expected_start = action_range.stop

    if expected_start != ACTION_SPACE_SIZE:
        raise ValueError(f"Action ranges end at {expected_start}, expected {ACTION_SPACE_SIZE}")


validate_action_schema()
