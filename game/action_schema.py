"""Authoritative fixed ranges and slot catalogues for Hansa interactions."""

from dataclasses import dataclass


ACTION_SPACE_SIZE = 768
ACTION_SCHEMA_VERSION = 2
# SHA-256 of every assigned ``index:decoded-action`` pair in index order.
# The compatibility test fails if an assigned index or meaning changes without
# an intentional schema-version/fingerprint update.
ACTION_SCHEMA_FINGERPRINT = "f0ef36c33722b35380463b4309e09ddbed40ce8c37a2c1d48fb7a6aecacca16d"

BONUS_MARKER_TYPES = (
    "SwapOffice",
    "Move3",
    "UpgradeAbility",
    "3Actions",
    "4Actions",
    "ExchangeBonusMarker",
    "Tribute4EstablishingTP",
    "BlockTradeRoute",
)
BONUS_MARKER_SLOT_BY_TYPE = {name: slot for slot, name in enumerate(BONUS_MARKER_TYPES)}
BONUS_MARKER_PAYMENT_TYPES = BONUS_MARKER_TYPES + ("PlaceAdjacent",)

TILE_TYPES = (
    "DisplaceAnywhere",
    "+1Action",
    "+1IncomeIfOthersIncome",
    "+1DisplacedPiece",
    "+4PtsPerOwnedCity",
    "+7PtsPerCompletedAbility",
)
TILE_SLOT_BY_TYPE = {name: slot for slot, name in enumerate(TILE_TYPES)}


class ActionSchemaCompatibilityError(ValueError):
    """Raised when an artifact cannot be used with the active action schema."""


def action_schema_metadata() -> dict[str, int | str]:
    """Return the exact action-schema identity stored with persisted artifacts."""
    return {
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_space_size": ACTION_SPACE_SIZE,
        "action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT,
    }


def validate_action_schema_metadata(metadata, artifact="artifact") -> None:
    """Reject missing or incompatible schema identities without guessing a migration."""
    expected = action_schema_metadata()
    missing = [key for key in expected if metadata.get(key) is None]
    if missing:
        raise ActionSchemaCompatibilityError(
            f"{artifact} is missing action-schema metadata: {', '.join(missing)}"
        )
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ActionSchemaCompatibilityError(
            f"{artifact} uses an incompatible action schema: {'; '.join(mismatches)}. "
            "Use an explicit migration or the matching runtime."
        )


@dataclass(frozen=True)
class ActionRange:
    """One permanent family range with padding that never shifts later families."""

    name: str
    start: int
    capacity: int
    active_capacity: int
    interaction_type: str | None

    @property
    def stop(self) -> int:
        return self.start + self.capacity

    @property
    def end(self) -> int:
        return self.stop - 1

    @property
    def active_stop(self) -> int:
        return self.start + self.active_capacity

    @property
    def reserved_capacity(self) -> int:
        return self.capacity - self.active_capacity

    def contains(self, index: int) -> bool:
        return self.start <= index < self.stop

    def is_assigned(self, index: int) -> bool:
        return self.start <= index < self.active_stop


# The first eight families have permanent boundaries. Activating a padded slot
# inside one family never changes any later family's action numbers.
POST = ActionRange("post", 0, 256, 242, "PostInteraction")
ROUTE = ActionRange("route", 256, 320, 280, "RouteInteraction")
INCOME = ActionRange("income", 576, 16, 5, "IncomeInteraction")
BONUS_MARKER = ActionRange("bonus_marker", 592, 48, 41, "BonusMarkerInteraction")
TILE = ActionRange("tile", 640, 16, 6, "TileInteraction")
CITY = ActionRange("city", 656, 64, 52, "CityInteraction")
ABILITY = ActionRange("ability", 720, 8, 5, "AbilityInteraction")
SUPPLY = ActionRange("supply", 728, 2, 1, "SupplyInteraction")
PLAYER = ActionRange("player", 730, 6, 5, "PlayerInteraction")
CONTROL = ActionRange("control", 736, 8, 2, "ControlInteraction")
EXPANSION = ActionRange("expansion", 744, 24, 0, None)

ACTION_RANGES = (
    POST,
    ROUTE,
    INCOME,
    BONUS_MARKER,
    TILE,
    CITY,
    ABILITY,
    SUPPLY,
    PLAYER,
    CONTROL,
    EXPANSION,
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
        if not 0 <= action_range.active_capacity <= action_range.capacity:
            raise ValueError(f"{action_range.name} has invalid active capacity")
        if action_range.start != expected_start:
            raise ValueError(
                f"Expected {action_range.name} to start at {expected_start}, "
                f"got {action_range.start}"
            )
        if (action_range.active_capacity == 0) != (action_range.interaction_type is None):
            raise ValueError(
                f"{action_range.name} must assign an interaction type exactly "
                "when it has active slots"
            )
        expected_start = action_range.stop

    if expected_start != ACTION_SPACE_SIZE:
        raise ValueError(f"Action ranges end at {expected_start}, expected {ACTION_SPACE_SIZE}")


validate_action_schema()
