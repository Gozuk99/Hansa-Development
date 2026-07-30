"""Fixed player interactions for action schema version 2.

These values identify what the player selected. The authoritative game state
determines what that interaction means and whether it is legal.
"""

from dataclasses import dataclass
from enum import Enum


class GameAction:
    """Base type for one fixed Hansa interaction location."""


class PieceShape(str, Enum):
    TRADER = "trader"
    MERCHANT = "merchant"


@dataclass(frozen=True)
class PostInteraction(GameAction):
    """Select one map post using the Trader or Merchant interaction."""

    post_slot: int
    shape: PieceShape


@dataclass(frozen=True)
class RouteInteraction(GameAction):
    """Select one fixed interaction location belonging to a route.

    Slots retain the original route layout:
    0 = route body / points or contextual route target
    1-2 = endpoint office interactions
    3-6 = endpoint outcome boxes (upgrade, prestige, or Additional TP)
    """

    route_slot: int
    interaction_slot: int


@dataclass(frozen=True)
class IncomeInteraction(GameAction):
    """Select one of five composition interactions interpreted by workflow."""

    merchant_count: int


@dataclass(frozen=True)
class BonusMarkerInteraction(GameAction):
    """Select a fixed unused-marker or opponent-used-marker location."""

    marker_slot: int


@dataclass(frozen=True)
class TileInteraction(GameAction):
    """Select a fixed tile/payment/Income-Favour interaction location."""

    tile_slot: int


@dataclass(frozen=True)
class CityInteraction(GameAction):
    """Select one map-defined city interaction location."""

    city_interaction_slot: int


@dataclass(frozen=True)
class AbilityInteraction(GameAction):
    """Select one player-board ability box."""

    ability_slot: int


@dataclass(frozen=True)
class SupplyInteraction(GameAction):
    """Select one fixed player-supply source."""

    supply_slot: int


@dataclass(frozen=True)
class PlayerInteraction(GameAction):
    """Select one fixed player seat."""

    player_slot: int


@dataclass(frozen=True)
class ControlInteraction(GameAction):
    """Select one fixed workflow control."""

    control_slot: int
