"""Structured player decisions for action schema version 1.

These immutable values describe choices only. They do not calculate legality
or mutate game state; those responsibilities remain with later engine work.
"""

from dataclasses import dataclass
from enum import Enum


class GameAction:
    """Base type for every structured game action."""


class DeclineAction(GameAction):
    """Base type for an explicit decline decision."""


class PassAction(GameAction):
    """Base type for passing an optional decision."""


class PieceShape(str, Enum):
    TRADER = "trader"
    MERCHANT = "merchant"


class RouteOutcome(str, Enum):
    POINTS = "points"
    OFFICE = "office"
    UPGRADE = "upgrade"
    PRESTIGE = "prestige"
    ADDITIONAL_TRADING_POST = "additional_trading_post"


class BonusMarkerType(str, Enum):
    SWAP_OFFICE = "SwapOffice"
    MOVE_THREE = "Move3"
    UPGRADE_ABILITY = "UpgradeAbility"
    THREE_ACTIONS = "3Actions"
    FOUR_ACTIONS = "4Actions"
    EXCHANGE_BONUS_MARKER = "ExchangeBonusMarker"
    TRIBUTE_TRADING_POST = "Tribute4EstablishingTP"
    BLOCK_TRADE_ROUTE = "BlockTradeRoute"
    PLACE_ADJACENT = "PlaceAdjacent"


class EmperorsFavourType(str, Enum):
    DISPLACE_ANYWHERE = "DisplaceAnywhere"
    ONE_ACTION = "+1Action"
    INCOME_FAVOUR = "+1IncomeIfOthersIncome"
    ONE_DISPLACED_PIECE = "+1DisplacedPiece"
    OWNED_CITY_POINTS = "+4PtsPerOwnedCity"
    COMPLETED_ABILITY_POINTS = "+7PtsPerCompletedAbility"


class IncomeFavourChoice(str, Enum):
    TRADER = "trader"
    MERCHANT = "merchant"
    DECLINE = "decline"


class DisplacementSource(str, Enum):
    GENERAL_STOCK = "general_stock"
    PERSONAL_SUPPLY = "personal_supply"
    BOARD = "board"


class DisplacementPieceKind(str, Enum):
    MANDATORY = "mandatory"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class SelectPost(GameAction):
    """Select a stable post slot and requested piece shape."""

    post_slot: int
    shape: PieceShape


@dataclass(frozen=True)
class SelectRoute(GameAction):
    route_slot: int


@dataclass(frozen=True)
class SelectRouteOutcome(GameAction):
    outcome: RouteOutcome


@dataclass(frozen=True)
class SelectRouteEndpoint(GameAction):
    endpoint_slot: int


@dataclass(frozen=True)
class SelectCityUpgradeSlot(GameAction):
    upgrade_slot: int


@dataclass(frozen=True)
class SelectPrestigeValue(GameAction):
    value: int


@dataclass(frozen=True)
class SelectPieceShape(GameAction):
    shape: PieceShape


@dataclass(frozen=True)
class SelectIncome(GameAction):
    merchant_count: int


@dataclass(frozen=True)
class SelectTwoPieceMix(GameAction):
    merchant_count: int


@dataclass(frozen=True)
class SelectTributeIncome(GameAction):
    merchant_count: int


@dataclass(frozen=True)
class ActivateBonusMarker(GameAction):
    marker_type: BonusMarkerType


@dataclass(frozen=True)
class SelectUsedBonusMarker(GameAction):
    marker_type: BonusMarkerType


@dataclass(frozen=True)
class BuyEmperorsFavour(GameAction):
    tile_type: EmperorsFavourType


@dataclass(frozen=True)
class SelectBonusMarkerPayment(GameAction):
    marker_type: BonusMarkerType


@dataclass(frozen=True)
class RespondToIncomeFavour(DeclineAction):
    choice: IncomeFavourChoice


@dataclass(frozen=True)
class SelectPlayer(GameAction):
    player_slot: int


@dataclass(frozen=True)
class SelectCity(GameAction):
    city_slot: int


@dataclass(frozen=True)
class SelectOfficePair(GameAction):
    pair_slot: int


@dataclass(frozen=True)
class SelectAbility(GameAction):
    ability_slot: int


@dataclass(frozen=True)
class SelectDisplacementSource(GameAction):
    source: DisplacementSource


@dataclass(frozen=True)
class SelectDisplacementPiece(GameAction):
    kind: DisplacementPieceKind


@dataclass(frozen=True)
class FinishMovePickup(GameAction):
    pass


@dataclass(frozen=True)
class FinishDisplacement(GameAction):
    pass


@dataclass(frozen=True)
class DeclineDisplacementOptionalPieces(DeclineAction):
    pass


@dataclass(frozen=True)
class EndTurn(GameAction):
    pass


@dataclass(frozen=True)
class ForgoBonusMarker(PassAction):
    pass


@dataclass(frozen=True)
class ConfirmBonusMarkerReplacement(GameAction):
    pass
