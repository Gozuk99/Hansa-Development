"""Complete Hansa player decisions for atomic action schema version 2."""

from dataclasses import dataclass
from enum import Enum


class GameAction:
    """Immutable semantic decision accepted by the Hansa rules engine."""


class PieceShape(str, Enum):
    TRADER = "trader"
    MERCHANT = "merchant"


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


@dataclass(frozen=True)
class PlaceFromPersonalSupply(GameAction):
    post_id: str
    shape: PieceShape


@dataclass(frozen=True)
class DisplaceOpponent(GameAction):
    post_id: str
    shape: PieceShape


@dataclass(frozen=True)
class PickUpPiece(GameAction):
    post_id: str


@dataclass(frozen=True)
class PlaceHeldPiece(GameAction):
    post_id: str


@dataclass(frozen=True)
class PlaceDisplacedPiece(GameAction):
    destination_post_id: str


@dataclass(frozen=True)
class PlaceOptionalDisplacementPiece(GameAction):
    shape: PieceShape
    destination_post_id: str


@dataclass(frozen=True)
class PickUpDisplacementFallbackPiece(GameAction):
    source_post_id: str


@dataclass(frozen=True)
class PlaceHeldDisplacementFallbackPiece(GameAction):
    destination_post_id: str


@dataclass(frozen=True)
class CompleteRouteForPoints(GameAction):
    route_id: str


@dataclass(frozen=True)
class ClaimRouteOffice(GameAction):
    route_id: str
    city_id: str


@dataclass(frozen=True)
class UpgradeFromRoute(GameAction):
    route_id: str
    city_id: str
    ability_id: str


@dataclass(frozen=True)
class ClaimRoutePrestige(GameAction):
    route_id: str
    prestige_value: int


@dataclass(frozen=True)
class ClaimAdditionalTradingPost(GameAction):
    route_id: str
    city_id: str
    shape: PieceShape


@dataclass(frozen=True)
class SelectTributeRoute(GameAction):
    route_id: str


@dataclass(frozen=True)
class SelectBlockedRoute(GameAction):
    route_id: str


@dataclass(frozen=True)
class SelectBonusMarkerReplacementRoute(GameAction):
    route_id: str


@dataclass(frozen=True)
class SwapAdjacentOffices(GameAction):
    city_id: str
    left_office_id: str
    right_office_id: str


@dataclass(frozen=True)
class ClaimGreenCity(GameAction):
    city_id: str
    shape: PieceShape


@dataclass(frozen=True)
class TakeIncome(GameAction):
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
class ExchangeForUsedBonusMarker(GameAction):
    target_player_id: str
    marker_type: BonusMarkerType


@dataclass(frozen=True)
class BuyEmperorsFavour(GameAction):
    tile_type: EmperorsFavourType


@dataclass(frozen=True)
class SelectBonusMarkerPayment(GameAction):
    marker_type: BonusMarkerType


@dataclass(frozen=True)
class RespondToIncomeFavour(GameAction):
    choice: IncomeFavourChoice


@dataclass(frozen=True)
class SelectAbility(GameAction):
    ability_id: str


@dataclass(frozen=True)
class FinishMovePickup(GameAction):
    pass


@dataclass(frozen=True)
class FinishDisplacement(GameAction):
    """Decline all remaining optional pieces and finish displacement."""


@dataclass(frozen=True)
class EndTurn(GameAction):
    """Forgo unused optional markers and advance to replacement or next player."""
