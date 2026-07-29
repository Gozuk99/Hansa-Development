# Atomic Hansa Action Schema Version 2

## Status

This document defines the unactivated 768-entry schema that replaces the
superseded version-1 draft.

- `ACTION_SCHEMA_VERSION = 2`
- `ACTION_SPACE_SIZE = 768`
- active capacity: 626
- reserved capacity: 142
- registry: `game/action_schema.py`

The current 620-entry production mask and dispatcher remain unchanged.

## Decision boundary

One enabled neural-network index must decode into one complete, immediately
executable Hansa decision. The codec does not encode UI clicks or incomplete
route, city, shape, source, or outcome selections.

Legitimate rules workflows remain staged when an action changes authoritative
engine state. Examples include Move pickup and placement, tile payments, and
board-fallback pickup followed by placement.

## Allocation

| Range | Indices | Capacity | Active semantic family |
|---|---:|---:|---|
| `POSITION` | `0–351` | 352 | Complete post, movement, and displacement decisions |
| `ROUTE` | `352–511` | 160 | Complete route outcomes and route targets |
| `CITY` | `512–557` | 46 | Complete Swap Office and green-city decisions |
| `INCOME` | `558–562` | 5 | Normal Income composition |
| `EXACT_TWO` | `563–565` | 3 | Exact-two-piece composition |
| `TRIBUTE_INCOME` | `566–568` | 3 | Tribute income composition |
| `BONUS_MARKER` | `569–577` | 9 | Bonus-marker activation |
| `USED_BONUS_MARKER` | `578–609` | 32 | Complete opponent/used-marker exchange |
| `TILE` | `610–617` | 8 | Tile, payment, or Income Favour decision |
| `ABILITY` | `618–622` | 5 | Ability decisions |
| `CONTROL` | `623–625` | 3 | Finish pickup, finish displacement, end turn |
| `RESERVED` | `626–767` | 142 | Reserved |

The active total is:

```text
352 + 160 + 46 + 5 + 3 + 3 + 9 + 32 + 8 + 5 + 3 = 626
```

## Capacity basis

### Position and displacement

Ordinary post selection needs at most `121 × 2 = 242` identities. Atomic
displacement is larger. On the largest supported map, after excluding the
occupied displacement origin, the maximum is:

```text
2 Merchant identities × 120 Merchant-compatible destinations
+ 1 Trader identity × 112 Trader-compatible destinations
= 352
```

The contexts are mutually exclusive, so the family needs 352 slots.

### Route

Ordinary completion requires no more than 136 map-specific outcomes. Additional
Trading Post requires:

```text
40 routes × 2 endpoints × 2 shapes = 160
```

Tribute, Block, and marker replacement require at most 40 route targets each
and are mutually exclusive contexts. The family therefore needs 160 slots.

### City

Map 2 contains the largest static adjacent-office-pair catalogue at 46.
Green-city placement requires at most six `(city, shape)` choices. These
workflows are mutually exclusive, so the family needs 46 slots.

## Complete action families

Position actions include:

- `PlaceFromPersonalSupply`
- `DisplaceOpponent`
- `PickUpPiece`
- `PlaceHeldPiece`
- `PlaceDisplacedPiece`
- `PlaceOptionalDisplacementPiece`
- `PickUpDisplacementFallbackPiece`
- `PlaceHeldDisplacementFallbackPiece`

Route actions include:

- `CompleteRouteForPoints`
- `ClaimRouteOffice`
- `UpgradeFromRoute`
- `ClaimRoutePrestige`
- `ClaimAdditionalTradingPost`
- `SelectTributeRoute`
- `SelectBlockedRoute`
- `SelectBonusMarkerReplacementRoute`

City actions include:

- `SwapAdjacentOffices`
- `ClaimGreenCity`

The remaining families contain complete income, marker, tile, ability, and
control decisions defined in `game/structured_actions.py`.

Exchange Bonus Marker uses one complete
`ExchangeForUsedBonusMarker(target_player_id, marker_type)` decision. With up
to four opponents and eight exchangeable marker types, its stable family
requires `4 × 8 = 32` slots; there is no separate player-selection stage.

## Displacement semantics

The displaced piece, optional piece, shape, and destination are not separate
codec stages.

- `PlaceDisplacedPiece(destination)` places the mandatory piece whose shape is
  already stored in engine state.
- `PlaceOptionalDisplacementPiece(shape, destination)` distinguishes an
  optional piece, including an optional piece matching the displaced shape.
- General Stock and Personal Supply priority is determined by rules state, not
  selected through another action.
- Board fallback legitimately uses a pickup action followed by a held-piece
  placement action.
- `FinishDisplacement` declines all remaining optional pieces and finishes the
  workflow. Decline and finish are not separate actions.

## Route semantics

Route, outcome, endpoint, upgrade, prestige value, and Additional Trading Post
shape are combined into outcome-specific actions. The schema contains no
executable `SelectRouteOutcome`, `SelectRouteEndpoint`,
`SelectCityUpgradeSlot`, `SelectPrestigeValue`, or generic shape fragment.

## State-aware codec

`ActionCodec` receives a frozen `ActionCodecContext` containing a stable
workflow catalogue and the complete legal subset for one authoritative state
revision. The engine will own the legal set in the later legality milestone.
Map and workflow definitions provide the catalogue of complete actions that
may occupy the family in that context.

```python
context = ActionCodecContext.from_actions(
    state_revision,
    legal_actions,
    action_catalogue,
)
decision = codec.build_decision(context)
index = codec.encode(action, context)
action = codec.decode(index, context)
label = codec.describe(index, context)
```

Within each family, the complete action catalogue is sorted deterministically
and assigned to family-local slots. Legality changes the mask but does not
compact the catalogue or shift another action's index. The same frozen context
must be used to create the mask and interpret the selected index.

Required invariants:

- every legal action maps to exactly one enabled index;
- every enabled index maps to exactly one legal action;
- duplicate actions are rejected;
- a family exceeding capacity is rejected;
- reserved indices never decode;
- inactive indices fail distinctly;
- identical context and index always produce the same action.

The codec translates legality; it does not calculate legality.

## Compatibility

Version 2 is incompatible with the superseded version-1 draft. Version 1 had no
production artifacts, so no runtime migration is provided. The legacy
620-entry production action system remains in service until a later milestone
switches it deliberately.
