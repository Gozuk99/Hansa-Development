# Action Schema Version 1

## Status and scope

This document defines the authoritative 768-entry action allocation that later
milestones will expose through structured actions and `ActionCodec`.
Milestone 2 does not activate this schema or change game behavior.

- `ACTION_SCHEMA_VERSION = 1`
- `ACTION_SPACE_SIZE = 768`
- valid indices: `0–767`
- active capacity: 400 indices
- permanently reserved capacity: 368 indices
- authoritative registry: `game/action_schema.py`

Every active range has one stable semantic family. Reserved ranges have no
structured action and must always be masked once this schema is activated.

## Design boundary

The engine will own rules and structured legality. `ActionCodec` will be the
only component allowed to translate between structured actions and these
indices. AI, GUI, replay, history, and debugging code will query the codec
rather than duplicate ranges or arithmetic. The engine and AI must remain
headless and must not import drawing modules.

Several current one-step integer actions combine a target and an outcome.
Version 1 represents those as explicit structured selections, such as selecting
a route and then selecting its outcome. This removes unrelated contextual
reuse without changing rules; the structured action and codec milestones will
define the state transitions.

## Complete allocation

All ranges are half-open in code (`start <= index < stop`) and inclusive in the
table.

| Constant | Inclusive range | Capacity | Status | Structured action family |
|---|---:|---:|---|---|
| `POST_TRADER` | `0–120` | 121 | Active | `SelectPost` |
| `POST_MERCHANT` | `121–241` | 121 | Active | `SelectPost` |
| `POST_RESERVED` | `242–255` | 14 | Reserved | — |
| `ROUTE_SELECT` | `256–295` | 40 | Active | `SelectRoute` |
| `ROUTE_OUTCOME` | `296–300` | 5 | Active | `SelectRouteOutcome` |
| `ROUTE_ENDPOINT` | `301–302` | 2 | Active | `SelectRouteEndpoint` |
| `ROUTE_UPGRADE_SLOT` | `303–304` | 2 | Active | `SelectCityUpgradeSlot` |
| `PRESTIGE_VALUE` | `305–308` | 4 | Active | `SelectPrestigeValue` |
| `PIECE_SHAPE` | `309–310` | 2 | Active | `SelectPieceShape` |
| `ROUTE_RESERVED` | `311–575` | 265 | Reserved | — |
| `INCOME_MERCHANT_COUNT` | `576–580` | 5 | Active | `SelectIncome` |
| `EXACT_TWO_MERCHANT_COUNT` | `581–583` | 3 | Active | `SelectTwoPieceMix` |
| `TRIBUTE_MERCHANT_COUNT` | `584–586` | 3 | Active | `SelectTributeIncome` |
| `PIECE_CHOICE_RESERVED` | `587–607` | 21 | Reserved | — |
| `BONUS_MARKER_ACTIVATE` | `608–616` | 9 | Active | `ActivateBonusMarker` |
| `BONUS_MARKER_TAKE_USED` | `617–624` | 8 | Active | `SelectUsedBonusMarker` |
| `BONUS_MARKER_RESERVED` | `625–639` | 15 | Reserved | — |
| `TILE_BUY` | `640–645` | 6 | Active | `BuyEmperorsFavour` |
| `TILE_PAYMENT` | `646–653` | 8 | Active | `SelectBonusMarkerPayment` |
| `INCOME_FAVOUR_RESPONSE` | `654–656` | 3 | Active | `RespondToIncomeFavour` |
| `TILE_RESERVED` | `657–671` | 15 | Reserved | — |
| `PLAYER_TARGET` | `672–676` | 5 | Active | `SelectPlayer` |
| `CITY_TARGET` | `677–706` | 30 | Active | `SelectCity` |
| `OFFICE_PAIR` | `707–713` | 7 | Active | `SelectOfficePair` |
| `ABILITY_UPGRADE` | `714–718` | 5 | Active | `SelectAbility` |
| `CHOICE_RESERVED` | `719` | 1 | Reserved | — |
| `DISPLACEMENT_SOURCE` | `720–722` | 3 | Active | `SelectDisplacementSource` |
| `DISPLACEMENT_PIECE_KIND` | `723–724` | 2 | Active | `SelectDisplacementPiece` |
| `DISPLACEMENT_RESERVED` | `725–751` | 27 | Reserved | — |
| `FINISH_MOVE_PICKUP` | `752` | 1 | Active | `FinishMovePickup` |
| `FINISH_DISPLACEMENT` | `753` | 1 | Active | `FinishDisplacement` |
| `DECLINE_DISPLACEMENT_OPTIONAL` | `754` | 1 | Active | `DeclineDisplacementOptionalPieces` |
| `END_TURN` | `755` | 1 | Active | `EndTurn` |
| `FORGO_BONUS_MARKER` | `756` | 1 | Active | `ForgoBonusMarker` |
| `CONFIRM_BONUS_MARKER_REPLACEMENT` | `757` | 1 | Active | `ConfirmBonusMarkerReplacement` |
| `CONTROL_RESERVED` | `758–767` | 10 | Reserved | — |

The ranges are contiguous, non-overlapping, and cover every index from 0
through 767 exactly once.

## Encoding and decoding rules

### Post choices

`POST_TRADER` and `POST_MERCHANT` use the selected map's stable flattened post
order.

```text
Trader:   index = POST_TRADER.start + post_slot
Merchant: index = POST_MERCHANT.start + post_slot
decode:   post_slot = index - range.start
```

Slots beyond the selected map's post count are illegal and masked. The
structured action always means selecting that post and shape. Phase-specific
rules decide whether selection places, picks up, displaces, or relocates a
piece; the index meaning does not change.

### Route and route-outcome choices

```text
route index = ROUTE_SELECT.start + route_slot
decode route_slot = index - ROUTE_SELECT.start
```

`ROUTE_OUTCOME` uses these stable local choices:

| Local choice | Meaning |
|---:|---|
| 0 | Complete route for points/marker handling without a step-three benefit |
| 1 | Establish a normal office |
| 2 | Develop an ability |
| 3 | Claim a special-prestige value |
| 4 | Establish an Additional Trading Post office |

The remaining route decisions use:

```text
endpoint index = ROUTE_ENDPOINT.start + endpoint_slot       # 0 or 1
upgrade index  = ROUTE_UPGRADE_SLOT.start + upgrade_slot    # 0 or 1
prestige index = PRESTIGE_VALUE.start + value_slot          # 0 through 3
shape index    = PIECE_SHAPE.start + shape_slot             # Trader, Merchant
```

Prestige slots map to `(7, 8, 9, 11)`. Endpoint and upgrade slots are decoded
relative to the previously selected route. Separating these choices prevents
ordinary upgrades, prestige values, and Additional Trading Post choices from
sharing an index.

### Income and piece mixes

```text
normal income Merchant count = index - INCOME_MERCHANT_COUNT.start  # 0–4
exact-two Merchant count      = index - EXACT_TWO_MERCHANT_COUNT.start  # 0–2
Tribute Merchant count        = index - TRIBUTE_MERCHANT_COUNT.start  # 0–2
```

Eastern and Britannia exact-two effects share `SelectTwoPieceMix` because the
index has the same stable meaning: the Merchant count among exactly two
pieces. Tribute remains separate because it selects a composition among up to
two pieces.

### Bonus markers

`BONUS_MARKER_ACTIVATE` local slots are:

| Local slot | Marker |
|---:|---|
| 0 | `SwapOffice` |
| 1 | `Move3` |
| 2 | `UpgradeAbility` |
| 3 | `3Actions` |
| 4 | `4Actions` |
| 5 | `ExchangeBonusMarker` |
| 6 | `Tribute4EstablishingTP` |
| 7 | `BlockTradeRoute` |
| 8 | `PlaceAdjacent` |

`BONUS_MARKER_TAKE_USED` uses local slots 0–7 in the same order, excluding
`PlaceAdjacent`, which cannot be taken through an exchange.

```text
index = range.start + marker_slot
marker_slot = index - range.start
```

Activation and selecting an opponent's used marker are separate ranges.

### Emperor tiles, payment, and Income Favour

`TILE_BUY` local slots map in order to:

1. `DisplaceAnywhere`
2. `+1Action`
3. `+1IncomeIfOthersIncome`
4. `+1DisplacedPiece`
5. `+4PtsPerOwnedCity`
6. `+7PtsPerCompletedAbility`

`TILE_PAYMENT` uses local slots 0–7 for the eight exchangeable marker types in
the bonus-marker table. `INCOME_FAVOUR_RESPONSE` uses local slots 0, 1, and 2
for Trader, Merchant, and decline.

For each range:

```text
index = range.start + local_slot
local_slot = index - range.start
```

### Player, city, office, ability, and shape choices

`PLAYER_TARGET` and `CITY_TARGET` use player-list and selected-map city-list
slots, respectively.

```text
player index = PLAYER_TARGET.start + player_slot
city index   = CITY_TARGET.start + city_slot
```

`OFFICE_PAIR` local slots 0–6 identify an adjacent office pair within the
previously selected city, in left-to-right order. Seven slots cover London's
eight offices on the largest supported Britannia board. `ABILITY_UPGRADE`
local slots 0–4 identify the selected map's stable five-entry upgrade list.
`PIECE_SHAPE` uses slot 0 for Trader and slot 1 for Merchant.

These stable selections replace the current compact list of whichever
player/city/office choices happen to be eligible.

### Displacement choices

`DISPLACEMENT_SOURCE` local slots identify:

| Local slot | Source |
|---:|---|
| 0 | General Stock |
| 1 | Personal Supply |
| 2 | A piece already on the board |

`DISPLACEMENT_PIECE_KIND` uses slot 0 for the mandatory displaced piece and
slot 1 for an optional additional piece. Post destination and shape choices use
the post and shape families rather than overloading a displacement control
index.

### Finish, decline, and turn controls

Indices 752–757 have individual meanings and no contextual reuse:

- 752 finishes only the pickup portion of Move 3 or Move Any 2;
- 753 finishes a displacement whose required placements are complete;
- 754 declines remaining optional displacement pieces;
- 755 ends the current turn;
- 756 forgoes optional bonus-marker use;
- 757 confirms the pending bonus-marker replacement transition.

These controls replace the unrelated meanings currently combined at index 618
and the displacement meaning currently combined with Additional Trading Post
at index 619.

## Authoritative staged workflows

The same stable selection family may participate in multiple workflows only
when its meaning is unchanged. For example, `ROUTE_SELECT` always selects a
route slot; the pending structured workflow determines why a route is being
selected, but never changes the index into a post, city, marker, or control.
The codec must encode and decode each step below exactly as listed.

| Current workflow | Version-1 structured selection sequence |
|---|---|
| Place a Trader or Merchant on an empty post | `POST_TRADER` or `POST_MERCHANT` |
| Displace an opponent on a post | `POST_TRADER` or `POST_MERCHANT`, describing the piece placed on that post |
| Normal Move pickup and placement | Post selection for each pickup, followed by post selection for each held-piece destination; `FINISH_MOVE_PICKUP` ends pickup early without discarding held placements |
| Move 3 | `BONUS_MARKER_ACTIVATE[Move3]`, opponent post selections, optional `FINISH_MOVE_PICKUP`, then destination post selections |
| Eastern Move Any 2 | activation of the permanent effect, own/opponent post selections, optional `FINISH_MOVE_PICKUP`, then destination post selections |
| Displacement response | `DISPLACEMENT_PIECE_KIND`, `DISPLACEMENT_SOURCE` when a source choice exists, `PIECE_SHAPE`, then a destination post selection; board fallback uses `DISPLACEMENT_SOURCE[board]` followed by the owned post selected for pickup and then its destination post |
| Decline optional displacement pieces | `DECLINE_DISPLACEMENT_OPTIONAL`; `FINISH_DISPLACEMENT` remains the distinct completion after all required placement work |
| Complete a route | `ROUTE_SELECT`, then `ROUTE_OUTCOME`; office adds `ROUTE_ENDPOINT`, ability adds `ROUTE_ENDPOINT` and `ROUTE_UPGRADE_SLOT`, and prestige adds `PRESTIGE_VALUE` |
| Additional Trading Post | `BONUS_MARKER_ACTIVATE[PlaceAdjacent]`, `ROUTE_SELECT`, `ROUTE_OUTCOME[Additional Trading Post]`, `ROUTE_ENDPOINT`, then `PIECE_SHAPE` |
| Normal Income | `INCOME_MERCHANT_COUNT` |
| Eastern or Britannia exact-two-piece effect | `EXACT_TWO_MERCHANT_COUNT`, followed by destination post selections when the effect places pieces |
| Tribute income response | `TRIBUTE_MERCHANT_COUNT` |
| Activate a bonus marker | `BONUS_MARKER_ACTIVATE` |
| Exchange Bonus Marker | `BONUS_MARKER_ACTIVATE[ExchangeBonusMarker]`, `PLAYER_TARGET`, then `BONUS_MARKER_TAKE_USED` |
| Swap Office | `BONUS_MARKER_ACTIVATE[SwapOffice]`, `CITY_TARGET`, then `OFFICE_PAIR`; the office-pair slot is the left-to-right adjacent-pair slot in that selected city, not a compact global eligibility ordinal |
| Upgrade Ability marker | `BONUS_MARKER_ACTIVATE[UpgradeAbility]`, then `ABILITY_UPGRADE` |
| Tribute Trading Post target | `BONUS_MARKER_ACTIVATE[Tribute4EstablishingTP]`, then `ROUTE_SELECT`; this replaces the current first-post-as-route encoding |
| Block Trade Route target | `BONUS_MARKER_ACTIVATE[BlockTradeRoute]`, then `ROUTE_SELECT`; this replaces the current first-post-as-route encoding |
| Eastern green-city effect | activation of the permanent effect, `CITY_TARGET`, then `PIECE_SHAPE`; the city slot is the selected map's stable city slot, not a compact eligible-choice ordinal |
| Buy an Emperor's Favour tile | `TILE_BUY`, followed by one `TILE_PAYMENT` for each required marker payment |
| Income Favour response | `INCOME_FAVOUR_RESPONSE` |
| Replace a pending bonus marker | `ROUTE_SELECT`; this replaces the current dedicated replacement-route range and always decodes as selecting that stable route slot |
| Forgo optional bonus-marker use | `FORGO_BONUS_MARKER` |
| End the turn | `END_TURN`; if marker replacement must begin, the engine exposes `CONFIRM_BONUS_MARKER_REPLACEMENT` as a separate transition rather than reusing `END_TURN` |

Pending workflow state is engine state, not index semantics. Structured action
types introduced later must preserve these sequences, and the eventual codec
must reject a selection whose family is not valid for the pending workflow.

## Reserved-index contract

The following 368 indices are permanently reserved in schema version 1:

- `242–255`
- `311–575`
- `587–607`
- `625–639`
- `657–671`
- `719`
- `725–751`
- `758–767`

Once version 1 is activated:

- every reserved index must always be false in the legal-action mask;
- reserved indices must not decode into structured actions;
- no active action may encode into a reserved index;
- assigning meaning to a reserved index is an incompatible schema change and
  requires an `ACTION_SCHEMA_VERSION` bump.

## Validation

`validate_action_schema()` verifies:

- every range name is unique;
- every capacity is positive;
- ranges are contiguous and non-overlapping;
- active ranges name exactly one structured action family;
- reserved ranges have no structured action family;
- the final range ends exactly at `ACTION_SPACE_SIZE`.

`tests/test_action_schema.py` verifies the version, size, complete `0–767`
coverage, active-range semantics, and explicit reserved ranges. Milestone 2
does not alter the current 620-entry mask, dispatcher, GUI, save formats,
histories, or checkpoints.
