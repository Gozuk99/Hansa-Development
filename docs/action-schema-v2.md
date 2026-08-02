# Hansa Interaction Schema Version 2

## Status

This document defines the active 768-entry AI interaction schema.

- `ACTION_SCHEMA_VERSION = 2`
- `ACTION_SPACE_SIZE = 768`
- assigned interaction slots: 639
- reserved family capacity: 129
- registry: `game/action_schema.py`

AI, headless play, and the GUI all use this schema. Structured actions execute
directly through the engine; there is no 620-entry compatibility adapter.

## Interaction boundary

One index identifies one permanent physical interaction location. Authoritative
engine state determines what selecting that location means and whether it is
legal. The codec neither calculates legality nor numbers a filtered list of
currently legal decisions.

Examples:

- a post interaction can place, pick up, displace, or relocate a piece;
- a route-body interaction can complete a route, select a Tribute/Block target,
  or place a replacement bonus marker;
- an Income interaction selects a piece composition whose total is defined by
  the active workflow.

## Allocation

| Range | Indices | Capacity | Used | Padding | Permanent interaction |
|---|---:|---:|---:|---:|---|
| `POST` | `0–255` | 256 | 242 | 14 | Post and piece shape |
| `ROUTE` | `256–575` | 320 | 280 | 40 | Route body, endpoint office, or drawn endpoint outcome |
| `INCOME` | `576–591` | 16 | 5 | 11 | Resource composition |
| `BONUS_MARKER` | `592–639` | 48 | 41 | 7 | Owned marker or opponent-used marker |
| `TILE` | `640–655` | 16 | 6 | 10 | Tile choice or Favour response |
| `CITY` | `656–719` | 64 | 52 | 12 | Adjacent-office boundary or green-city/shape |
| `ABILITY` | `720–727` | 8 | 5 | 3 | Player-board ability box |
| `SUPPLY` | `728–729` | 2 | 1 | 1 | Optional same-shape displacement piece source |
| `PLAYER` | `730–735` | 6 | 5 | 1 | Fixed player seat, including Exchange target |
| `CONTROL` | `736–743` | 8 | 2 | 6 | Finish current workflow or end turn |
| `EXPANSION` | `744–767` | 24 | 0 | 24 | Future interaction family |

The 129 reserved slots are distributed inside permanent family boundaries.
Activating padding in one family cannot shift a later family.

Emperor's Favour payment reuses the existing bonus-marker interactions. The
player selects the marker being spent; payment does not create a second set of
tile-family actions for the same physical markers.

## Route interactions

The original route layout is retained inside the first 280 route-family slots:

- local `0–39`: route bodies;
- local `40–119`: two endpoint-office locations per route;
- local `120–279`: four drawn endpoint-outcome locations per route;
- local `280–319`: reserved route capacity.

Route body, office, upgrade, and prestige are separate interactions because the
rules can offer them simultaneously after controlling a route. The player must
distinguish skipping step 3, taking the leftmost office at either endpoint,
developing one of two printed abilities, or taking one of four printed prestige
spaces. These are selectable board locations, not staged abstract verbs.

The route-body slot is also the target during mutually exclusive workflows for:

- Tribute for Establishing a Trading Post;
- Block Trade Route;
- replacement bonus-marker placement.

Both promo-marker rules say to place the marker above a chosen trade route.
Selecting a post is therefore not the rules interaction. Replacement also
selects a route, so a second 40-entry route numbering is unnecessary.

## City interactions

City slots are assigned from static map topology, never from the filtered legal
list:

- local `0–45`: each adjacent pair of standard printed offices, enumerated by
  map city order and left-office position;
- local `46–51`: each `(green city, Trader)` and
  `(green city, Merchant)` location on the Eastern map;
- local `52–63`: reserved city capacity.

Map 2/3 has the maximum 46 printed adjacent-office boundaries and three green
cities, requiring six shape-specific green-city interactions. A city alone is
insufficient for Swap Office because one city can have multiple eligible
adjacent pairs. A shape-neutral green-city slot is also insufficient when both
piece shapes are legally available.

## Control interactions

Two permanent controls are sufficient:

1. `Finish current workflow` ends optional pickup/placement work or declines
   remaining optional displacement placements.
2. `End turn` forgoes remaining optional turn opportunities and advances into
   replacement processing or the next player.

The active workflow makes the first control unambiguous. Optional displacement
piece selection occupies the first `SUPPLY` slot, not a finish control, and
must not share the Additional Trading Post marker slot.

## Compatibility

Version 2 is incompatible with both the superseded version-1 draft and the
earlier semantic-decision version-2 proposal. Existing 620-output checkpoints
cannot load into the 768-output network and are rejected by normal model shape
validation. There is no automatic migration from those checkpoints.
