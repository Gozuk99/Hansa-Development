# AI Observation Design

## Purpose

This document defines the active player-visible AI observation used by the
shared model, headless play, and training.

The governing rule is:

> The AI receives only information that the ordinary player making the
> decision could know at that moment.

The engine's complete state is not the AI input.

## Result

The headless observation builder returns:

```text
features
legal_action_mask[768]
```

The observer is always `game.players[game.active_player]`, and the legal mask
belongs to that same player. GUI and headless AI use this one builder. Drawing
code does not filter private information.

`features` is a fixed 4,241-value `int16` tensor. Its groups are laid out in
this order: `game`, `players`, `cities`, `routes`, `optional_components`, then
`workflow`. A model adapter may cast or normalize the values but must preserve
that layout.

Schema versions and fingerprints are checkpoint/file compatibility metadata.
They are never neural-network features.

The implemented observation contract is version 2. Version 2 hides opponents'
used bonus-marker identities outside the Exchange Bonus Marker workflow. Shared
model checkpoints store its version, 4,241-value size, and fingerprint alongside
the action-schema metadata. Unknown or shape-incompatible schemas are rejected.
The one explicit exception is a version-1 model or training checkpoint: its
same-shaped weights and optimizer may transfer into version 2, and its next save
records version 2. Old observation datasets are not silently reinterpreted.

## Relative players

Player slots rotate for one shared model:

```text
slot 0 = acting player
slot 1 = next occupied seat
slot 2 = next occupied seat
slot 3 = next occupied seat, or padding
slot 4 = next occupied seat, or padding
```

An owner value of `0` means nobody. Values `1` through `5` refer to relative
player slots `0` through `4`. The current normal-turn player is stored as a
relative owner value because displacement and optional income can make another
player the acting player.

Color is a GUI identity, not a strategic feature. Turn order is represented by
the relative slots rather than duplicated as another permutation.

## Stable board slots

Slots never depend on which actions are legal:

- city slot = index in `selected_map.cities`;
- route slot = index in `selected_map.routes`;
- post slot = index in `route.posts`;
- office slot = current left-to-right index in `city.offices`.

The selected map ID distinguishes map-specific catalogues. Tests must lock the
city and route ordering so a map edit cannot silently change trained inputs.
Added green-city offices retain their actual prepended/appended order.

## Concrete feature groups

The first implementation uses the fixed groups below. Boolean fields use
`bool`; IDs and counters use at least `int16`. A model adapter may later convert
them to floating point, but the rules engine does not normalize them.

### Game: one record

| Field | Shape | Meaning |
|---|---:|---|
| `map_id` | 1 | Map 1, 2, or 3 |
| `player_count` | 1 | 3 through 5 |
| `turn_number` | 1 | Current turn counter |
| `round_number` | 1 | Current round counter |
| `current_player` | 1 | Relative normal-turn owner |
| `phase` | 1 | `TurnPhase` ID from the table below |
| `mission_cards_enabled` | 1 | Optional module flag |
| `emperors_favour_enabled` | 1 | Optional module flag |
| `score_end_threshold` | 1 | 20 |
| `full_city_count` | 1 | Visible completed-city count |
| `full_city_threshold` | 1 | Selected-map threshold |
| `replacement_pool_count` | 1 | Number of face-down markers remaining |
| `east_west_completed_count` | 1 | Public Map 1 counter |
| `east_west_completed_players` | 5 | Relative-player flags for players who already scored this connection |
| `bonus_pool_exhausted_during_claim` | 1 | Public end-condition progress from claiming when the supply was empty |
| `pending_tribute_income_owners` | 5 | Relative owners queued to make public Tribute income choices |
| `game_end_pending_immediate_resolution` | 1 | End condition reached while public responses still require resolution |

Do not add a second “ending condition reached” field; it is derived from the
scores, counters, and thresholds above.

### Players: `[5]`

| Field | Shape per player | Meaning |
|---|---:|---|
| `present` | 1 | Real player or padding |
| `score` | 1 | Public score |
| `final_score` | 1 | Zero until public final scoring |
| `general_stock` | 2 | Trader, Merchant counts |
| `personal_supply` | 2 | Trader, Merchant counts |
| `ability_positions` | 5 | Keys, Privilege, Book, Actions, Bank track positions |
| `actions_remaining` | 1 | Public remaining actions |
| `unused_bonus_markers` | 15 | Public face-up marker type IDs, then padding |
| `used_bonus_markers` | 15 | Acting player's exact types; opponents use one hidden-marker ID per face-down marker. The selected Exchange target's exact types become visible while choosing the exchanged marker. |
| `owned_tiles` | 6 | One boolean per Emperor's Favour tile type |
| `map3_privileges` | 3 | Cardiff, Carlisle, London counters |
| `mission_cities` | 3 | Acting player's card only; otherwise zero |
| `mission_visible` | 1 | True only for the acting player's own card |

Office and post ownership are not repeated here; they are already in the board
groups.

### Cities: `[30]`

| Field | Shape per city | Meaning |
|---|---:|---|
| `present` | 1 | Real city or padding |
| `city_type` | 1 | Printed city color/type ID |
| `upgrade_types` | 2 | Printed upgrade IDs in drawn order |
| `tribute_owners` | 4 | Relative owners of the printed Tribute slots |
| `offices` | 10 records | Current offices in left-to-right order |

Each office record contains:

| Field | Shape | Meaning |
|---|---:|---|
| `present` | 1 | Real office or padding |
| `place_adjacent_added` | 1 | Office created by Additional Trading Post |
| `shape` | 1 | Trader or Merchant |
| `privilege` | 1 | Printed privilege color |
| `points` | 1 | Printed point value |
| `owner` | 1 | Relative owner |
| `occupying_shape` | 1 | Actual Trader/Merchant, or none |

City control and city-full status are derived from the ordered offices and are
not duplicated.

Ten offices is a practical model capacity, not a game rule. Observation
creation must fail clearly if a city exceeds ten; it must never truncate.

### Routes: `[40]`

| Field | Shape per route | Meaning |
|---|---:|---|
| `present` | 1 | Real route or padding |
| `endpoint_cities` | 2 | City slot IDs |
| `region` | 1 | None, Scotland, or Wales |
| `route_type` | 1 | Printed route color/type |
| `required_merchants` | 1 | Number of Merchant-only posts |
| `bonus_marker` | 1 | Face-up marker type, or none |
| `permanent_marker` | 1 | Permanent marker type, or none |
| `tribute_owners` | 5 | Visible relative owners, then padding |
| `block_owners` | 5 | Visible relative owners, then padding |
| `posts` | 5 records | Printed post order |

Each post record contains:

| Field | Shape | Meaning |
|---|---:|---|
| `present` | 1 | Real post or padding |
| `required_shape` | 1 | Any shape or Merchant-only |
| `owner` | 1 | Relative owner |
| `occupying_shape` | 1 | Trader, Merchant, or none |

Route completion is derived from posts and is not duplicated.

### Public optional components

| Field | Shape | Meaning |
|---|---:|---|
| `available_tiles` | 6 | Public Emperor's Favour choices |
| `special_prestige` | 4 × 3 | Printed value, privilege, relative owner |
| `pending_replacement_markers` | 15 | Drawn marker types once the player reaches the placement phase; hidden before then |

The six Emperor's Favour effects are derived from public tile ownership rather
than copied into separate owner fields.

## Category IDs

IDs are fixed by the observation implementation, not discovered from a set or
filtered legal list:

```text
piece:      0 none, 1 Trader, 2 Merchant
privilege:  0 none, 1 White, 2 Orange, 3 Pink, 4 Black
region:     0 none, 1 Scotland, 2 Wales
phase:      0 ACTIONS
            1 DISPLACEMENT
            2 MOVE_PIECES
            3 BONUS_MARKER_CHOICE
            4 BUY_TILE_PAYMENT
            5 INCOME_FAVOUR_RESPONSE
            6 TRIBUTE_INCOME_RESPONSE
            7 PLACE_ADJACENT_ROUTE
            8 PERMANENT_ROUTE_PIECE_SELECTION
            9 REPLACE_BONUS_MARKERS
           10 TURN_COMPLETE
           11 GAME_OVER
```

Bonus-marker, permanent-marker, tile, upgrade, city-type, and route-type IDs
must be declared once beside the encoder and protected by exact mapping tests.
They are not schema IDs shown to the AI; they are numeric representations of
physical information the player can see.

## Workflow context

Only visible selections needed to interpret the next action are included. The
fixed workflow record contains:

| Field | Shape | Meaning |
|---|---:|---|
| `held_pieces` | 5 × 3 | Shape, relative owner, region |
| `pickups_remaining` | 1 | Move/bonus-move pickup count |
| `placements_remaining` | 1 | Move/bonus/permanent placement count |
| `displaced_shape` | 1 | Mandatory displaced piece shape |
| `displaced_piece_placed` | 1 | Mandatory-piece progress |
| `optional_displaced_selected` | 1 | Optional-source progress |
| `displacement_remaining` | 1 | Pieces still required/available |
| `original_displacement_route` | 1 | Route slot plus one; zero means none |
| `bonus_workflow` | 1 | Active bonus follow-up ID, or none |
| `selected_target_player` | 1 | Relative owner, or none |
| `selected_tile` | 1 | Pending tile type, or none |
| `first_payment_marker` | 1 | First selected payment marker, or none |
| `permanent_workflow` | 1 | None, route-piece selection, or Britannia placement |
| `pending_route_pieces` | 5 × 3 | Shape, relative owner, region of cleared-route pieces |
| `replacement_count` | 1 | Markers still requiring route placement |

The held-piece fields are phase-independent because Move 3 and Move Any 2 use
`BONUS_MARKER_CHOICE` while pieces are picked up and replaced.

`pending_route_pieces` represents the engine's visible
`pending_route_piece_choices`. Britannia's separate two-piece workflow uses
`permanent_workflow` plus `placements_remaining`; it does not invent a generic
selection count that the engine does not have.

All unused workflow fields are zero. The legal mask supplies destination and
choice availability; internal search lists and legality-helper values are not
features.

`Game.get_legal_actions()` and `ai_action_mask()` evaluate legality for
`active_player` in response phases. The observation builder pairs that same
player's visible information with the returned legal mask.

## Hidden information

The feature groups never contain:

- any Mission Card except the acting player's own card;
- opponents' face-down used bonus-marker identities, except for the selected
  target while resolving Exchange Bonus Marker;
- face-down replacement-marker types or order;
- drawn replacement-marker types before the player reaches marker placement;
- random-generator state or future random results;
- schema versions or fingerprints;
- Python object IDs/references;
- model weights, filenames, or difficulty;
- GUI coordinates or button state;
- save-file metadata or checksums.

Changing another player's Mission Card or shuffling the hidden marker pool
without changing its visible count must not change the observation.

## Numeric and capacity failures

The encoder uses `int16`, so scores and turn counters do not wrap at 255.
Capacity overflow is always an explicit error, never silent truncation.

Practical collection capacities are:

```text
players 5; cities 30; offices/city 10; routes 40; posts/route 5;
bonus markers/player 15 unused + 15 used; tiles 6; held pieces 5;
pending replacement markers 15; route Tribute owners 5; route Block owners 5.
```

## Verification

Tests must prove:

- identical states and acting players produce identical observations;
- consistently rotating seats and ownership references preserves the relative
  representation;
- the legal mask belongs to the acting player;
- all supported maps, player counts, modules, and phases fit the capacities;
- the acting player's Mission Card changes their observation;
- opponents' Mission Cards do not change it;
- changing an opponent's used marker types without changing the visible count
  does not change it outside Exchange;
- the selected Exchange target's used marker types are visible during selection;
- hidden marker types/order do not change it;
- each visible field changes only its documented feature location;
- values do not overflow;
- capacity overflow fails clearly;
- GUI and headless AI receive the same observation.

The previous 4,445-value encoder was replaced by this 4,241-value
player-visible encoder. `HansaNN` consumes this input directly, and model and
training checkpoints store its exact schema identity.
