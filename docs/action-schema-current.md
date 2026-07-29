# Current 620-Entry Action Schema Inventory

## Scope and status

This document inventories the production action schema as it exists before the
planned 768-entry migration. It describes behavior; it does not define the
future schema.

- Current action-space size: **620**
- Valid indices: **`0–619`**
- Primary constants: `ai/action_options.py` and `map_data/constants.py`
- Authoritative public boundary: `Game.legal_action_mask()` and
  `Game.apply_action()` in `game/game_info.py`
- Inventory date: 2026-07-29

No index is explicitly designated as permanently reserved. Capacity-based
entries that do not correspond to an object or valid option in the current
state remain masked.

## Top-level allocation

| Absolute range | Count | Current family | Mask function | Dispatch function |
|---|---:|---|---|---|
| `0–241` | 242 | Post selection/placement, by requested shape | `mask_post_action` | `map_claim_post_action` |
| `242–521` | 280 | Route completion outcomes | `mask_claim_route` | `map_claim_route_action` |
| `522–526` | 5 | Income and contextual piece compositions | `mask_income_actions` | `map_income_action` |
| `527–534` | 8 | Bonus-marker activation or exchanged-marker selection | `mask_bm` | `map_bm_action` |
| `535–542` | 8 | Emperor tile, marker payment, or Income Favour response | `mask_buy_tile` | `map_buy_tile_action` |
| `543–582` | 40 | Replacement bonus-marker route target | `mask_replace_bm` | `map_replace_bm_action` |
| `583–612` | 30 | Contextual player, office-pair, or green-city choice | `mask_bm_city_actions` | `map_bm_city_actions` |
| `613–617` | 5 | Bonus-marker ability upgrade | `mask_bm_upgrade_ability` | `map_bm_upgrade_ability` |
| `618` | 1 | Contextual finish/end operation | `mask_end_turn` | `map_end_turn_action` |
| `619` | 1 | Additional Trading Post activation or displacement source choice | `mask_place_adjacent` | phase branch in `_perform_action_from_index` |

`masking_out_invalid_actions()` concatenates these ten tensors in the table's
order and then calls `restrict_mask_to_turn_phase()`.

## Detailed index inventory

### `0–241`: post actions

The map's posts are flattened in route-list order and then post-list order.
`MAX_POSTS` is 121.

| Absolute range | Encoding |
|---|---|
| `0–120` | Flattened post `index`, requested shape `square`/Trader |
| `121–241` | Flattened post `index - 121`, requested shape `circle`/Merchant |

The same indices have several phase/state-dependent meanings:

- normal empty post: place the requested shape from Personal Supply;
- normal opponent-owned post: displace it using the requested replacement
  shape;
- normal own occupied post: begin or continue a Move pickup;
- normal held-piece state: place the next held piece on an empty post;
- displacement response: place a displaced or optional piece;
- displacement board fallback: pick up one of the displaced player's pieces
  already on the board, then later place that held piece;
- Move 3: pick up an opponent piece or place the next held piece;
- Move Any 2: pick up an own/opponent piece or place the next held piece;
- Eastern `Place 2 from route`: place the next selected route piece;
- Britannia `Place 2`: place the next held piece in Wales or Scotland;
- Tribute Trading Post: selecting the first post of a route selects that route
  as the Tribute target;
- Block Trade Route: selecting the first post of a route selects that route as
  the Block target.

Contextual duplication exists during several pickup workflows. For an occupied
post, both its Trader and Merchant indices may be enabled even though the
piece's actual shape is already known from the board. Dispatch ignores the
requested shape for the pickup itself.

Capacity behavior:

- maps contain 98–121 posts in supported configurations;
- entries beyond the selected map's flattened post count are unreachable and
  remain masked;
- the 4–5-player Britannia board uses all 121 post positions, so no post
  position is globally reserved across all supported maps.

Relevant source:

- `map_data/constants.py`: `MAX_POSTS = 121`
- `ai/action_options.py`: `mask_post_action`,
  `map_claim_post_action`, `check_if_any_post_BM_flag_set`
- `game/game_actions.py`: post, move, displacement, Tribute, and Block
  mutations used by dispatch

### `242–521`: route outcomes

The map's routes are addressed in route-list order. `MAX_ROUTES` is 40.

#### `242–281`: complete route without step-3 benefit

```text
index = 242 + route_index
```

Meaning: complete the controlled route for endpoint scoring and marker
handling, but claim no office, ability upgrade, or special-prestige space.

#### `282–361`: claim an office

```text
index = 282 + route_index * 2 + endpoint_city_index
endpoint_city_index ∈ {0, 1}
```

Meaning: complete the route and establish the next office in the selected
endpoint city.

#### `362–521`: four route-choice slots

```text
index = 362 + route_index * 4 + route_choice
route_choice ∈ {0, 1, 2, 3}
```

This range has three contextual interpretations:

1. Ordinary upgrade city:
   `route_choice = endpoint_city_index * 2 + upgrade_slot`; complete the route
   and develop that endpoint's selected ability.
2. Special-prestige route:
   `route_choice` selects prestige value `(7, 8, 9, 11)`. The usual endpoint
   and upgrade-slot interpretation is not used.
3. Pending Additional Trading Post:
   `route_choice = endpoint_city_index * 2 + shape_index`, where shape index 0
   is Trader and 1 is Merchant. It creates an additional office in that city
   using the selected route-piece shape.

This is significant contextual index reuse: the same absolute route-choice
index can mean an ability, a special-prestige value, or an additional office.

Capacity behavior:

- supported maps contain 32–40 routes;
- entries for route positions beyond the selected map's route count are
  unreachable and masked;
- the 4–5-player Britannia board uses all 40 route positions;
- route-choice slots without a corresponding static upgrade are masked during
  ordinary route completion, but some of those slots are reused by special
  prestige or Additional Trading Post;
- there is no explicit reserved subrange.

Relevant source:

- `map_data/constants.py`: `MAX_ROUTES = 40`
- `ai/action_options.py`: `mask_claim_route`,
  `map_claim_route_action`
- `game/game_actions.py`: `claim_route_for_points`,
  `claim_route_for_office`, `claim_route_for_upgrade`,
  `claim_route_for_additional_office`

### `522–526`: income and piece-composition choices

The local choice is `index - 522`, from 0 through 4.

| Context | Meaning |
|---|---|
| Normal Income | Requested Merchant count `0–4`; remaining Bank capacity is filled with available Traders |
| Eastern permanent `Place 2 from route` | Merchant count among exactly two selected pieces; only choices `0–2` can be legal |
| Britannia permanent `Place 2` | Merchant count among exactly two pieces drawn using General Stock, then Personal Supply, then board priority; only `0–2` can be legal |
| Tribute income response | Merchant count in the up-to-two-piece income; upper legal choice depends on available stock |

Indices `525–526` are unused in the exact-two-piece contexts but remain active
normal Income choices. None is globally reserved.

Relevant source:

- `ai/action_options.py`: `mask_income_actions`, `map_income_action`
- `drawing/action_ui.py`: `_income_action_label`
- `drawing/drawing_utils.py`: `draw_context_action_buttons`

### `527–534`: bonus-marker type choices

| Index | Type |
|---:|---|
| `527` | `SwapOffice` |
| `528` | `Move3` |
| `529` | `UpgradeAbility` |
| `530` | `3Actions` |
| `531` | `4Actions` |
| `532` | `ExchangeBonusMarker` |
| `533` | `Tribute4EstablishingTP` |
| `534` | `BlockTradeRoute` |

Ordinarily these indices activate an unused marker of that type. During the
second half of Exchange Bonus Marker, after a target player is selected, the
same indices select a used marker type to take from that player.

Relevant source:

- `ai/action_options.py`: `mask_bm`, `map_bm_action`
- `drawing/action_ui.py`: `BONUS_MARKER_NAMES`, `action_label`

### `535–542`: tiles, payments, and Income Favour

The local choice is `index - 535`.

Initial Emperor tile purchase:

| Index | Tile |
|---:|---|
| `535` | `DisplaceAnywhere` |
| `536` | `+1Action` |
| `537` | `+1IncomeIfOthersIncome` |
| `538` | `+1DisplacedPiece` |
| `539` | `+4PtsPerOwnedCity` |
| `540` | `+7PtsPerCompletedAbility` |
| `541–542` | No initial tile mapping |

When choosing unused bonus markers as payment, all eight entries instead map
to the same marker-type order as `527–534`.

During an Income Favour response:

| Index | Meaning |
|---:|---|
| `535` | take one Trader |
| `536` | take one Merchant |
| `537` | decline |
| `538–542` | masked/unreachable in this context |

Relevant source:

- `ai/action_options.py`: `mask_buy_tile`, `map_buy_tile_action`
- `drawing/action_ui.py`: `_tile_action_label`
- `drawing/game_window.py`: `tile_actions`

### `543–582`: replacement-marker route target

```text
index = 543 + route_index
```

The range selects one of up to 40 routes on which to place a pending standard
bonus marker. Entries beyond the selected map's route count are unreachable.
All 40 positions exist on the largest supported Britannia board, so none is
globally reserved.

Relevant source:

- `ai/action_options.py`: `mask_replace_bm`,
  `map_replace_bm_action`
- `drawing/action_ui.py`: `action_label`
- `drawing/game_window.py`: post-click fallback computes `543 + route_index`

### `583–612`: contextual player/city/office choice

The local ordinal is `index - 583`, from 0 through 29.

| Context | Meaning of local ordinal |
|---|---|
| Exchange Bonus Marker, before target chosen | player-list index |
| Swap Office | ordinal in the freshly enumerated list of all eligible `(city, adjacent office pair)` choices |
| Eastern green city | ordinal in the freshly enumerated list of eligible `(dark-green city, shape)` choices |

This range does not always identify the same persistent city. Swap and green
choices are compacted eligible-choice ordinals whose meaning depends on the
current state and enumeration order.

`map_bm_city_actions` contains a trailing city-index loop for green-city
handling, but the earlier `waiting_for_bm_green_city` branch always returns.
That trailing branch therefore appears unreachable in the current dispatcher.
This is documented as an observation, not removed in Milestone 1.

Capacity behavior:

- `MAX_CITIES` is 30 and the largest map has 30 cities;
- player-target entries use only the active player-list prefix;
- eligible swap/green choice entries use only the generated choice-list
  prefix;
- unused suffix entries are masked, but no suffix is formally reserved.

Relevant source:

- `map_data/constants.py`: `MAX_CITIES = 30`
- `ai/action_options.py`: `mask_bm_city_actions`,
  `map_bm_city_actions`
- `drawing/action_ui.py`: `_city_context_label`

### `613–617`: ability upgrade choice

```text
index = 613 + upgrade_city_list_index
```

During `UpgradeAbility`, the five entries correspond to the selected map's
`upgrade_cities` list order. The currently supported maps each expose five
entries. Fully developed abilities are masked.

Relevant source:

- `ai/action_options.py`: `mask_bm_upgrade_ability`,
  `map_bm_upgrade_ability`
- `drawing/action_ui.py`: `action_label`

## Complete trace of index `618`

### Mask generation

`mask_end_turn()` creates the one-entry family later concatenated at absolute
index 618. It enables the entry in these contexts:

- Move 3 or Move Any 2 may stop picking up early, provided no held pieces
  remain;
- displacement may finish after the mandatory displaced piece is placed and
  no held fallback piece remains;
- ordinary turn completion/replacement workflows may confirm ending the turn
  or forgoing optional bonus-marker use, subject to the current state flags.

`restrict_mask_to_turn_phase()` permits 618 during:

- `DISPLACEMENT`;
- `BONUS_MARKER_CHOICE`;
- `REPLACE_BONUS_MARKERS`;
- `TURN_COMPLETE`.

It is excluded from Move Pieces, payment, Income Favour, Tribute response,
Additional Trading Post route selection, permanent route-piece selection, and
Game Over phases.

### Dispatch

`_perform_action_from_index()` sends 618 to `map_end_turn_action()`.
`map_end_turn_action()` branches in this order:

1. finish the pickup portion of Move 3 or Move Any 2;
2. finish displacement via `finish_displacement()`;
3. otherwise enter/continue ordinary end-turn and pending marker-replacement
   behavior.

Thus 618 currently combines at least three structured meanings:

- finish optional marker movement;
- finish displacement/decline remaining optional displacement pieces;
- end or confirm ending the current turn.

### Other dependencies

- `drawing/action_ui.py`: contextual label
  `"Finish displacement (decline optional pieces)"` or `"Finish / End turn"`;
- `drawing/drawing_utils.py`: draws the clickable end/finish button whenever
  618 is legal;
- `drawing/game_window.py`: keyboard `E` applies 618 whenever legal;
- `game/game_runner.py`: deterministic policy always prioritizes 618;
- tests refer directly to 618 in `test_core_actions.py`,
  `test_drawing.py`, `test_eastern_hanseatic.py`,
  `test_game_configuration.py`, `test_standard_bonus_markers.py`, and
  `test_turn_structure.py`;
- `docs/DRAWING_ARCHITECTURE.md` and `docs/RULES_COMPLIANCE_MATRIX.md`
  describe this index directly.

## Complete trace of index `619`

### Mask generation

`mask_place_adjacent()` creates the one-entry family concatenated at absolute
index 619.

During `TurnPhase.DISPLACEMENT`, it means:

- explicitly select an optional General Stock or Personal Supply piece whose
  shape matches the still-unplaced mandatory displaced piece;
- it is exposed only while more than one placement remains, the mandatory
  piece has not been placed, no optional-source selection is already pending,
  and the priority source contains the matching shape.

During `TurnPhase.ACTIONS`, it means:

- activate an unused `PlaceAdjacent`/Additional Trading Post marker when at
  least one controlled route has a legal additional-office outcome.

`restrict_mask_to_turn_phase()` permits 619 only in the displacement range
`(618, 620)` or by returning the unrestricted mask during ordinary Actions.

### Dispatch

`_perform_action_from_index()` branches on phase:

- displacement: `select_optional_displaced_shape(game)`;
- every other permitted context: `map_place_adjacent_action(game)`.

The displacement branch constrains the subsequent post mask to the displaced
shape. The Additional Trading Post branch sets
`waiting_for_bm_place_adjacent`; the later city/shape choice is encoded in
`362–521`.

### Other dependencies

- `drawing/action_ui.py`: contextual displacement or Additional Trading Post
  label;
- the GUI action browser makes any legal index, including 619, clickable;
- tests refer directly to 619 in `test_core_actions.py`,
  `test_eastern_hanseatic.py`, and `test_standard_bonus_markers.py`;
- `docs/RULES_COMPLIANCE_MATRIX.md` describes both uses.

Index 619 is the clearest current example of unrelated semantic reuse and must
be split in the future schema. Milestone 1 does not change it.

## Turn-phase filtering

`restrict_mask_to_turn_phase()` applies these hard-coded absolute ranges:

| Phase | Permitted half-open ranges |
|---|---|
| `ACTIONS` | no additional restriction |
| `DISPLACEMENT` | `[0,242)`, `[618,620)` |
| `MOVE_PIECES` | `[0,242)` |
| `BONUS_MARKER_CHOICE` | `[0,242)`, `[527,535)`, `[583,613)`, `[613,618)`, `[618,619)` |
| `BUY_TILE_PAYMENT` | `[535,543)` |
| `INCOME_FAVOUR_RESPONSE` | `[535,543)` |
| `TRIBUTE_INCOME_RESPONSE` | `[522,527)` |
| `PLACE_ADJACENT_ROUTE` | `[362,522)` |
| `PERMANENT_ROUTE_PIECE_SELECTION` | `[522,527)` |
| `REPLACE_BONUS_MARKERS` | `[543,583)`, `[618,619)` |
| `TURN_COMPLETE` | `[527,535)`, `[618,619)` |
| `GAME_OVER` | none |

These ranges are a second action-layout definition independent of the
top-level family constants.

## Mask-generation and legality locations

Production mask construction is concentrated in `ai/action_options.py`:

- `masking_out_invalid_actions`
- `restrict_mask_to_turn_phase`
- `mask_post_action`
- `mask_claim_route`
- `mask_income_actions`
- `mask_bm`
- `mask_buy_tile`
- `mask_replace_bm`
- `mask_bm_city_actions`
- `mask_bm_upgrade_ability`
- `mask_end_turn`
- `mask_place_adjacent`

Supporting legality helpers are spread across:

- `game/game_actions.py`: especially post, route, displacement, bonus-marker,
  tile, and regional legality helpers;
- `game/game_info.py`: `Game.turn_phase`, `Game.legal_action_mask`,
  `Game.apply_action`;
- `game/turn_state.py`: `TurnPhase`;
- `player_info/player_attributes.py`: supply, ability, action, movement, and
  displacement state;
- `map_data/map_attributes.py`: route, city, office, post, upgrade, and marker
  predicates.

The current mask partly calls shared engine predicates and partly recreates
legality directly. There is no engine method returning structured legal
actions in the current schema.

## Decoding and dispatch locations

The top-level decoder is `_perform_action_from_index()` in
`ai/action_options.py`. It uses cumulative family sizes, followed by literal
branches for 618 and 619.

Family decoders in that file are:

- `map_claim_post_action`
- `map_claim_route_action`
- `map_income_action`
- `map_bm_action`
- `map_buy_tile_action`
- `map_replace_bm_action`
- `map_bm_city_actions`
- `map_bm_upgrade_ability`
- `map_end_turn_action`
- `map_place_adjacent_action`

`Game.apply_action()` in `game/game_info.py` validates the integer type,
checks `0 <= index < TOTAL_ACTIONS`, regenerates the legal mask, rejects a
masked index, and invokes the decoder.

## GUI interpretations

- `drawing/action_ui.py`
  - `action_label` contains the top-level numeric range interpretation;
  - `_route_action_label`, `_income_action_label`, `_tile_action_label`, and
    `_city_context_label` reinterpret indices from current state.
- `drawing/game_window.py`
  - `GameWindow.legal_actions` reads nonzero mask indices;
  - `GameWindow.action_for_click` computes post indices using `121`, office
    indices using `242 + MAX_ROUTES`, points-only route indices using `242`,
    replacement routes using `543`, and fixed tile indices `535–540`;
  - `_upgrade_action_for_click` maps drawn upgrade boxes back to route-choice
    indices;
  - keyboard `E` invokes 618;
  - `choose_ai_action` indexes the model's output tensor with legal indices.
- `drawing/drawing_utils.py`
  - `redraw_window` treats 618 as the end/finish button;
  - `draw_context_action_buttons` displays legal choices in `522–526`.

The GUI therefore contains independent numeric knowledge that must be migrated
with the codec in later milestones.

## Hard-coded sizes, offsets, and ranges

### Production code

| File | Hard-coded dependency |
|---|---|
| `map_data/constants.py` | `OUTPUT_SIZE = 620`, `MAX_POSTS = 121`, `MAX_ROUTES = 40`, `MAX_CITIES = 30` |
| `ai/action_options.py` | family sizes `242, 280, 5, 8, 8, 40, 30, 5, 1, 1`; post offset 121; family endpoints 242/522/527/535/543/583/613/618/619/620; route factors 2 and 4; phase ranges listed above |
| `game/game_info.py` | range check against imported `TOTAL_ACTIONS` |
| `game/game_runner.py` | exact mask length 620; post modulo 121; policy ranges `0–241`, `242–521`, `522–526`; literal 618; route decoding boundaries 282 and 362 |
| `drawing/action_ui.py` | boundaries 121, 242, 522, 527, 535, 543, 583, 613, 618, 619 and family-local offsets |
| `drawing/game_window.py` | post offset 121; route base 242; replacement base 543; fixed tile indices 535–540; literal 618 |
| `drawing/drawing_utils.py` | literal 618; contextual range `522–526` |

`ai/ai_model.py` does not contain an active literal output size; `HansaNN`
accepts `output_size`. Its stale comments say 616. Callers supply the current
620 through `OUTPUT_SIZE`.

### Tests with schema-number coupling

- `tests/test_legal_actions.py`: upper bound 620;
- `tests/test_turn_structure.py`: out-of-range 620 and direct 618 references;
- `tests/test_core_actions.py`: direct 618/619 and family slices;
- `tests/test_drawing.py`: `522–526` and 618;
- `tests/test_game_configuration.py`: direct family bases, post offset, and
  618. The values 620 in its display-size test are viewport dimensions, not
  action-space dependencies;
- `tests/test_eastern_hanseatic.py`: constants 618 and 619;
- `tests/test_standard_bonus_markers.py`: constants 618 and 619 and range end
  618;
- other action tests define local family bases such as 242, 522, 527, 535,
  543, 583, and 613.

### Documentation with stale or direct schema numbers

- `docs/RULES_COMPLIANCE_MATRIX.md` correctly states a 620-entry mask and
  directly documents 618/619;
- `docs/DRAWING_ARCHITECTURE.md` directly documents 618;
- `docs/REPOSITORY_ASSESSMENT.md` and `docs/FIRST_MILESTONE_PLAN.md` still
  describe the older 619-entry layout that ended at 618. These are historical
  documents, not production definitions.

## Models, checkpoints, saves, histories, and replay dependencies

### Neural-network output and checkpoints

- `map_data/constants.py`: `OUTPUT_SIZE = 620`;
- `player_info/player_attributes.py`: constructs `HansaNN(INPUT_SIZE,
  OUTPUT_SIZE)` for legacy direct player loading;
- `game/game_config.py`: `_load_ai_model` constructs the same model for GUI AI
  seats;
- `ai/ai_model.py`: `layer3 = nn.Linear(1024, output_size)`;
- model files are raw PyTorch `state_dict` objects loaded with `torch.load`
  and saved with `torch.save`;
- no action-space size or schema version metadata is stored;
- a checkpoint with a different final-layer shape fails through PyTorch's
  state-dict shape validation. `GameConfiguration` catches that runtime error
  and silently starts that AI seat with a new model after printing a warning;
  direct `Player(load_model=True)` does not add that fallback.

### Saves and difficult-state snapshots

- `BoardData.save_game_state_JSON` and load helpers in `ai/game_state.py`
  serialize mutable game state but do not store action indices, action-space
  size, or schema version;
- `game_state_JSON.json` and JSON files under `training_data/` use that
  unversioned state shape;
- these files therefore depend indirectly on action semantics only when a
  caller computes or applies an action after loading;
- the existing repository assessment identifies those saves as incomplete and
  not exact checkpoints.

### History and replay

- `game/game_runner.py` records raw integer indices in
  `GameRunResult.action_trace`;
- `tests/test_complete_game.py` compares those tuples for deterministic runs;
- the trace has no schema version and is not persisted by the runner;
- no separate production replay loader, event history, serialized training
  sample pipeline, or replay buffer was found;
- the README explicitly lists exact save/load and replayable history as future
  work.

Changing index meanings would therefore invalidate raw traces even though
there is currently no persisted replay format enforcing compatibility.

## Currently unused, unreachable, and reserved entries

- **Explicitly reserved indices:** none.
- **Out-of-map capacity:** post, route, replacement-route, and city-family
  suffixes are masked when the selected map has fewer than the maximum object
  count.
- **Contextually unused entries:** `525–526` for exact-two-piece selection,
  `541–542` for initial tile buying, and `538–542` for Income Favour.
- **Invalid route subslots:** ordinary route upgrade slots without a matching
  endpoint upgrade are masked. Their absolute indices may still acquire a
  different meaning during special prestige or Additional Trading Post.
- **Compacted-choice suffixes:** unused player, swap-pair, and green-city
  ordinals in `583–612` are masked.
- **Apparently unreachable code path:** the trailing green-city city-index
  branch of `map_bm_city_actions`, after its earlier green-choice return.

Because several families are contextual and depend on dynamically enumerated
choices, this inventory does not claim that every nominally active index is
reachable in an actual legal game. Proving per-index reachability requires the
later exhaustive audit milestone. No uncertainty has been resolved by
guessing.

## Supported-map capacity evidence

Construction of every supported map/player-count combination reports:

| Map | Players | Cities | Routes | Posts | Upgrade entries |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 27 | 32 | 101 | 5 |
| 1 | 4 | 27 | 34 | 107 | 5 |
| 1 | 5 | 27 | 34 | 107 | 5 |
| 2 | 3–5 | 28 | 32 | 98 | 5 |
| 3 | 3 | 26 | 35 | 111 | 5 |
| 3 | 4–5 | 30 | 40 | 121 | 5 |

These counts explain why the current maxima are 30 cities, 40 routes, and 121
posts. They do not prove that every contextual choice slot can become legal.

## Inventory conclusions for the next milestone

The current layout is exactly 620 entries, but it is not a one-meaning-per-index
schema. Contextual reuse occurs throughout the post, route-choice, income,
bonus-marker, tile/payment, city-choice, 618, and 619 families. Numeric layout
knowledge is duplicated across masking, dispatch, the runner, GUI, tests, and
documentation. Checkpoints and integer action traces are unversioned.

Milestone 2 must use this inventory to define—but not yet activate—a
non-overlapping 768-entry versioned schema with explicit reserved capacity.
