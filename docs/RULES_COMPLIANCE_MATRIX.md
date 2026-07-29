# Hansa Teutonica Rules Compliance Matrix

## Purpose

This document maps the supplied rules to the current game engine, action-index dispatcher, legal-action masks, and automated tests.

It is an audit, not a declaration that the implementation is correct. A rule is only considered verified when a focused test constructs the relevant position, checks which actions are legal, applies the action, and checks every required state change.

## Project Scope

The project supports **3–5 players only**.

All two-player rules are explicitly out of scope, including:

- The two-player pawn
- Province movement and borders
- Pawn-based action restrictions
- Two-player setup questions in the FAQ

The FAQ retains its two-player section as source material, but those rules must not create engine requirements, action-space entries, tests, or development tasks.

## Sources

- [Hansa Teutonica Big Box Rulebook](hansa-teutonica-big-box-rulebook.md)
- [Hansa Teutonica FAQ v5](Hansa_Teutonica_FAQ_v5.md)
- [Promo Bonus Markers](BigBoxPromoBonusMarkers.md)

The copied rulebook may contain source or OCR errors. Where a value looks questionable, this matrix records the conflict rather than choosing an interpretation silently.

## Status Legend

| Status | Meaning |
| --- | --- |
| **Implemented** | A clear implementation exists and matches the reviewed rule at a high level. It may still need more edge-case tests. |
| **Partial** | Some of the rule exists, but choices, edge cases, ordering, or validation are incomplete. |
| **Conflict** | Current behavior appears to contradict the supplied rules or another part of the implementation. |
| **Missing** | No meaningful implementation was found. |
| **Unverified** | Code exists, but current tests do not establish rule correctness. |

## Current Test Baseline

The current tests establish only that:

- A model-free game can be constructed.
- Seeded setup is repeatable for the tested cases.
- A fresh game exposes at least one action in a 620-entry mask.
- A deterministic baseline can finish selected three-player games on maps 1 and 2.

They do **not** yet prove that individual actions, scoring, expansions, or complete games follow all rules.

Existing tests:

- `tests/test_game_setup.py`
- `tests/test_legal_actions.py`
- `tests/test_complete_game.py`

## Core Setup and Turn Structure

| Rule | Rule source | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- | --- |
| Support 3–5 players | Game Setup | `game.setup.validate_game_configuration`; `Game.__init__`, `Game.create_players` | N/A | All supported player counts construct; invalid counts are rejected | **Implemented** | The supported range is centralized and explicitly excludes two-player play. Complete-game verification remains a separate concern. |
| Select the correct board layout for player count | Base and expansion setup | `Game.assign_map`; player-count branches in `Map1` and `Map3` | N/A | Exact city/route topology for all nine map/player-count combinations | **Implemented** | Map 1 and Britannia select their 3-player versus 4–5-player geometry; the Eastern map has one layout for all supported counts. |
| Starting personal supply and general stock depend on player order | Game Setup | `game.setup.starting_inventory`; `Player.__init__` | N/A | Exact inventories for player orders 1–5 and game sizes 3–5 | **Implemented** | Each player begins with 11 available traders split by order and one merchant in personal supply. |
| Cover ability spaces with 15 traders and 3 merchants; begin score at zero | Player Writing Desk setup | Ability indices; `locked_ability_traders`; `locked_ability_merchants`; `score` | N/A | Complete physical totals of 27 traders and 4 merchants for every player order | **Implemented** | Locked pieces are derived from exposed ability spaces. The remaining trader is represented by the zero-valued score field rather than as a separate board object. |
| Starting abilities | Player Writing Desk | `Player.__init__`; upgrade methods and unlock helpers | Route-upgrade and bonus-marker masks | Exact starting values and complete progression tests for all five tracks | **Implemented** | Keys `1,2,2,3,4`; Actions `2,3,3,4,4,5`; Privilege white through black; Book `2,3,4,5`; Bank `3,4,7,C`. Upgrades release the correct trader or merchant, repeated Action values do not grant an extra current-turn action, and maximum spaces cannot be upgraded again. The engine uses `50` internally for Bank `C` (complete/all). |
| Starting bonus markers and supply | Game Setup | `Map.assign_bm_pool_default`, `Map.assign_starting_bonus_markers` | Replacement marker mask | Exact starting-marker types and supply counts on every map | **Implemented** | Every map starts with Move 3, Swap Office, and Place Adjacent on routes, plus the 12-marker supply. |
| Permanent bonus markers are assigned by map and player count | Eastern and Britannia setup | Permanent-marker routes in `Map2` and `Map3` | Permanent-marker route workflow | Exact route/type fixtures for Eastern and both Britannia layouts | **Implemented** | Eastern has four permanent markers. Britannia has two on the 4–5-player board and the same two plus Carlisle–Isle of Man on the 3-player board. |
| Optionally deal one secret Mission card on map 1 | Mission Cards optional module | `use_mission_cards`; seeded `Map1.mission_cards`; `assign_mission_cards` | N/A | Disabled by default; unique three-city cards for 3–5 players when enabled; rejected on other maps | **Implemented** | Mission Cards are optional and map-1-only. When enabled, the nine cards are shuffled, one is dealt per player, and unused cards remain undealt. |
| Optionally select Emperor’s Favour tiles on any map | Emperor’s Favour optional module | `use_emperors_favour`; `Game.initialize_tile_pool` | Tile purchase mask | Disabled by default; every map/player count checks allowed types, exact count, and uniqueness when enabled | **Implemented** | Emperor’s Favour is optional on every supported map. When enabled, the display is a seeded random subset of the six unique tiles equal to player count. |
| Begin with an empty board and zeroed public counters | Game Setup | `Game.__init__`; map/city/route constructors | N/A | Every map/player-count combination checks posts, offices, scores, completed cities, East–West count, and pending replacements | **Implemented** | Completed Cities begins at zero, no trading posts or routes are occupied, every score is zero, and no end condition or replacement obligation is active. |
| Turn order and actions per turn | Game Play | `Game.advance_turn`; `Player.start_turn`, `spend_action`, `grant_actions`, `forfeit_remaining_actions` | `map_end_turn_action`, `mask_end_turn` | Direct action-count, switching, round numbering, early-end rejection, optional marker forfeit, and +1 Action tile tests | **Implemented** | Action counts cannot go negative. Turn advancement and the next player’s allowance are initialized in one place. |
| Finish one action before beginning another | Game Play | `Game.apply_action`; `Game.turn_phase`; pending workflow state | Authoritative mask validation plus `restrict_mask_to_turn_phase` | Rejected masked/out-of-range actions; phase-exclusive masks; advancement guards; sequential end-of-turn obligations | **Implemented** | The supported mutation boundary validates every action against the current legal mask before dispatch. Displacement, movement, tile payment, marker choices, and marker replacement restrict legal actions to their own workflow. Low-level functions are implementation details. |
| Deterministic setup | Development requirement | `Game(seed=...)` owns the RNG passed into map, marker, mission-card, and tile setup | N/A | Same-seed setup across all maps; global RNG isolation | **Implemented** | Setup randomness no longer temporarily reseeds or consumes Python’s global random generator. Runtime stochastic policies use their own seeded generator. |

## Core Actions

| Rule | Rule source | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- | --- |
| A) Income up to current Bank capacity | Game Play A | `Player.income_action`; `map_income_action` | `mask_income_actions` | Every Bank value, mixed shapes, insufficient stock, over-capacity rejection, and conservation | **Implemented** | `C` means all and is represented internally by `50`. The indexed action chooses a merchant count and automatically fills the remaining capacity with traders, producing a strategically equivalent maximal transfer. |
| B) Place one tradesman from personal supply on a free point | Game Play B | `claim_post_action` | Post indices `0–241`; `mask_post_action` | Action/supply cost, required merchant point, conservation, and Britannia permissions | **Implemented** | Only an empty compatible point is exposed; one personal-supply piece and one action are consumed. |
| C) Displace an opposing trader for one extra piece | Game Play C | `displace_action`, `displace_claim`, `displace_to` | Post mask and displacement phase | Exact actor cost and two-piece opponent relocation | **Implemented** | The replacing piece comes from personal supply and one additional payment returns to general stock. |
| C) Displace an opposing merchant for two extra pieces | Game Play C | Same as above | Same as above | Exact three-piece cost and relocation count | **Implemented** | Merchant displacement requires the replacing piece plus two additional personal-supply pieces. |
| Displaced pieces go to nearest available adjacent routes | Game Play C | `gather_empty_adjacent_posts`, `get_adjacent_routes` | Displacement phase in `mask_post_action` | Nearest-route restriction, required shape, general-stock, personal-supply, and board fallback | **Implemented** | The displaced piece is mandatory. Each placement stops at the nearest route distance with a compatible empty post. Optional pieces may use any shape available from the current rule-ordered source that fits a nearest post; a board-fallback piece uses its own shape and restarts the search from the original route. |
| Displaced player may decline additional pieces | Game Play C | `finish_displacement` | Contextual action index `618` during displacement | Decline after mandatory displaced-piece placement | **Implemented** | The displaced piece must be replaced; any still-unclaimed extras may then be declined without skipping held pieces. |
| D) Move 2–5 own tradesmen according to Book value | Game Play D | `Player.start_move`, `pick_up_piece`, `place_piece`, `move_action` | Post mask and move phase | Book pickup cap, early completion, placement, and one-action cost | **Implemented** | A player may move up to the exposed Book value; all picked-up pieces must be placed before the action ends. |
| D) Swap one own trader and merchant | Game Play D | General move workflow | Post mask and move phase | Focused trader/merchant position swap | **Implemented** | Both pieces are picked up and placed into each other’s vacated points within one Move action. |
| Normal moves cannot displace opponents | Game Play D | `move_action` | Move-phase post mask | Opponent pickup and occupied-target rejection | **Implemented** | Normal movement only picks up the active player’s pieces and only places onto empty compatible points. |

## Creating a Trade Route

| Rule | Rule source | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- | --- |
| Route can be created only when every point is occupied by the active player | Game Play E | `Route.is_controlled_by`; route-claim functions | `mask_claim_route` | Empty, mixed, and fully controlled route masks | **Implemented** | Route creation is explicit; merely filling a route does not create it. |
| Step 1: score controllers of both endpoint cities | Game Play E.1 | `score_route`; `City.get_controller` | Performed by every route alternative | Exact endpoint increments and rightmost-office tie | **Implemented** | Each endpoint controller gains one point before the selected step-3 result is resolved. |
| Step 2: take a route bonus marker | Game Play E.2 | `handle_bonus_marker`, `finalize_route_claim` | Route actions plus replacement phase | Ownership, immediate replacement draw, and empty-supply trigger | **Implemented** | The acquired marker becomes usable after the route action. Its replacement is drawn immediately; inability to draw triggers game end only after the action completes. |
| Replacement marker must be placed at end of turn | Game Play E.2 | `pending_bonus_markers`; `assign_new_bonus_marker_on_route` | Indices `543–582`; replacement phase | All three placement restrictions and pending-marker identity | **Implemented** | Placement follows optional +3/+4 actions and cannot target marked, occupied, or fully closed routes. |
| Step 3 offers exactly one of office, ability, or special prestige | Game Play E.3 | Separate route-claim functions | Fixed route action alternatives | Simultaneous legal alternatives followed by single-result invalidation | **Implemented** | Points-only represents skipping step 3; applying any one alternative clears the route and makes every other alternative unavailable. |
| 3a) Establish the leftmost vacant trading post | Game Play E.3a | `City.update_next_open_office_ownership`; `claim_route_for_office` | Office route actions | Leftmost placement, route-piece origin, and stock return | **Implemented** | The office receives a matching piece from the completed route; no vacant office can be skipped. |
| Office requires matching trader/merchant shape | Game Play E.3a | `City.has_required_piece_shape`; route reset functions | `mask_claim_route` | Positive matches plus circle-only→square and square-only→circle rejection | **Implemented** | A square office requires a trader on the completed route; a round office requires a merchant. Any number of the opposite shape cannot substitute. |
| Office color requires sufficient Privilege | Game Play E.3a | `Player.player_can_claim_office` | `mask_claim_route` | Insufficient and sufficient privilege thresholds | **Implemented** | White, orange, pink, and black follow the exposed Privilege progression. |
| Gold-coin point is awarded only when establishing the marked office | Rulebook and FAQ | `Office.awards_points`; `City.update_next_open_office_ownership` | Office route action | Establishment award and no-award office swap regression | **Implemented** | Printed coin points are granted once when that office is first established, never when controllers later swap. |
| Completing a city advances the Completed Cities count | Game Play E.3a | `Game.check_for_game_end` recalculates the public count | Route completion | First completion and tenth-city terminal timing | **Implemented** | The derived counter is updated during the route action; completing the terminal city ends only after the action is fully resolved. |
| 3b) Develop an adjacent ability | Game Play E.3b | `claim_route_for_upgrade`; `Player.perform_upgrade` | Upgrade route actions and maximum-aware mask | Route return, released piece, immediate Action benefit, and max rejection | **Implemented** | A maxed ability is not exposed as an alternative. |
| 3c) Place a merchant for special prestige | Game Play E.3c | `Upgrade.claim_prestige`; route upgrade path | Four per-route prestige-value choices | Merchant and privilege requirements, choosing any eligible 7/8/9/11 space, ownership, and stock exclusion | **Implemented** | The four existing per-route upgrade slots represent the four printed prestige spaces, preserving the player’s strategic choice. |
| Route pieces return to general stock except an office/special-prestige piece | Game Play E | `update_stock_and_reset`; global invariants | Every route alternative | Per-alternative stock assertions and 27-trader/4-merchant invariants after every simulated action | **Implemented** | Physical piece totals are checked continuously, including held and displaced pieces. |

## East–West Connection

| Rule | Rule source | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- | --- |
| Connection requires a continuous chain of the active player’s offices | Game Play E.3a | `check_for_east_west_connection`, `has_east_west_connection` | Triggered after office claims | Connected, disconnected, and opponent-interrupted paths on all maps | **Implemented** | Both endpoints and every city in the path must contain an office belonging to the active player. |
| First three players score 7/4/2 once each | Game Play E.3a | `east_west_completed_count`, `players_who_completed_east_west` | N/A | Three-player award order plus repeat-call rejection | **Implemented** | Each player can receive the award once; later completions receive no connection points. |
| Map-specific endpoints | Other Game Boards | `east_west_cities` in map definitions | N/A | Exact endpoint fixtures plus path tests | **Implemented** | Stendal–Arnheim, Lübeck–Danzig, and York–Oxford match the supported boards. |

## End of Game and Final Scoring

| Rule | Rule source | Engine implementation | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| End after the action that reaches 20+ points | End of Game; FAQ | Route dispatch calls `Game.check_for_game_end` after the selected route alternative and immediate effects | Route scoring, office establishment, upgrades, bonus-marker collection, and twentieth-point timing | **Implemented** | The entire route action resolves before terminal scoring. Remaining ordinary actions are then forfeited. |
| End when a replacement marker must be drawn but supply is empty | End of Game | `bonus_pool_exhausted_during_claim` is set only on a required failed draw | Empty supply alone does not end; failed required draw does | **Implemented** | The newly collected marker remains collected, no pending replacement is created, and the game ends after the route action. |
| End on 10 completed cities; 8 on Britannia | End of Game | `selected_map.max_full_cities` and derived full-city count | Exact map-1 and Britannia thresholds; office-route timing; Additional Trading Post remains an already-occupied extension | **Implemented** | The threshold is evaluated after office establishment and complete route resolution. |
| Pending replacement markers on the silver plate do not score | FAQ | Pending marker types remain on `Game`, outside both player marker lists | Collected marker versus two pending replacements | **Implemented** | Only used and unused markers actually collected by the player count. |
| Score points already earned during play | Final Scoring A | `player.score` feeds `final_score` | Exact initial-score fixture and idempotent finalization | **Implemented** | In-game prestige is copied once into the recomputed final total. |
| Score 4 points per fully developed non-key ability | Final Scoring B | `Game.finalize_end_of_game_points` | All five tracks maxed yields exactly 16 ability points | **Implemented** | Privilege, Book, Actions, and Bank score; City Keys are explicitly excluded. |
| Score collected bonus markers by 1/3/6/10/15/21 table | Final Scoring C | `get_bonus_marker_points` | Every boundary from zero through ten-plus markers | **Implemented** | Used and unused collected markers count equally. |
| Score special-prestige spaces | Final Scoring D | `selected_map.specialprestigepoints.get_special_prestige_points_for_player` | Exact occupied-space value | **Implemented** | Values are read directly from the map’s special-prestige spaces. |
| Score 2 points per controlled city | Final Scoring E | `City.get_controller`; final scoring loop | Exact controlled-city total and rightmost-office tie fixture | **Implemented** | Additional offices count normally for control and only one controller scores each city. |
| Score largest connected office network multiplied by City Keys | Final Scoring F; FAQ | `dfs_network_size`, `calculate_largest_network` | Multiple offices in connected cities, disconnected opponent city, exact multiplier | **Implemented** | Counts every office in the largest connected component of cities containing the player’s offices, then multiplies by City Keys. |
| Resolve final-score ties using rulebook tie breakers | Final Scoring | `end_the_game` | Final score, least-developed Actions, largest network score, then shared victory | **Implemented** | Tie breakers are applied in the published order. |

## Standard Bonus Markers

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Exchange Trading Posts | Adjacent-pair enumeration and `City.swap_office_pair` | BM index 527 plus contextual adjacent-pair choice | Multiple eligible pairs; city controller may exchange; shape/privilege ignored; no gold points; additional offices excluded | **Implemented** | Every occupied adjacent pair containing exactly one of the player’s standard offices is selectable. Special-prestige spaces are not offices and never enter the choices. |
| Develop 1 Ability | `Player.perform_upgrade` | BM index 529 plus one of five ability choices | All five choices exposed; released piece; fully developed filtering; marker remains spent | **Implemented** | Develops exactly one non-maxed ability and transfers its leftmost trader or merchant to personal supply without spending an action. |
| Additional Trading Post | `waiting_for_bm_place_adjacent`; `claim_route_for_additional_office` | Appended activation index 619, then four contextual route slots encode city and route-piece shape | Activation legality; city and trader/merchant choice; occupied lowest standard office; route clearing; marker spending; swap exclusion; conservation | **Implemented** | This modifies route creation step 3a. It may be chosen even when a normal office is available, uses a selected piece from that route, creates the lowest-valued office to the left, and never participates in exchanges. Existing action indices 0–618 remain unchanged. |
| +3 Actions | `BonusMarker.handle_3_actions` | BM index 530 | Adds exactly three without spending an action; spent-marker preservation | **Implemented** | May be used at any point in the owner’s turn, including after ordinary actions reach zero but before the turn is finalized. |
| +4 Actions | `BonusMarker.handle_4_actions` | BM index 531 | Adds exactly four without spending an action; spent-marker preservation | **Implemented** | Same timing as +3 Actions. |
| Move 3 opponent tradesmen (“Remove 3 Resources” in the FAQ) | `BonusMarker.handle_move3`, `move_action` | BM index 528, opponent post choices, contextual early-finish index 618, then destinations | Multiple owners and shapes; up-to-three early transition; swapping through vacated posts; no displacement; spent-marker preservation | **Implemented** | These are two names for the same marker. A trader or merchant each counts as one piece. Route completion remains a separate action. Britannia adds country restrictions audited in its own section. |

## Mission Cards and Emperor’s Favour

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Secret mission-card setup and AI information | `use_mission_cards`; `Map1.assign_mission_cards`; `Player.mission_card`; perspective-filtered AI observation | N/A | Disabled by default; map 1 uniqueness and three-city card structure; enablement rejected on other maps; own card visible before game end; opponents’ cards hidden | **Implemented** | This optional module can only be enabled for map 1. Exact engine state retains every dealt card. Throughout play, an acting AI observes its own three mission cities so reinforcement learning can pursue them, while opponents’ cards remain hidden. |
| Mission final scoring | `Game.get_mission_card_points`; `Game.finalize_end_of_game_points` | N/A | One point for any office without control; three-city control bonus; loss of bonus when one city is not controlled | **Implemented** | At game end, actual board ownership is evaluated: each listed city containing at least one of the player’s offices scores 1 point. Controlling all three, including the normal rightmost-office tie break, adds 5 points, for a maximum of 8. |
| Buy one Emperor’s Favour tile at start of turn using two unused markers | `buy_tile`; tile state on `Game` | Tile indices `535–542`; `mask_buy_tile` | Exact two-marker payment; explicit selection from more than two; duplicate marker types; invalid timing/payment; turn forfeiture | **Implemented** | The six tile choices use the first six context-sensitive slots. During payment all eight slots identify marker types. Exactly two distinct unused marker objects move to the used area, the tile leaves the display, and the buyer forfeits all actions and acquires no second tile that turn. |
| Six tile effects | Owner fields on `Game`; displacement, turn-start, income-response, and scoring hooks | Contextual Favour response uses tile slots for trader, merchant, or decline | Displace-anywhere ownership; extra action; other-player income with both shape choices and decline; extra displaced trader; four-point city control; seven-point completed abilities | **Implemented** | All six effects are isolated in tests. The optional income response interrupts completion of the other player’s Income action, permits either available shape or decline, and does not trigger on its owner’s Income action. |

## Eastern Hanseatic League

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Waren offers Actions or Bank upgrade | Two upgrade types on Waren | Two contextual upgrade slots for each adjacent route | Both choices legal; ordinary office prohibited; Additional Trading Post remains available | **Implemented** | Each Waren route offers Actions or Bank. An office requires the Eastern Additional Trading Post exception. |
| Green/yellow cities cannot receive ordinary offices | `DARK_GREEN` route exclusion; Eastern-aware Additional Trading Post and permanent-marker paths | Ordinary office mask rejects; contextual alternatives choose city and shape | Ordinary rejection; first Additional office; completed-city behavior; right/left placement | **Implemented** | Belgard, Waren, and Dresden use the shared special-city representation. Additional offices go left; permanent-marker offices go right. |
| Maritime routes require merchants | Required-circle `Post` shapes | Ordinary placement, displacement, and route masks | Every maritime route’s exact one- or two-merchant requirement | **Implemented** | Required boat positions accept merchants only and route creation still requires the entire route to be controlled. |
| Permanent marker resolves immediately and is not collected | Persistent `Route.permanent_bonus_marker`; deferred terminal resolution | Route completion enters the marker’s contextual workflow before game finalization | All four effects; marker persists; player collections unchanged | **Implemented** | The marker is reusable whenever its maritime route is created and never contributes to collected-marker scoring. |
| Move any 2 tradesmen | `waiting_for_bm_move_any_2`; shared move engine | Own/opponent pickup, early-finish index 618, then destinations | Own piece, opponent preservation, early finish, swap-capable vacated destinations | **Implemented** | Up to two pieces of either ownership may be moved without displacement or action cost. |
| Develop Privilege | `perform_upgrade("Privilege")` | Immediate permanent-marker resolution | Privilege progression, released trader, no collected marker | **Implemented** | At maximum, the effect safely produces no further upgrade. |
| Establish office in green/yellow city | Shape-aware `City.claim_green_city` | Contextual city-and-shape choices | Trader and merchant choices; personal-supply source; placement right of all occupied offices | **Implemented** | Any special city may be selected; the office is added immediately to the right and can complete that city. |
| Place 2 tradesmen from route | `pending_route_piece_choices`; `waiting_for_place2_from_route` | Contextual 0/1/2-merchant composition, then two destination posts | Integrated maritime-route completion; exact composition; exactly two placements; conservation | **Implemented** | The chosen two pieces never enter general stock; all unchosen route pieces do. |

## Britannia

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Cardiff grants one Wales placement/displacement per turn | Live Cardiff control; `brown_priv_count` | `check_brown_blue_priv` in mask and engine | Control changes, availability, and single consumption | **Implemented** | Permission is recalculated from the current rightmost-tiebreak city controller at turn start. Legality checks do not consume it. |
| Carlisle grants one Scotland placement/displacement per turn | Live Carlisle control; `blue_priv_count` | Same | Layout and regional permission scenarios | **Implemented** | Present only where Scotland is in play and consumed by one B or C action. |
| London grants one Wales or Scotland permission per turn | Live London control; `london_priv_count` | Shared fallback in `consume_region_privilege` | Wales use prevents a later Scotland use | **Implemented** | London supplies one shared permission, not one counter per country. |
| Regional restrictions do not apply to route creation | Route claims do not call regional privilege check | Route mask | Complete-game and route-action coverage | **Implemented** | Permissions guard only ordinary placement and displacement, never route creation. |
| Displaced pieces stay in country of origin or England | `valid_region_transition`; region-aware adjacent-route search | Displacement post mask | Core displacement and transition coverage | **Implemented** | England remains England; Wales/Scotland may use their origin country or England, without spending permission. |
| Normal moves obey country restrictions | `Player.is_valid_region_transition` | Post mask | All directional transition cases | **Implemented** | Wales/Scotland may move to the same country or England; England cannot move outward. |
| Move 3 marker keeps each piece within one country | Exact origin/destination region equality on Britannia | BM move mask | Wales versus England destinations and multi-owner workflow | **Implemented** | This marker deliberately uses a stricter rule than normal movement. |
| Britannia maritime permanent markers | Player-count-specific route fields and immediate handlers | Route/post and composition masks | Both board sides and every marker location | **Implemented** | Two southeast markers occur on both boards; 3-player also has Carlisle–Isle of Man. Move Any 2 stays within country. |
| Place 2 in Wales/Scotland | `pending_britannia_place2`; `waiting_for_place2_in_scotland_or_wales` | Shape composition then shape-aware regional post mask | General-stock source and both destination regions | **Implemented** | Pieces are sourced from general stock, then personal supply, then the board, and consume no additional action. |
| Regional Wales and Scotland end-game scoring | `calculate_britannia_region_points` | Included in final-score breakdown | 7/4/2, pooled ties, rounding down, and Isle of Man | **Implemented** | Wales is always scored; Scotland is added on the 4–5-player board. Isle of Man participates in both ladders. |

## Promo Bonus Markers

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Optionally use exactly 15 total markers with a player-chosen promo mix | `bonus_marker_supply`; `Map.configure_bonus_marker_supply` | Setup API and repeatable `--bonus-marker` CLI option | Default excludes promos; explicit seeded mix; exactly 12 supply plus 3 starting markers; unknown, excess, and wrong-count rejection | **Implemented** | Promo markers never enter default play. The caller explicitly supplies all 12 replacement-supply marker types; physical component limits are enforced and the 3 fixed starting markers preserve exactly 15 total. |
| Exchange Bonus Marker | Explicit pending exchange marker and target player on `Game` | BM index 532, contextual player choice, then used-marker type choice | Chosen opponent; only used markers; acquired marker becomes unused; Exchange marker remains spent at opponent | **Implemented** | The player chooses an opponent who has used markers, then chooses one marker type from that player’s used area. The exchanged marker is immediately usable and the Exchange marker moves to the chosen opponent’s used area. |
| Tribute for Establishing a Trading Post | Route-level `tribute_owners`; committed trader conservation; queued tribute-income response | BM index 533, unrestricted route choice, then contextual two-piece Income composition | One-trader setup; every route eligible; self-trigger; neighboring-route isolation; persistent marker; two-piece shape choice | **Implemented** | The marker and one trader remain on the selected route. Establishing an office in either neighboring city queues two tradesmen from general stock for every Tribute marker on that route, including the active player’s own marker. |
| Block Trade Route | Route-level `block_marker_owners`; committed trader conservation; ordinary-placement surcharge | BM index 534 plus unrestricted route choice; ordinary post mask includes surcharge affordability | One-trader setup; every route eligible; marker persistence; extra-piece payment and piece conservation | **Implemented** | Each Block marker on the route adds one extra tradesman returned to general stock for action B placement. It applies to every player, including the marker owner; movement and displacement are not charged. |

## Highest-Priority Remaining Risks

1. **Saved-state coverage is not yet complete for every contextual workflow.**
   Britannia’s pending Place 2 choice is serialized, but exact round trips for
   every in-progress marker, displacement, and optional-module phase still need
   dedicated fixtures.

2. **The displaced-player workflow still needs broader exhaustion scenarios.**
   Core adjacency, source fallback, and optional-extra-piece behavior are tested,
   but dense-board cases should be expanded before calling the whole engine
   flawless.

3. **Complete-game smoke tests prove termination, not strategic rule coverage.**
   Keep scenario tests as the authority for individual rules while expanding
   deterministic full-game traces.

## Recommended Test Implementation Order

1. Setup values and piece conservation
2. Income combinations
3. Ordinary placement and required shapes
4. Trader and merchant displacement, including adjacency fallback
5. Normal movement and early completion
6. Route creation and ordered substeps
7. Office privilege, shape, coin, and city-completion rules
8. Ability upgrades
9. Bonus-marker acquisition and replacement
10. End triggers and exact final scoring
11. Standard bonus markers
12. Eastern map rules
13. Britannia regional rules and scoring
14. Promo markers, mission cards, and Emperor’s Favour

For each row, the preferred test pattern is:

1. Construct the smallest relevant position.
2. Assert the complete legal-action set for that decision.
3. Apply one structured action or action index.
4. Assert piece conservation, scores, ownership, counters, and pending phase.
5. Assert illegal alternatives remain unavailable.
