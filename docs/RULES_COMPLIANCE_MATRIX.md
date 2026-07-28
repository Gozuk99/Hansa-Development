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
- A fresh game exposes at least one action in a 619-entry mask.
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
| Starting personal supply and general stock depend on player order | Game Setup | `game.setup.starting_inventory`; `Player.__init__` | N/A | Exact inventories for player orders 1–5 and game sizes 3–5 | **Implemented** | Each player begins with 11 available traders split by order and one merchant in personal supply. |
| Starting abilities | Player Writing Desk | `Player.__init__`; upgrade methods and unlock helpers | Route-upgrade and bonus-marker masks | Exact starting values and complete progression tests for all five tracks | **Implemented** | Keys `1,2,2,3,4`; Actions `2,3,3,4,4,5`; Privilege white through black; Book `2,3,4,5`; Bank `3,4,7,C`. Upgrades release the correct trader or merchant, repeated Action values do not grant an extra current-turn action, and maximum spaces cannot be upgraded again. The engine uses `50` internally for Bank `C` (complete/all). |
| Starting bonus markers and supply | Game Setup | `Map.assign_bm_pool_default`, `Map.assign_starting_bonus_markers` | Replacement marker mask | Exact starting-marker types and supply counts on every map | **Implemented** | Every map starts with Move 3, Swap Office, and Place Adjacent on routes, plus the 12-marker supply. |
| Turn order and actions per turn | Game Play | `Game.advance_turn`; `Player.start_turn`, `spend_action`, `grant_actions`, `forfeit_remaining_actions` | `map_end_turn_action`, `mask_end_turn` | Direct action-count, switching, round numbering, early-end rejection, optional marker forfeit, and +1 Action tile tests | **Implemented** | Action counts cannot go negative. Turn advancement and the next player’s allowance are initialized in one place. |
| Finish one action before beginning another | Game Play | `Game.turn_phase`; pending flags, held pieces, and displacement state | `restrict_mask_to_turn_phase` prevents action-family leakage | Advancement guards; phase-exclusive masks; displacement-before-marker-replacement ordering | **Implemented** | Turn advancement is phase-guarded. Displacement, movement, tile payment, marker choices, and marker replacement restrict legal actions to their own workflow. Replacing legacy booleans with structured phase payloads remains a maintainability improvement, not a correctness blocker for this milestone. |
| Deterministic setup | Development requirement | `Game(seed=...)` owns the RNG passed into map, marker, mission-card, and tile setup | N/A | Same-seed setup across all maps; global RNG isolation | **Implemented** | Setup randomness no longer temporarily reseeds or consumes Python’s global random generator. Runtime stochastic policies use their own seeded generator. |

## Core Actions

| Rule | Rule source | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- | --- |
| A) Income up to current Bank capacity | Game Play A | `Player.income_action`; `map_income_action` | `mask_income_actions` | None | **Partial / Conflict** | Basic transfers exist. Track-value conflict noted above. Validate every square/circle combination, insufficient stock, and `C/all`. |
| B) Place one tradesman from personal supply on a free point | Game Play B | `claim_post_action` | Post indices `0–241`; `mask_post_action` | Fresh mask only | **Partial** | Basic placement and required shapes exist. No focused conservation, action-cost, blocked-route, or region tests. |
| C) Displace an opposing trader for one extra piece | Game Play C | `displace_action`, `displace_claim`, `displace_to` | Post mask and dispatcher | Complete games exercise some displacement paths | **Partial** | Core cost and displaced placement exist, but the flow is complex and formerly deadlocked. Needs scenario tests for stock, supply, and board fallback. |
| C) Displace an opposing merchant for two extra pieces | Game Play C | Same as above | Same as above | None | **Unverified** | Merchant-specific cost and three-piece relocation need direct tests. |
| Displaced pieces go to nearest available adjacent routes | Game Play C | `gather_empty_adjacent_posts`, `get_adjacent_routes` | Displacement phase in `mask_post_action` | None | **Partial** | Breadth-first adjacency exists. Required-shape and region fallback recently needed fixes and remain unverified. |
| Displaced player may decline additional pieces | Game Play C | No explicit decline action found | No explicit action index | None | **Missing** | The rule allows declining additional pieces from stock/supply. Current displacement counters appear to require all calculated pieces. |
| D) Move 2–5 own tradesmen according to Book value | Game Play D | `Player.start_move`, `pick_up_piece`, `place_piece`, `move_action` | Post mask and dispatcher | Complete games exercise moves | **Partial** | Pickup/placement exists. There is no explicit “finish early” action while fewer than the maximum have been moved; behavior is driven by placing held pieces. |
| D) Swap one own trader and merchant | Game Play D | General move system may allow it | Same | None | **Unverified** | Needs a focused test showing two occupied positions can be exchanged legally. |
| Normal moves cannot displace opponents | Game Play D | `move_action` distinguishes owned and empty posts | Mask filters normal move targets | None | **Implemented / Unverified** | Add negative legality tests. |

## Creating a Trade Route

| Rule | Rule source | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- | --- |
| Route can be created only when every point is occupied by the active player | Game Play E | `Route.is_controlled_by`; route-claim functions | `mask_claim_route` | None | **Implemented / Unverified** | Add positive and negative masks for mixed, empty, and fully controlled routes. |
| Step 1: score controllers of both endpoint cities | Game Play E.1 | `score_route`; `City.get_controller` | Performed by route-claim functions | None | **Partial** | City controller and rightmost-office tie logic exist. Exact ordering and score increments need tests. |
| Step 2: take a route bonus marker | Game Play E.2 | `handle_bonus_marker`, `finalize_route_claim` | Route actions plus replacement phase | None | **Partial** | Marker acquisition exists. Verify ownership, used/unused state, replacement count, and end-game interaction. |
| Replacement marker must be placed at end of turn | Game Play E.2 | `replace_bonus_marker`; `assign_new_bonus_marker_on_route` | Indices `543–582`; `mask_replace_bm` | None | **Partial** | Three placement restrictions appear in the mask. Turn-end ordering needs direct tests. |
| Step 3 offers exactly one of office, ability, or special prestige | Game Play E.3 | Separate route-claim functions | Fixed route action alternatives | None | **Partial** | The fixed action space represents choices, but tests must prove only legal alternatives are exposed and exactly one is applied. |
| 3a) Establish the leftmost vacant trading post | Game Play E.3a | `City.update_next_open_office_ownership`; `claim_route_for_office` | Office route actions | None | **Implemented / Unverified** | Required privilege and piece shape are checked by the mask. Mutation-side legality should also be asserted. |
| Office requires matching trader/merchant shape | Game Play E.3a | `City.has_required_piece_shape`; route reset functions | `mask_claim_route` | None | **Partial** | Add scenarios where a controlled route has and lacks the required shape. |
| Office color requires sufficient Privilege | Game Play E.3a | `Player.player_can_claim_office` | `mask_claim_route` | None | **Implemented / Unverified** | Add tests for white, orange, pink, and black thresholds. |
| Gold-coin point is awarded only when establishing the marked office | Rulebook and FAQ | `Office.awards_points`; `City.update_next_open_office_ownership` | Office route action | None | **Partial** | Office placement awards points. FAQ confirms swaps must not award the coin; add a regression test. |
| Completing a city advances the Completed Cities count | Game Play E.3a | Count is recalculated in `Game.check_for_game_end` | N/A | None | **Partial** | There is no persistent track mutation on office placement; end checking derives the count from full cities. Verify equivalence and timing. |
| 3b) Develop an adjacent ability | Game Play E.3b | `claim_route_for_upgrade`; `Player.perform_upgrade` | Upgrade route actions and mask | None | **Partial** | Upgrades and released pieces exist. Ability maximum checks and action-track side effects need tests. |
| 3c) Place a merchant for special prestige | Game Play E.3c | `Upgrade.claim_highest_prestige`; route upgrade path | Route-upgrade actions | None | **Partial** | Needs tests for merchant requirement, privilege requirement, occupied spaces, and scoring values. |
| Route pieces return to general stock except an office/special-prestige piece | Game Play E | `update_stock_and_reset` | Applied by route functions | None | **Partial** | Central conservation rule; high-priority invariant and scenario tests are needed. |

## East–West Connection

| Rule | Rule source | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- | --- |
| Connection requires a continuous chain of the active player’s offices | Game Play E.3a | `check_for_east_west_connection`, `has_east_west_connection` | Triggered after office claims | None | **Partial** | DFS exists. Add disconnected, connected, and opponent-interrupted examples for each map. |
| First three players score 7/4/2 once each | Game Play E.3a | `east_west_completed_count`, `players_who_completed_east_west` | N/A | None | **Implemented / Unverified** | Snapshot support for the set is currently absent, but runtime logic exists. |
| Map-specific endpoints | Other Game Boards | Hard-coded `east_west_cities` in map definitions | N/A | None | **Unverified** | Verify Stendal–Arnheim, Lübeck–Danzig, and Oxford–York definitions against each board. |

## End of Game and Final Scoring

| Rule | Rule source | Engine implementation | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| End after the action that reaches 20+ points | End of Game; FAQ | `Game.check_for_game_end` is called from route dispatch | Complete games only | **Partial** | FAQ requires finishing the entire route action. Verify every route substep completes before terminal state is finalized. |
| End when a replacement marker must be drawn but supply is empty | End of Game | `check_for_game_end` ends whenever `bonus_marker_pool` is empty | None | **Conflict** | Current condition can end a game merely because the pool is empty, even if the active action did not require drawing a replacement marker. |
| End on 10 completed cities; 8 on Britannia | End of Game | `selected_map.max_full_cities` and derived full-city count | None | **Partial** | Values appear map-specific. Timing and Additional Trading Post interaction need tests. |
| Pending replacement markers on the silver plate do not score | FAQ | Replacement count exists; final scoring counts player marker lists | None | **Unverified** | Add a terminal-state test with an empty supply and a pending replacement. |
| Score points already earned during play | Final Scoring A | `player.score` feeds `final_score` | Complete games assert final score exists | **Implemented / Unverified** | Add exact scoring fixtures. |
| Score 4 points per fully developed non-key ability | Final Scoring B | `Game.finalize_end_of_game_points` | None | **Conflict** | Code scores `keys`, `book`, `actions`, and `bank`, but omits `privilege`. The rulebook excludes City Keys from this category and includes the other four abilities. |
| Score collected bonus markers by 1/3/6/10/15/21 table | Final Scoring C | `get_bonus_marker_points` | None | **Implemented / Unverified** | Add boundary tests for 0 through 10+ markers and exclude pending replacements. |
| Score special-prestige spaces | Final Scoring D | `get_special_prestige_points_for_player` | None | **Implemented / Unverified** | Add exact ownership/value tests. |
| Score 2 points per controlled city | Final Scoring E | `City.get_controller`; final scoring loop | None | **Implemented / Unverified** | Add ties resolved by rightmost office. |
| Score largest connected office network multiplied by City Keys | Final Scoring F; FAQ | `dfs_network_size`, `calculate_largest_network` | None | **Partial** | Code counts offices rather than merely cities, consistent with FAQ. Test multiple disconnected networks and multiple offices in one city. |
| Resolve final-score ties using rulebook tie breakers | Final Scoring | `end_the_game` returns every highest-scoring player | None | **Missing** | Current code treats equal final scores as shared winners and does not implement the published tie-break sequence. |

## Standard Bonus Markers

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Exchange Trading Posts | `City.swap_offices`, `BonusMarker.handle_swap_office` | BM index 527 plus city choice | None | **Partial** | Must reject additional offices and special-prestige spaces; must not award gold-coin points. |
| Develop 1 Ability | `Player.perform_upgrade` | BM index 529 plus ability choice | None | **Partial** | Maximum-value filtering has a likely key-case inconsistency in `mask_bm_upgrade_ability`. |
| Additional Trading Post | `City.claim_office_with_bonus_marker` | Represented through city/route interaction | None | **Partial** | Confirm timing: it modifies route creation step 3a and cannot later participate in swaps. |
| +3 Actions | `BonusMarker.handle_3_actions` | BM index 530 | None | **Implemented / Unverified** | Verify it adds actions without consuming an action and interacts correctly with turn end. |
| +4 Actions | `BonusMarker.handle_4_actions` | BM index 531 | None | **Implemented / Unverified** | Same as above. |
| Move 3 opponent tradesmen | `BonusMarker.handle_move3`, `move_action` | BM index 528 plus post choices | None | **Partial** | Core flow exists. Ownership, “up to 3,” swapping, route-completion consequences, and Britannia country restrictions need tests. |
| Remove 3 Resources FAQ marker | No corresponding type found in standard BM mapping | None | None | **Missing / Source question** | The FAQ discusses this marker, but it is not present in the supplied Big Box component list or current action space. Decide whether it belongs in project scope. |

## Mission Cards and Emperor’s Favour

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Secret mission-card setup | `Map1.assign_mission_cards`; `Player.mission_card` | N/A | Map 1 uniqueness and three-city card structure; other maps have no cards | **Implemented** | Mission cards are assigned only for the original map and are unique within a game. |
| Mission final scoring | `Game.finalize_end_of_game_points` | N/A | None | **Partial** | Code awards one point per controlled mission city plus five for all three. Add exact fixtures. |
| Buy one Emperor’s Favour tile at start of turn using two unused markers | `buy_tile`; tile state on `Game` | Tile indices `535–542`; `mask_buy_tile` | None | **Conflict / Partial** | Tile strings and payment state have had inconsistent field names and mapping directions. Multi-marker selection remains unverified. |
| Six tile effects | Owner fields on `Game`; selected action/scoring hooks | Mostly indirect | None | **Partial** | Some effects exist: displacement range, +1 action, income response, extra displaced piece, city points, and ability points. Each needs an isolated test. |

## Eastern Hanseatic League

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Waren offers Actions or Bank upgrade | Map 2 city data supports multiple upgrade types | Route-upgrade mask has two slots per city | None | **Partial** | Verify both choices and office alternative. |
| Green/yellow cities cannot receive ordinary offices | `DARK_GREEN` checks; `City.claim_green_city` | Route mask and BM city mask | None | **Partial** | Exact yellow/green distinctions and Additional Trading Post path need tests. |
| Maritime routes require merchants | `Route.required_circles`; required-shape posts | Post and route masks | None | **Partial** | Verify one- and two-merchant routes and route clearing. |
| Permanent marker resolves immediately and is not collected | `Route.has_permanent_bm_type`; `handle_bonus_marker` | Route completion invokes permanent effects | None | **Partial** | Test all four Eastern permanent markers and confirm they do not enter player marker collections. |
| Move any 2 tradesmen | `waiting_for_bm_move_any_2`; `move_action` | Post phase | None | **Partial** | Recently corrected masking still needs own/opponent, swap, and early-finish tests. |
| Develop Privilege | Permanent marker path through route handling | Ability mutation | None | **Partial** | Add max-value and released-piece tests. |
| Establish office in green/yellow city | `waiting_for_bm_green_city`; `City.claim_green_city` | City choice | None | **Partial** | Confirm source of piece and left/right placement rule. |
| Place 2 tradesmen from route | `handle_bonus_marker` and held/reset pieces | Post choices | None | **Partial** | Verify exactly two eligible route pieces are retained and placed. |

## Britannia

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Cardiff grants one Wales placement/displacement per turn | `cardiff_priv`, `brown_priv_count` | `check_brown_blue_priv` in mask and engine | None | **Conflict / Partial** | Privilege ownership and counters exist, but `claim_post_action` appears to decrement the regional counter after `Game.check_brown_blue_priv` has already decremented it, risking double consumption. |
| Carlisle grants one Scotland placement/displacement per turn | `carlisle_priv`, `blue_priv_count` | Same | None | **Conflict / Partial** | Same double-decrement risk. |
| London grants one Wales or Scotland permission per turn | `london_priv`; both counters refreshed | Same | None | **Partial** | Code provides separate brown and blue counters, which may permit one action in each country rather than one total action. Confirm rule interpretation and test. |
| Regional restrictions do not apply to route creation | Route claims do not call regional privilege check | Route mask | None | **Implemented / Unverified** | Add a route-completion test across regional borders. |
| Displaced pieces stay in country of origin or England | Region-aware adjacency functions | Displacement mask | None | **Partial** | Needs Wales, Scotland, and England scenarios. |
| Normal moves obey country restrictions | `Player.is_valid_region_transition` | Post mask | None | **Partial** | Current transition rules allow Wales/Scotland to same region or England and England only to England. Test all directions. |
| Move 3 marker keeps each piece within one country | General region transition logic | BM move mask | None | **Partial** | Verify opponent pieces from multiple countries cannot be combined illegally. |
| Britannia maritime permanent markers | Permanent marker route fields and handlers | Route/post mask | None | **Partial** | Test each board side and marker location. |
| Place 2 in Wales/Scotland | `waiting_for_place2_in_scotland_or_wales` | Post mask | None | **Partial** | Verify source priority: general stock, then personal supply, then board. |
| Regional Wales and Scotland end-game scoring | No scoring routine found | N/A | None | **Missing** | Required 7/4/2 awards, tie resolution, and Isle of Man double-region treatment are not implemented. |

## Promo Bonus Markers

| Rule | Engine implementation | Mask/dispatcher | Tests | Status | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Use exactly 15 total markers with chosen promo mix | `assign_bm_pool_random` can select from expanded types | N/A | None | **Partial** | No setup API exposes an explicit chosen mix or guarantees the promo rule’s intended composition. |
| Exchange Bonus Marker | `waiting_for_bm_exchange_bm`; exchange handler | BM index 532 | None | **Partial** | Verify only another player’s used markers are eligible and the exchanged marker remains spent. |
| Tribute for Establishing a Trading Post | `Route.establish_tribute_on_route`, `award_tributes` | BM index 533 plus route/post choice | None | **Partial** | Verify setup cost, unrestricted route choice, self-triggering, neighboring cities, and two-piece income. |
| Block Trade Route | `Route.establish_blocked_route`; blocked-post placement cost | BM index 534 plus route/post choice | None | **Partial** | Verify setup cost and one additional piece returned for each ordinary placement by every player, including owner. |

## Highest-Priority Conflicts

1. **Final ability scoring uses the wrong ability set.**  
   `Game.finalize_end_of_game_points` includes City Keys and omits Privilege.

2. **An empty bonus-marker supply ends the game too broadly.**  
   The rule triggers only when a player must draw a replacement and cannot; `Game.check_for_game_end` currently treats an empty pool itself as sufficient.

3. **Britannia regional privilege may be consumed twice.**  
   `Game.check_brown_blue_priv` decrements a counter, and `claim_post_action` appears to decrement it again.

4. **Britannia regional end-game scoring is missing.**

5. **Final tie breakers are missing.**

6. **Emperor’s Favour purchase/payment flow has conflicting field and mapping designs.**

7. **The displaced player cannot explicitly decline optional additional pieces.**

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
