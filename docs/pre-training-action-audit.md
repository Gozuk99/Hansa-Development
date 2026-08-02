# Pre-Training Action Audit

Status: **READY FOR INITIAL CONTROLLED TRAINING**

This audit checks whether the game and the AI agree about what a player may do.
It does not judge whether the AI makes intelligent choices.

## Action system

- Schema version: **2**
- Neural-network action outputs: **768**
- Assigned interactions: **639**
- Reserved for future additions: **129**
- Existing assigned action numbers moved by this milestone: **0**

The audit found and corrected one omitted legal choice: an Additional Trading
Post bonus marker may be spent to buy an Emperor's Favour tile. Payment now
reuses that marker's existing action instead of creating a new action number.

## Automated audit matrix

The repeatable audit command is:

```powershell
python -B tools/pre_training_audit.py --out audit_milestone8.json
```

Results from seed 124 complete games:

| Map | Players | Actions | Final scores | Repeated identically |
|---:|---:|---:|---|---|
| 1 | 3 | 161 | 23, 6, 3 | Yes |
| 1 | 4 | 321 | 11, 10, 19, 6 | Yes |
| 1 | 5 | 617 | 17, 13, 6, 5, 9 | Yes |
| 2 | 3 | 244 | 17, 13, 6 | Yes |
| 2 | 4 | 448 | 18, 3, 8, 8 | Yes |
| 2 | 5 | 539 | 3, 13, 13, 4, 18 | Yes |
| 3 | 3 | 197 | 20, 13, 9 | Yes |
| 3 | 4 | 238 | 3, 9, 19, 7 | Yes |
| 3 | 5 | 381 | 8, 6, 3, 13, 6 | Yes |

- Fresh starting positions checked: **18 of 18 passed**
- Starting-position seeds: **124 and 125**
- Complete games checked: **9 of 9 passed**
- Complete-game actions executed: **3,146**
- Maps checked: **1, 2, and 3**
- Player counts checked: **3, 4, and 5**

Each starting-position check tried every currently legal interaction on copied
game states. It verified that the interaction was enabled, decoded correctly,
executed without an error, preserved the game rules, and produced the same
result twice.

## Rules and action-family coverage

| Area | Evidence |
|---|---|
| Posts, placement, movement, and displacement | `test_core_actions.py`, `test_action_validation.py` |
| Route points, offices, upgrades, and prestige | `test_core_actions.py`, `test_action_validation.py` |
| Income choices and Bank limits | `test_core_actions.py` |
| Standard and promotional bonus markers | `test_standard_bonus_markers.py`, `test_promo_bonus_markers.py` |
| Emperor's Favour tiles and payment choices | `test_optional_modules.py`, `test_action_validation.py` |
| Eastern map and green-city choices | `test_eastern_hanseatic.py` |
| Britannia regions and permanent markers | `test_britannia.py` |
| Finish, decline, pass, and end turn | `test_turn_structure.py`, `test_action_validation.py` |
| Final scoring and game-end conditions | `test_final_scoring.py` |
| Saving, restoring, and identical masks | `test_action_validation.py` |
| Replay restoration and identical masks | `test_action_validation.py`, `test_action_schema_versioning.py` |
| Schema compatibility and unchanged meanings | `test_action_schema.py`, `test_action_codec.py`, `test_action_schema_versioning.py` |

## Index reachability

- All **639 assigned indices** decode, describe themselves, and return to the
  same index when encoded again.
- The nine baseline complete games naturally selected **317 distinct indices**.
- The remaining **322** were not selected by that simple game-playing policy.
  They include alternate board locations and optional choices covered by focused
  tests. “Not selected” does not mean “illegal” or “unreachable.”
- Proven unreachable assigned indices: **none**.
- All **129 reserved indices** remained disabled.

The JSON audit output records the exact observed and unobserved lists so later
audits can compare coverage without changing action numbers.

## Optional rules

Mission Cards, Emperor's Favour, promotional bonus markers, Eastern-map rules,
and Britannia rules have focused tests. The nine complete-game audit runs use
the normal seeded setup and do not attempt every possible combination of all
optional modules in one game.

## Known limitations

- The baseline player is designed to finish games legally, not play well.
- Legality helpers and structured-action execution are engine-owned. AI,
  headless, and GUI callers share `Game.get_legal_actions()` and the 768-entry
  codec without a legacy 620-entry dispatch adapter.
- Natural full games will not choose every alternate location or optional
  interaction. Focused tests provide coverage for those branches.

## Remaining failures

- Blocking: **0**
- Major: **0**
- Known test failures: **0**

## Readiness decision

The action system is ready for **initial controlled self-play training**.

This means training may begin with schema version 2 while results are monitored.
It does not mean reward quality, learning speed, or playing strength has been
validated. Models, replays, and datasets must retain the schema metadata, and
future actions must use reserved family slots without moving existing actions.
