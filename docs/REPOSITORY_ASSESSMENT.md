# Archived Hansa Teutonica AI Repository Assessment

> This assessment describes the repository before the 768-entry structured-action
> migration. Its action counts and implementation paths are historical, not current.

## Executive Summary

The project has a substantial playable rules engine, a board renderer, exact
versioned saves, a fixed neural-network observation/action interface, and an
experimental self-play loop.

The highest priority is making the game engine deterministic, serializable, and testable without pygame or PyTorch. Training should remain paused until saving, loading, legal actions, and complete-game simulation are demonstrably reliable.

The inspection used only read-only repository listing, text searches, Python syntax parsing, core-module imports, and Git history inspection. All 14 Python files parsed successfully, and the core modules imported successfully with the installed Python 3.11, pygame, and PyTorch environment. The documented main script was intentionally not run because it begins training and writes model files.

## 1. Current Game-Engine Structure

The repository is organized into five main areas:

- `game/game_info.py` contains the top-level `Game` object. It owns the map, players, current turn, bonus-marker workflows, displacement workflow, tile ownership, end-game state, and scoring.
- `game/game_actions.py` contains state-changing rule functions: claiming and displacing posts, moving pieces, completing routes, claiming offices, upgrades and points, assigning bonus markers, and buying tiles.
- `map_data/map_attributes.py` defines the board objects: `Map`, `City`, `Office`, `Route`, `Post`, and `BonusMarker`. The three concrete map files construct hard-coded boards.
- `player_info/player_attributes.py` defines `Player`, `DisplacedPlayer`, and `PlayerBoard`.
- `hansa_game.py` is the interactive GUI entry point and delegates game drawing and input to `drawing/game_window.py`.

In plain English, the engine is a mutable object graph. A `Game` points to a map and players; cities point to offices and routes; routes point to cities and posts; offices and posts point back to owning `Player` objects. Actions directly mutate this graph.

There is no environment-style API such as:

```text
state = reset(...)
legal_actions = state.legal_actions()
next_state, event = apply_action(state, action)
```

Instead, module-level functions receive a mutable `game`, change several fields, print diagnostics, and sometimes switch players. Multi-step moves are represented by flags and temporary fields on `Game` and `Player`.

That can support interactive play, but it makes exact checkpointing, replay, branching simulation, and automated testing difficult.

## 2. How Game State Is Represented

There are three different notions of state.

### Runtime state

The authoritative runtime state is the mutable `Game` object graph.

Important fields include:

- Map number and concrete map
- Players and current player
- Scores, abilities, stocks, supplies, bonus markers, and tiles
- Ownership of offices and route posts
- Bonus-marker pool and route markers
- Turn counters and `actions_remaining`
- Numerous `waiting_for_*` flags
- Displacement state
- Pieces currently being picked up or held
- Pending tile purchases
- End-game counters and special privileges

These fields collectively determine what can legally happen next.

### Neural-network observation

`ai/observation_encoder.py` converts the object graph into a fixed tensor of
4,445 values.

It pads the representation to fixed maximums:

- 30 cities
- 40 routes
- 121 posts
- Up to five players
- Fixed encodings for colors, ownership, upgrades, bonus markers, tiles, and pending-action flags

This tensor is an observation for the model, not a reversible serialization format. It necessarily loses or compresses information and should not be treated as an exact saved game.

### Exact saved games

`game/persistence.py` stores complete, versioned `.hansa` snapshots. The
interactive game exposes Save Game during play and Load Saved Game from the
main menu. `ai/observation_encoder.py` is only the neural-network input encoder;
it is intentionally separate from persistence.

## 3. Actions and Legal-Action Masking

The AI uses one fixed categorical action space with 619 entries.

| Index range | Meaning |
| --- | --- |
| 0-241 | Claim, displace, or move on one of 121 posts, square or circle |
| 242-521 | Complete one of 40 routes using one of seven route outcomes |
| 522-526 | Income choices |
| 527-534 | Use a bonus marker |
| 535-542 | Buy a tile or select bonus-marker payment |
| 543-582 | Place a replacement bonus marker on a route |
| 583-612 | Bonus-marker city selection |
| 613-617 | Bonus-marker ability upgrade |
| 618 | End turn |

`masking_out_invalid_actions(game)` builds nine smaller masks and concatenates them into a 619-element `uint8` tensor. A value of `1` means the action is currently legal.

`perform_action_from_index()` reverses the mapping and invokes the relevant engine function.

Multi-stage decisions are handled by changing state and then generating a different mask. For example:

1. Select the upgrade-ability bonus marker.
2. Set `waiting_for_bm_upgrade_ability`.
3. The next mask permits only the five upgrade choices.
4. Select the actual ability.

Displacement follows the same general pattern, using `waiting_for_displaced_player`, `displaced_player`, `all_empty_posts`, held pieces, and placement counters.

This is a reasonable starting strategy for neural-network output, but the representation is positional and fragile:

- Post, route, and city indices depend on hard-coded map-object ordering.
- Legal-action logic duplicates rules already present in mutation functions.
- Some masks contain unfinished validation, including a TODO for bonus-marker validity.
- Invalid AI actions can open a pygame error window and wait forever through `error_exit()`, coupling headless simulation to the GUI.
- One call invokes `error_exit(game)` without the required `route` argument and would raise `TypeError`.

For future replay and history, action indices alone are insufficient unless map ordering and action-space versions remain permanently stable. History should also record structured actions containing route identity, post identity, piece shape, actor, and action-space version.

## 4. Current Reinforcement Learning

Each `Player` constructs and loads its own `HansaNN` during initialization. Five model checkpoints of roughly 47 MB each are present.

The network is:

```text
4,445 inputs
-> 2,048 ReLU
-> 1,024 ReLU
-> 619 raw outputs
```

Each model has its own Adam optimizer. The main script performs epsilon-greedy action selection from masked outputs.

The reward system is hand-shaped in `player_info/reward_options.py`. Rule functions add or subtract rewards for events such as claiming posts, upgrading, completing routes, and final placement.

This is not currently valid Q-learning, despite the variable names:

- A target Q-value is calculated but never used.
- The actual loss is cross-entropy between all 619 outputs and the selected action index.
- Consequently, the update trains the network to predict the action it just selected, largely independently of whether its reward was good or bad.
- Future Q-values are not legal-action masked.
- There is no replay buffer, target network, episode transition record, or proper terminal-state target.
- The next state can belong to a different active player but is evaluated by the previous active player's model.
- Optimizer state, exploration state, RNG state, and training progress are not saved.
- Every newly created game reloads large model files while constructing players.

The existing `.pth` files should therefore be treated as experimental artifacts, not evidence of a trained Hansa agent.

## 5. Entry Points That Can Run

There is one documented executable GUI entry point:

```powershell
python hansa_game.py
```

`hansa_game.py` has an explicit `main()` boundary and an `if __name__ == "__main__"`
guard. Importing it does not initialize pygame, start training, or modify model
checkpoints. Running it opens the New Game menu, which can also load an exact
saved game before starting the interactive game window.

## 6. Major Bugs and Incomplete or Conflicting Designs

### Critical: JSON loading is broken in common cases

The privilege loader uses nonexistent names:

```python
game.cardiff_prov = game.player[...]
```

The runtime fields are `cardiff_priv` and `players`. Any saved non-null map-3 privilege will fail.

### Critical: Save and load formats disagree

Examples:

- The saver writes individual keys such as `player_bm_<type>`.
- The loader expects arrays named `player_unused_bonus_markers` and `player_used_bonus_markers`.
- Duplicate bonus markers of the same type overwrite each other because they use the same dictionary key.
- The saver writes `player_final_score`, `player_keys_index`, `player_mission_card_cities`, and `player_ending_turn`; the loader does not restore them.
- City upgrades are written but not restored.
- Several route and post attributes are written but not restored explicitly.

### Critical: Snapshots omit state needed for exact continuation

The following are not reliably preserved or reconstructed:

- `waiting_for_displaced_player`
- Displaced player, shape, and number of pieces remaining
- Original displaced route and allowed destination posts
- Held pieces and pickup or placement counters
- `players_who_completed_east_west`
- `game_end`
- Player rewards
- All aspects of pending multi-step workflows
- Random-generator state
- Event and action history
- Model, optimizer, and training metadata

Loading a save made during a compound action will not reproduce the same legal next action.

### Critical: Some values may not be JSON serializable

City tribute fields are saved as `Player` objects instead of player IDs or orders. Pending bonus-marker payment may similarly contain an object instead of a primitive value. Those states can make `json.dump()` fail.

### High: The RL update is mathematically inconsistent

The computed Bellman target is unused, while cross-entropy reinforces the selected action. Training should not resume until this is replaced.

### High: Engine, UI, and AI are coupled

- `Player` always constructs a neural network.
- Action mapping imports pygame and drawing code.
- Invalid engine conditions may open a GUI and spin forever.
- Rule functions print extensively or call `sys.exit`.
- The main script owns both training and interactive play.

This prevents lightweight headless simulations and makes failures hard to classify.

### High: Legal rules exist in multiple places

Manual click handlers, masks, action-index mapping, and mutation functions contain overlapping legality checks. These can disagree. A single authoritative legal-action generator is needed.

### Medium: No game history or replay log

Only snapshots exist. There is no durable record of:

- Who acted
- Which structured action was selected
- Before and after state identity
- Random choices
- Reward changes
- Turn boundaries
- Rule-generated side effects

Games therefore cannot currently be explained or reproduced from history.

### Medium: No tests or dependency definition

There are no unit tests, integration tests, `requirements.txt`, or project metadata. Syntax and imports pass, but rules and snapshot round trips are unverified.

## 7. Safest Order to Resume Development

1. **Freeze training and preserve current checkpoints.** Treat existing models and JSON files as historical artifacts. Do not use training behavior to validate engine correctness.

2. **Separate core game state from pygame and PyTorch.** A `Game` should be constructible without loading a model or opening a display. Move AI ownership out of `Player`; replace GUI error loops and `sys.exit` with ordinary exceptions or results.

3. **Define stable, structured action objects.** Examples include `ClaimPost(route_id, post_id, shape)`, `TakeIncome(circles)`, and `CompleteRoute(route_id, outcome, city_id)`. Keep the 619-index codec as an AI adapter, not the engine's native API.

4. **Create one authoritative legality interface.** `legal_actions(game)` should generate structured legal actions. The neural mask and manual UI validation should derive from the same collection.

5. **Make serialization complete and versioned.** Save only primitives and stable IDs. Include every field affecting future behavior, schema version, map definition version, pending compound-action state, RNG state, and current turn. Allow caller-selected filenames.

6. **Add invariant and round-trip tests.** For every representative state, `save -> load -> save` should preserve canonical state exactly. Legal actions before and after loading should match.

7. **Add an append-only action and event history.** Record structured actions plus deterministic consequences. Periodic snapshots plus an event log will support explanation, reproduction, and branching simulations.

8. **Build a headless simulator API.** It should load or clone a state, list legal actions, apply one manual action, ask a policy for one action, run until terminal, clone a starting position many times, and return final state plus complete history.

9. **Create separate entry points.** At minimum: manual pygame play, one-turn AI, full simulation, batch branching simulation, and training. None should execute merely because a module was imported.

10. **Only then repair RL.** Start with a random legal policy and rule-based baselines. Once full games and replay are reliable, implement a defined algorithm with reproducible evaluation.

## Recommended Immediate Milestone

The safest next milestone is not better AI. It is:

> A headless game can start, make only legal actions, reach a valid terminal state, and reproduce the same run from the same configuration and random seed.

After that, exact saved positions and replayable action history should be implemented before reinforcement-learning work resumes.
