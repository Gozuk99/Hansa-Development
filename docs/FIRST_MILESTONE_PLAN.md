# First Milestone Plan: Run and Verify a Complete Game

## Goal

Make the existing game runnable through a deterministic, headless path and verify that a fresh game can begin, proceed through legal actions, and finish at a valid terminal state.

This milestone is about game-engine correctness and repeatable verification. It does not attempt to improve reinforcement-learning quality.

## Out of Scope

- Changing the neural-network architecture
- Improving rewards or the training algorithm
- Starting or automating training
- Replacing the existing 619-action encoding
- Redesigning the pygame interface
- Completing exact save/load support
- Building large-scale alternate-outcome simulation

Those should follow only after complete games run reliably.

## Proposed Files and Changes

### 1. Add a headless game runner

Proposed file:

- `game/game_runner.py`

Responsibilities:

- Construct a game from explicit `map_num`, `num_players`, and seed values.
- Obtain the current legal-action mask.
- Select a legal action using a deterministic baseline policy.
- Apply exactly one action through the existing action dispatcher.
- Continue until the normal game-ending condition is reached.
- Stop with a useful diagnostic if there are no legal actions, state stops progressing, an exception occurs, or a configurable step limit is exceeded.
- Return a structured result containing the terminal reason, action count, turn count, final scores, and action trace.

The initial baseline policy should be deliberately simple. It is a test driver, not an AI-quality benchmark.

### 2. Permit game construction without neural-network loading

Proposed changes:

- `player_info/player_attributes.py`
- `game/game_info.py`

Add an optional construction mode or injected player policy that allows the engine and tests to create players without instantiating `HansaNN` or loading model checkpoints.

Requirements:

- Preserve current behavior for the existing pygame program unless the new option is explicitly used.
- Do not save, replace, or otherwise alter existing `.pth` files.
- Keep policy/model ownership separate from rules wherever practical.

### 3. Make automated action failures non-blocking

Proposed change:

- `ai/action_options.py`

Changes:

- Replace blocking pygame error loops in the headless path with descriptive exceptions.
- Include the action index and relevant route, post, player, and pending-action context in failures.
- Retain visual debugging only when explicitly requested.
- Fix the call that invokes `error_exit(game)` without its required `route` argument.
- Do not redesign the action-index layout in this milestone.

### 4. Establish deterministic randomness

Proposed changes:

- `game/game_info.py`
- Relevant map-construction files under `map_data/`
- `game/game_runner.py`

Changes:

- Create a dedicated seeded random-number generator for each game.
- Use it for tile selection, mission-card assignment, initial bonus markers, and other setup choices exercised by the headless runner.
- Avoid dependence on process-global `random` state in the verified path.
- Record the seed and game configuration in every run result.

If passing a random generator throughout the existing map code would make this milestone too large, first isolate and document the remaining global-random calls. The determinism test must still cover the supported configuration.

### 5. Add engine invariants

Proposed file:

- `game/invariants.py`

Run invariant checks after setup and after every applied action. Initial checks should cover:

- Stock, supply, action, pickup, and placement counts are nonnegative.
- A route post has both an owner and piece shape, or neither.
- `current_player`, `current_player_index`, and `active_player` agree with the current phase.
- Office, post, privilege, tile, and bonus-marker owners reference players in the current game.
- Displacement flags and `DisplacedPlayer` fields form a consistent state.
- Pending multi-step flags do not enable contradictory workflows.
- Piece totals remain conserved where the implemented rules require conservation.
- A terminal game has final scoring data for every player.

Invariant failures should raise an exception with enough context to reproduce the step.

### 6. Add focused automated tests

Proposed files:

- `tests/test_game_setup.py`
- `tests/test_legal_actions.py`
- `tests/test_turn_progression.py`
- `tests/test_complete_game.py`
- `tests/test_determinism.py`

Test coverage:

#### Setup

- Supported map and player-count combinations construct successfully.
- Initial player, map, stock, supply, and turn invariants hold.
- Headless construction does not load or save model files.

#### Legal actions

- A fresh game exposes at least one legal action.
- The mask always has exactly 619 entries.
- Every action selected by the runner was legal immediately before execution.
- Legal action dispatch does not open a pygame window.

#### Turn progression

- Actions decrement or preserve action counters according to the current workflow.
- A completed turn advances to the expected player.
- Compound actions prevent premature turn switching.
- At least one controlled displacement scenario can finish its placement sequence.

#### Complete game

- At least one known map/player/seed configuration reaches the engine's normal terminal condition.
- It finishes below a generous action limit.
- No invariant fails during the run.
- Final scores exist for every player.

#### Determinism

- Running the same map, player count, seed, and baseline policy twice produces the same structured action trace.
- Final scores and terminal reason also match.
- A small set of different seeds completes without invariant violations.

Tests should initially target a narrow known-good configuration. Expand the map/player matrix only after failures are diagnosed rather than weakening assertions.

### 7. Add a safe command-line entry point

Proposed file:

- `run_headless_game.py`

Intended usage:

```powershell
python run_headless_game.py --map 2 --players 5 --seed 124
```

Expected output:

- Map and player count
- Seed
- Number of actions and turns
- Terminal reason
- Final scores
- Invariant result

This command must not initialize pygame, train a model, or write model checkpoints.

### 8. Document the verified path

Proposed change:

- `README.md`

Document:

- How to run engine tests
- How to run one deterministic headless game
- How to repeat a known seed
- Supported map/player configurations
- Known incomplete rules
- The fact that the command does not train or overwrite models

## Implementation Order

Keep the work reviewable as four small changesets:

1. Headless game/player construction
2. Non-blocking action dispatch and invariant checks
3. Deterministic runner and command-line entry point
4. Lifecycle, determinism, and complete-game tests

Each changeset should pass its relevant tests before proceeding.

## Acceptance Criteria

The milestone is complete when:

- Core engine tests run without opening pygame.
- No neural-network checkpoint is loaded or saved during engine tests.
- A fresh supported game has legal actions and begins normally.
- At least one supported configuration reaches a normal game-ending condition.
- A small documented configuration matrix completes without invariant violations.
- The same configuration and seed produce the same action trace, terminal reason, and final scores.
- Illegal or stuck states fail with descriptive diagnostics instead of hanging.
- Existing model files and training behavior remain untouched.
- Any unsupported configurations or incomplete rules are explicitly documented.

## Expected Follow-up

After this milestone, the next priority should be exact, versioned state serialization and structured action history. Reinforcement-learning work should resume only after saved positions, legal actions, and complete-game replay are reliable.
