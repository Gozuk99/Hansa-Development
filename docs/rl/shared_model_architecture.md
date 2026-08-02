# Shared Reinforcement-Learning Model Architecture

This document resolves issue #1. It describes the current implementation and
the recommended ownership model for future training. It does not implement
training or change any model checkpoint.

## Recommendation

Use one shared Hansa model for every AI-controlled seat.

Each decision is still recorded for the player who made it. Sharing the model
means that all players use the same learned policy; it does not mix their
observations, actions, or results together. A training record must identify its
acting player and must receive only that player's eventual training target.

This is the best fit for the current observation and action interfaces:

- Every player has the same possible interactions through the fixed 768-entry
  action schema.
- `ObservationEncoder` rotates the players so the observer is always relative
  player zero. The same network can therefore interpret every seat consistently.
- The observation exposes only information visible to that observer. In
  particular, another player's secret mission card is not encoded.
- The legal-action mask is produced for the acting player and accompanies the
  same observation.

A separate model per seat would learn artificial differences between Player 1,
Player 2, and Player 3 even though they follow the same rules. It would also
divide experience among several models, require several checkpoints, and make
evaluation harder without providing a Hansa-specific benefit.

## Current Implementation

### Model ownership

`HansaNN` is a three-layer network with one output per action-schema entry. It
also constructs its own Adam optimizer.

The interactive configuration currently creates a separate `HansaNN` object
for every AI player. `GameConfiguration.create_game()` calls
`_load_ai_model(player.order)`, which looks for seat-specific files such as
`hansa_nn_model2.pth`. `Player.__init__()` contains the same older seat-specific
loading behavior when `load_model=True`.

In a configured game, each AI-controlled seat therefore receives one model and
each Human seat receives none. The older direct `Player(load_model=True)` path
instead constructs one model for every player created through that path.

Consequently, two AI seats currently have:

- separate model objects;
- separate randomly initialized weights when files are absent;
- separate optimizers; and
- separate seat-numbered checkpoint names.

The GUI is intended to perform inference only: it builds the acting player's
observation, passes it to that player's model, and selects among legal actions
using the configured difficulty. That path is currently broken before scoring.
`ObservationEncoder.FEATURE_SIZE` is 4,241, while `INPUT_SIZE` remains 4,445;
the configured `HansaNN` therefore expects the wrong input width and an AI turn
fails with a matrix-shape error. This must be corrected and versioned before AI
inference or training can be considered usable.

### Observations and actions

`ObservationEncoder.build(game)` returns:

- a fixed-size feature tensor from the acting player's perspective;
- the acting player's 768-entry legal-action mask; and
- the observer's player index.

Player data is rotated relative to the observer. Ownership identifiers are also
relative, so the same board relationship has the same meaning regardless of
seat or colour. Public state is visible to every observer, while the observer's
own mission card is visible only to that observer.

The engine owns legal actions through `Game.get_legal_actions()`. The central
codec maps them to stable indices, and `Game.apply_ai_action()` executes the
selected index. The player-relative observation design and action interface
support a shared policy after the input-width compatibility defect is fixed.

### Current trajectory format

There is no reinforcement-learning trajectory format yet.

`GameRunResult.action_trace` and `ReplayRecord.action_trace` store an ordered
tuple of action indices for deterministic replay. They also retain the map,
player count, seed, and action-schema identity. They do not store:

- the observation used for each choice;
- the acting player for each choice;
- the legal-action mask at that choice;
- rewards or final-result targets; or
- policy scores, probabilities, or optimizer data.

The replay trace is useful evidence and may later be referenced by training
data, but it is not itself sufficient training data.

### Current reward flow

Every `Player` owns a mutable `reward` total and a `Rewards` configuration.
Gameplay functions directly add or subtract shaped values for placements,
movement, route completion, offices, upgrades, and bonus markers.

`Rewards.get_end_game_placement_RL_rewards()` defines placement rewards, but no
active production path calls it. No code associates a reward with a recorded
decision, calculates returns, computes a loss, calls `backward()`, or updates an
optimizer. Earlier training examples in `ai/ai_model.py` are comments only.

Therefore, the current project contains an inference path, but its input-width
mismatch currently prevents it from running successfully, and no training path
exists. The existing reward totals must not be treated as a verified learning
signal until issue #3 defines reward ownership and credit assignment.

## Recommended Training Ownership

Create one model and one optimizer in the future training process, outside the
game engine. Give every AI seat a reference to that same frozen model while a
self-play batch is being generated.

Keep a separate trajectory for each player in each game. At every model
decision, record at least:

- observation features;
- legal-action mask;
- selected 768-schema action index;
- acting player identity;
- game and decision identifiers;
- action-schema and observation-schema identities; and
- the final target assigned to that acting player.

The acting player's identity is bookkeeping, not a secret network feature. The
observation is already normalized so the acting player occupies the same
relative position for every decision.

During one game or rollout batch, model weights must remain frozen. After all
required trajectories are complete, the learner may combine samples from every
seat, perform an update, and save one new checkpoint. Historical or evaluation
models must remain separate frozen model instances and must never share the
training optimizer.

## Reward Association

Each recorded decision belongs to exactly one acting player. A later reward or
final outcome must be attached only to that player's trajectory before samples
from different players are combined.

For example, if Green chooses an action, that sample uses Green's eventual
target. Blue's or Red's final result must not be written onto Green's sample.
All samples may train the same shared network after their ownership is correct.

The exact target—win, placement, score difference, return, or another
combination—is intentionally unresolved here. Issue #3 must define it before a
training loop is implemented.

## Migration Plan

1. Complete issue #3 and select the training objective and credit-assignment
   rules.
2. Define a versioned trajectory record containing observation, mask, action,
   acting player, and final target. Integrate it with the action-history work
   without treating a replay trace as sufficient training data.
3. Replace the stale 4,445 input constant with the versioned 4,241 observation
   contract, and reject checkpoints whose observation identity or input width
   does not match. Current checkpoints contain action-schema metadata but no
   observation-schema identity.
4. Change AI construction so a caller supplies a model to AI seats. Remove
   seat-numbered model ownership from `Player` and `GameConfiguration`, and
   remove optimizer ownership from inference-only `HansaNN` instances.
5. Add a dedicated headless self-play entry point that creates one shared model,
   freezes it during games, and records separate player trajectories.
6. Add one learner that updates the shared model only after the configured game
   or batch boundary.
7. Save one versioned checkpoint containing model state, optimizer state,
   action-schema identity, observation-schema identity, and training progress.
8. Verify that every seat uses the same model object during self-play, every
   sample has one acting owner, hidden information remains excluded, and no
   weights change before the update boundary.

Existing seat-numbered checkpoints should be treated as incompatible
experimental artifacts. Do not silently combine or migrate their weights. A
future migration tool may explicitly import one compatible checkpoint as an
initial shared model, but rejection is safer by default.

## Decision

The future training architecture will use one shared policy model for all Hansa
players, with separate player-owned trajectories and one learner-controlled
optimizer. The current observation and action interfaces support this design;
the current input-size constant, model-loading, and reward code do not yet
implement a usable shared training path.
