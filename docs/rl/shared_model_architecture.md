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
accepts the authoritative 4,241-value observation and returns 768 action scores.
Inference-only model objects do not own an optimizer.

`GameConfiguration.create_game()` loads at most one `hansa_nn_model.pth` and
stores it on the game runtime. Every AI-controlled seat uses that same object.
Human-only games load no model, and `Player` has no model field or model-loading
behavior.

The GUI builds the acting player's observation, passes it to the shared model,
masks illegal actions, and selects among legal scores using the configured
difficulty. Checkpoints must contain matching action- and observation-schema
metadata; old seat-numbered or otherwise incompatible checkpoints are rejected.

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
selected index. The player-relative observation and action interfaces therefore
support the implemented shared inference policy.

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

## Remaining Training Plan

1. Complete issue #3 and select the training objective and credit-assignment
   rules.
2. Define a versioned trajectory record containing observation, mask, action,
   acting player, and final target. Integrate it with the action-history work
   without treating a replay trace as sufficient training data.
3. Add a dedicated headless self-play entry point that creates one shared model,
   freezes it during games, and records separate player trajectories.
4. Add one learner that updates the shared model only after the configured game
   or batch boundary.
5. Save one versioned checkpoint containing model state, optimizer state,
   action-schema identity, observation-schema identity, and training progress.
6. Verify that every seat uses the same model object during self-play, every
   sample has one acting owner, hidden information remains excluded, and no
   weights change before the update boundary.

Existing seat-numbered checkpoints should be treated as incompatible
experimental artifacts. Do not silently combine or migrate their weights. A
future migration tool may explicitly import one compatible checkpoint as an
initial shared model, but rejection is safer by default.

## Decision

Inference now uses one shared policy model for all Hansa players. Future
training will retain separate player-owned trajectories and one
learner-controlled optimizer. Reward targets, trajectory collection, and model
updates remain deliberately unimplemented.
