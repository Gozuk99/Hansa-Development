# Shared Reinforcement-Learning Model Architecture

## Decision

Use one shared Hansa model for every AI-controlled seat. Each decision and reward remains associated with the player who made it; sharing the model does not mix observations, private information, or credit between players.

## Model Ownership

`HansaNN` accepts the fixed player-visible observation and produces one value for each entry in the 768-action schema. `GameConfiguration` loads at most one shared inference model. Human-only games do not load PyTorch models.

Training owns one shared model and one optimizer through `SelfPlayTrainer`, outside the game engine. The model is in evaluation mode and its weights remain frozen while a game is collected. Updates happen only between completed games or batches.

## Player-Visible Decisions

`ObservationEncoder.build(game)` returns:

- the fixed-size feature tensor from the acting player's perspective;
- the acting player's 768-entry legal-action mask; and
- the observer's seat index.

Player data and ownership identifiers are relative to the observer. Public state is visible to every player, while private Mission Cards are encoded only for their owner.

The engine owns legality through `Game.get_legal_actions()`. The central codec maps stable interactions to indices, and `Game.apply_ai_action()` executes the selected index. GUI code is not involved in headless inference or training.

## Training Trajectories

`TrainingDecision` records:

- the acting player's visible observation;
- the training legal-action mask;
- the selected 768-schema action;
- the acting seat;
- every player's projected-score delta caused by the interaction;
- the acting player's immediate reward; and
- the acting player's discounted reward-to-go.

`CompletedTrajectory` adds terminal rewards, final scores, winners, and the replayable action trace. This is separate from `ReplayRecord`, which remains a compact deterministic action trace rather than a training record.

## Reward Ownership

The authoritative design is in `docs/rl/reward_structure.md`.

Every player's reward stream remains separate. If Green acts and Blue receives a route point, the point enters Blue's reward stream; it is not credited to Green. When Green's decision is trained, its target is Green's discounted reward-to-go.

After a game, samples from all seats may be combined because ownership is already correct and every sample trains the same shared model.

## Tiered collection policy

Training assigns policy tiers randomly to seats for each game. The tiers do not
own separate models: each one samples differently from the legal rankings of the
same frozen `HansaNN`. Three-player games use tiers 1/3/5, four-player games use
1/2/4/5, and five-player games use 1/2/3/4/5.

Each decision records its tier, epsilon, top-k setting, selection method, model
rank, and legal-action count. Training progress aggregates wins, games,
selection behavior, and rewards by tier so later curriculum changes can be based
on measured results rather than seat order.

## Checkpoints

One versioned checkpoint stores:

- shared model weights;
- optimizer state;
- training progress and loss statistics;
- policy RNG state;
- training configuration, including gamma and tier definitions;
- source-state hashes; and
- exact observation and action schema identities.

Incompatible checkpoints fail clearly. Older seat-numbered or legacy reward checkpoints are not silently migrated.

## Remaining Work

1. Expand training beyond targeted near-end positions to full games.
2. Add held-out evaluation against frozen checkpoints and baseline policies.
3. Tune exploration, discounting, batches, and the network only after evaluation can measure improvement.
