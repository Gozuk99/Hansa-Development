# Shared Reinforcement-Learning Model Architecture

## Decision

Use one shared Hansa model for every AI-controlled seat. Each decision and reward remains associated with the player who made it; sharing the model does not mix observations, private information, or credit between players.

## Model Ownership

`HansaNN` accepts the fixed player-visible observation. Its existing
`4724 -> 2048 -> 1024 -> 1024` trunk feeds two independent `1024 -> 768` outputs.
The final shared layer is initialized as an exact identity transform with zero
bias so migrated models initially preserve their prior representation:

- the mature Q head (`layer3`) estimates expected return for every universal action;
- the policy head produces relative action-preference logits.

`GameConfiguration` loads at most one shared inference model. Human-only games do not load PyTorch models. The Q head remains the sole authority for gameplay selection during the policy experiment; policy output is shadow-only.

Training owns one shared model and one optimizer through `SelfPlayTrainer`, outside the game engine. The optimizer has a base-rate trunk/Q group and a separately configurable, initially faster policy-head group. The model is in evaluation mode and its weights remain frozen while a game is collected. Training performs one update per sampled block of 256 trajectory decisions. Mid, late, and end games are capped at four non-overlapping representative blocks and 1,024 sampled decisions; generated early games are capped at 4,096 sampled decisions. Early sampling targets 512 decisions from each chronological eighth of the complete trajectory, then redistributes unused capacity without splitting movement workflows. Penalized no-replacement-route failures and action-limit trajectories also train before the next game begins. Action-limit trajectories retain genuine rewards and penalties but receive no invented terminal reward; evaluation games never update the model.

## Policy objective

The policy head is trained from finalized trajectory quality, not from the action selector. For each sampled decision:

1. reward-to-go and any existing local movement target/adjustment are calculated exactly as for Q training;
2. the signed quality weight is `tanh(reward_to_go / policy_return_scale)`;
3. illegal actions are excluded;
4. legal equivalent action indices are collapsed to one semantic choice using their mean logit before normalization;
5. positive quality uses `-quality_weight * log(P(chosen semantic group))`;
6. negative quality uses `-abs(quality_weight) * log(1 - P(chosen semantic group))`;
7. a forced single semantic choice has zero policy loss.

Thus positive quality increases the chosen group's preference, negative quality decreases it, and weak/near-zero evidence makes a small update. `tanh` bounds extreme reward magnitudes without changing Q targets. No state-value head or Q-derived baseline is used: that keeps the new signal independent of Q rankings and preserves the sign of explicit hard local targets. Batch policy loss is the mean decision loss.

The reported joint objective is `q_loss + policy_loss_weight * policy_loss`, but its gradient sources are isolated during shadow training. Q loss updates and clips the complete shared trunk—including the final identity-initialized 1024-unit layer—and Q head normally. The policy head consumes detached output from that final shared layer, so policy loss updates and clips only the policy head at its own learning-rate multiplier. Policy gradients cannot alter any shared layer, the Q head, Q clipping, or Q optimizer state.

## Player-Visible Decisions

`ObservationEncoder.build(game)` returns:

- the fixed-size feature tensor from the acting player's perspective;
- the acting player's 768-entry legal-action mask; and
- the observer's seat index.

Player data and ownership identifiers are relative to the observer. Public state
is visible to every player, while private Mission Cards are encoded only for
their owner. Opponents' used bonus-marker counts are visible, but their
face-down identities remain hidden unless that opponent is the selected
Exchange Bonus Marker target.

The engine owns legality through `Game.get_legal_actions()`. The central codec maps stable interactions to indices, and `Game.apply_ai_action()` executes the selected index. Both heads always emit 768 values; policy normalization happens only over the current legal semantic choices. GUI code is not involved in headless inference or training.

## Training Trajectories

`TrainingDecision` records:

- the acting player's visible observation;
- the training legal-action mask;
- the selected 768-schema action;
- the acting seat;
- the game turn containing the decision;
- every player's projected-score delta caused by the interaction;
- the acting player's immediate reward; and
- the acting player's discounted reward-to-go; and
- an optional hard local target for an objectively pointless grouped movement workflow;
- an optional additive local adjustment for a completed Move's small efficiency penalty; and
- whether the decision may inherit terminal credit. Paid normal-Move workflows
  gain that credit only through an immediate claim of a route they helped complete.

`CompletedTrajectory` adds terminal rewards, final scores, winners, and the replayable action trace. This is separate from `ReplayRecord`, which remains a compact deterministic action trace rather than a training record.

## Reward Ownership

The authoritative design is in `docs/rl/reward_structure.md`.

Every player's reward stream remains separate. If Green acts and Blue receives a route point, the point enters Blue's reward stream; it is not credited to Green. When Green's decision is trained, its target is Green's discounted reward-to-go.

After a game, samples from all seats may be combined because ownership is already correct and every sample trains the same shared model.

## Tiered collection policy

Training assigns policy tiers randomly to seats for each game. The tiers do not
own separate models: each one samples differently from the legal rankings of the
same frozen `HansaNN`. Three-player training uses tiers 1/2 plus one uniformly
selected tier from 3/4/5. Four-player training uses 1/2/4/5, and five-player
training uses 1/2/3/4/5. Fixed three-player evaluation retains 1/3/5 for historical
comparability.
Outside epsilon exploration, each tier samples its Top-K semantic choices using
normalized `1 / sqrt(rank)` weights. Staged workflow selection keeps its separate
bounded-exploration policy.

Training selects one exploration mode per game independently of its curriculum
maturity. By default, 95% of games use each tier's configured epsilon and 5%
override every participating tier's effective epsilon to zero. Zero-epsilon
games still sample within the tier's normal Top-K using the same rank weights;
they are neither greedy Top-1 games nor evaluation games.

The maturity schedule is 50% fresh, 25% early, 10% mid, 10% late, and 5% end.
Fresh trajectories begin from the canonical untouched new-game setup.

Each decision records its tier, epsilon, top-k setting, selection method, model
rank, and legal-action count. Training progress aggregates wins, games,
selection behavior, and rewards by tier so later curriculum changes can be based
on measured results rather than seat order.

Shadow diagnostics compare the policy distribution with the Q ranking without consuming RNG or changing the selected action. Compact CSV fields report Q/policy top-1 agreement, the policy top choice's Q rank, policy entropy, cumulative policy mass in the top 2/5/10 choices, Q loss, policy loss, combined loss, and the fixed zero policy-to-trunk gradient scale.

Each completed trajectory also records movement-behavior diagnostics at the
trainer's existing workflow boundaries: paid Move share, pointless workflows,
repeated-Move penalties, all-Move turns, Moves that create a claimable route, and
immediate Move-to-Claim conversions. These metrics are logged for training and
evaluation but do not alter reward or target assignment.

## Checkpoints

Checkpoint schema version 6 stores:

- shared trunk, Q-head, and policy-head weights;
- both optimizer parameter groups and their state;
- training progress and loss statistics;
- policy RNG state;
- training configuration, including gamma and tier definitions;
- source-state hashes; and
- policy objective, learning rate, and policy-specific update age;
- exact observation and action schema identities.

Training checkpoint version 5 and model artifacts without policy parameters have an explicit Q-only migration. Existing trunk/Q weights are loaded exactly, their compatible Adam state is retained, and only the new policy head/optimizer state is initialized. A migrated policy head starts with zero weights and bias, producing neutral equal logits before learning. Its separately persisted policy-update counter starts at zero, while policy-to-trunk influence remains fixed at zero throughout shadow training. Newly saved artifacts carry the dual-head model schema version. Incompatible checkpoints fail clearly. The shape-compatible observation-v1
model/checkpoint has one explicit transfer path into observation version 2 so
trained weights are preserved while opponent used-marker identities become
hidden. The next save records version 2. Older seat-numbered or legacy reward
checkpoints are not silently migrated.

## Remaining Work

1. Use the separately reported fixed early-game and mid/late/end evaluation
   sets to measure changes before adjusting exploration, rewards, discounting,
   or the network.
2. Expand evaluation coverage when a new strategic focus is added, while
   retaining old fixed positions for historical comparison.




## When to Try a Different Neural-Network Architecture

Do not change the NN simply because training plateaus. First verify that the evaluation method, reward structure, curriculum, exploration, and training data are behaving correctly.

The current architecture should remain the baseline until there is measurable evidence that it is the limiting factor.

### Try a wider network when:

The model appears unable to retain enough different useful patterns at once, or performance improves when more capacity is provided.

Example:

`4724 → 4096 → 2048 → 768`

Wider means more neurons per layer and therefore more capacity, but also more parameters and computation.

### Try a deeper network when:

The observation contains the necessary information, training is healthy, but the model consistently struggles with combinations of otherwise-learned concepts or more complicated strategic relationships.

Example:

`4724 → 2048 → 1024 → 512 → 768`

Deeper means more stages in which learned patterns can be combined. More depth is not automatically better.

### Try a smaller or shallower network when:

A smaller model reaches approximately the same evaluation performance as the current model.

Example:

`4724 → 1024 → 512 → 768`

If it performs equally well, the larger model is probably wasting computation and may be harder to train.

### Try a bottleneck when:

There is a specific reason to force the model to learn a compact internal representation.

Example:

`4724 → 2048 → 256 → 768`

Do not make the bottleneck extremely small without a specific reason. A layer such as `1`, `2`, or `4` neurons would force almost the entire game state through only a few values and would likely destroy information needed to distinguish hundreds of actions.

### Most important rule

A plateau is a reason to investigate, not a reason by itself to redesign the NN.

Change the architecture only after other likely causes of the plateau have been ruled out and evaluation results suggest that model capacity or representation is actually limiting performance.
