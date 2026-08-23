# Reinforcement-Learning Rewards and Credit Assignment

## Decision

Training uses score-derived rewards after every selected interaction and winner-only rewards after final scoring. The shared model remains frozen while a game is active. Training performs one update per started block of 256 trajectory decisions, capped at four non-overlapping representative updates and 1,024 sampled decisions per game. Penalized no-replacement-route failures and action-limit trajectories also train; an action-limit trajectory keeps its genuine rewards and penalties but receives no invented terminal reward. Evaluation games never update the model.

Representative sampling keeps every pickup and placement from a selected normal
Move or permanent Move Any 2 workflow together. A movement penalty is still
recorded only once when the workflow completes. Objectively pointless movement
workflows receive a local negative target instead of inheriting terminal credit;
grouping keeps every interaction that produced that outcome in the same sample.
Paid normal-Move workflows also begin without terminal credit. They regain it
only when the next non-Move paid action claims a route completed by their
uninterrupted same-turn Move chain, and only for workflows that contributed to
that route. Permanent Move Any 2 is unaffected.

Authoritative projected scoring provides the main reward signal. A smaller set of
explicit training-only rewards and penalties supplements it where score changes
alone do not describe efficiency or short-term progress. These adjustments do
not alter game rules or scoring.

## Primary Objective, Placement, and Player Count

The primary objective is to win. A winner's terminal reward scales with final
prestige, so winning with a stronger score is better than winning with a weaker
score. Second through fifth place receive no placement reward; they retain only
their legitimate decision rewards. This avoids teaching the model to settle for
second place instead of pursuing a possible win.

The engine applies the rulebook tie-breakers before training identifies the
winner. If those tie-breakers still leave shared winners, every remaining winner
receives the winner terminal reward. Reward constants are identical in three-,
four-, and five-player games; player count changes the competitive environment,
not the value of a prestige point.

## Previous Reward Mutation Inventory

Before this design, `Player.reward` was changed in the following places.

### `game/game_actions.py`

- `claim_post_action()` rewarded placing near a normal bonus marker, permanent marker, upgrade city, or ordinary route.
- `displace_action()` applied the same location rewards with arbitrary displacement deductions.
- `move_action()` removed and restored location rewards as pieces moved, with an extra early-finish penalty.
- `score_route()` added 100 when the acting player received a route point and subtracted 100 when an opponent received one. The opponent's own reward was not increased.
- `claim_route_for_office()` gave a fixed office reward.
- `claim_route_for_office()` and `claim_route_for_additional_office()` gave a fixed Additional Trading Post reward.
- `claim_route_for_upgrade()` gave fixed rewards for ability upgrades or a prestige-space claim.
- `handle_bonus_marker()` gave fixed rewards for collecting a normal or permanent marker and an arbitrary penalty when Claim Green City could not be resolved.

### `player_info/reward_options.py`

- `Rewards.get_end_game_placement_RL_rewards()` added fixed placement rewards of `+2000`, `+500`, `0`, `-500`, and `-2000`.
- The same method added `+150` to the active player for triggering game end, even if that player did not win.
- Numerous configured values were never connected to gameplay, including completed-ability, East-West, city-control, displacement, and most bonus-marker-use rewards.

### `training/self_play.py`

- Training called terminal rewards indirectly through `game.players[0].reward_structure`.
- It subtracted the loaded state's starting `Player.reward` total from the final total.
- It assigned that same whole-game player return to every decision made by that player.

These paths mixed rules, training policy, and credit assignment. They also rewarded strategically neutral actions, omitted several scoring sources, and could not identify which decision produced a reward.

## Authoritative Projected Prestige

After each selected interaction, training compares the acting player's projected final prestige total with the value immediately before the interaction.

The projection uses the same calculation as final scoring:

- current in-game prestige, including gold-coin offices and East-West awards;
- four points for each completed non-Keys ability;
- Emperor's Favour additions to completed abilities and controlled cities;
- the current bonus-marker set score;
- printed Coellen/special-prestige values;
- two points per controlled city;
- current largest-network size multiplied by Keys;
- Mission Card projection;
- Britannia regional projection; and
- every other category included by authoritative final scoring.

The immediate reward for a decision is:

```text
100 × (projected score after the decision - projected score before the decision)
```

This naturally produces positive or negative rewards. Examples include:

- gold-coin office: `+100`;
- East-West: `+700`, `+400`, or `+200`;
- completing an ability: normally `+400`;
- gaining city control: normally `+200`;
- losing city control: normally `-200`;
- bonus-marker reward: `100 ×` the change in the marker-set score;
- special-prestige space: `100 ×` its printed value; and
- network, mission, Britannia, or Emperor scoring: `100 ×` the projected change.

If one interaction changes several categories, its reward is the net projected-score change. Opponents' score losses do not become an extra denial reward for the acting player.

## Neutral Actions

The following receive no automatic base reward merely for being performed.
Specific results can still earn the focused training rewards or penalties
documented below:

- Income;
- normal Move;
- displacement or being displaced;
- Move 3;
- +3 or +4 Actions;
- Swap Office;
- Block Route;
- buying Emperor's Favour; and
- blocking or denying an opponent.

They still receive a reward when their completed interaction changes the acting player's authoritative projected score.

### Income efficiency

Income has no positive reward merely for being performed. During training, a
normal Income decision receives a configurable penalty when a finite Bank is not
fully used because General Stock contains too few pieces:

```text
-income_penalty_scale × (Bank capacity - pieces received) / Bank capacity
```

The default scale is `100`. A full-capacity Income has no penalty, and Bank
`C/all` is exempt because it always takes every available piece. The adjustment
belongs only to that Income decision and does not change Income legality, piece
movement, or later reward-to-go.

### Current training-only adjustments

The current trainer also applies these deliberately shaped signals:

- `+50` for completing a route, before subtracting prestige awarded to opponents;
- `+50` for each net claimable route produced by a completed normal Move, or
  `-50` for each previously claimable route broken by that Move;
- `+10` for concentrating at least two moved pieces on one route. Each player
  can earn this only once per route until that route is claimed;
- `+250` when the player's next paid action claims one of the routes that the
  preceding Move made fully controlled. Filling multiple routes still awards
  the combination once, and taking any other paid action first forfeits it;
- `-250` when a paid route claim gives the acting player no office, upgrade,
  bonus marker, permanent-marker benefit, or actual/projected points;
- `+25` for disrupting an immediately following player's valuable completed route;
- `+5` for placing on a route where the player already has a piece, or `+3` when
  doing so through displacement;
- `+250` for an intermediate Privilege, Book, Actions, or Bank upgrade, except
  the first Actions upgrade receives `+400`;
- a workflow-local additive `-200` for moving only one piece, `-100` for moving
  only two when the player's Book permits at least three, and `-200` for a
  second consecutive Move when Book permits at least four. These adjustments
  modify only the offending Move's calculated return and do not enter the
  reward stream for earlier decisions;
- a local `-1,000` target for a normal Move or permanent Move Any 2 workflow
  that leaves every affected post with the same owner and shape as before, or
  merely rearranges the same owner/shape totals among equivalent posts of one
  non-maritime route. Maritime posts remain distinct;
- a local `-1,500` target for the third and every subsequent consecutive
  completed normal Move. A different normal action or a new turn resets the
  sequence; and
- a local `-500` target for every normal Move workflow in a completed turn when
  the player spent at least two actions and every paid action was Move. Any
  other paid action prevents this whole-turn penalty; and
- `-500` for the player responsible for leaving no legal route on which to place
  a required replacement bonus marker.

These are training signals, not Hansa prestige. They are tested separately from
the authoritative score projection.

## Terminal Rewards

After final scoring:

- each eventual winner receives `100 ×` that winner's final total score;
- non-winners receive no terminal reward;
- the player whose decision triggered game end receives another `+150` only if that player is an eventual winner; and
- triggering game end while losing gives no terminal bonus.

Normal decisions retain their immediate and terminal rewards. A losing player
keeps legitimate reward from the final decision even though their terminal
reward is zero. Paid normal-Move workflows retain their immediate and later
non-terminal rewards but receive terminal credit only through the immediate
Move-to-claim chain described above. Explicitly identified pointless movement
patterns instead use their local negative training target.

## Per-Decision Credit Assignment

Every `TrainingDecision` records:

- the acting player's visible observation;
- the training legal-action mask;
- the selected action index;
- the acting player;
- the game turn containing the decision;
- the projected-score reward delta for every player caused by that interaction;
- the acting player's immediate reward delta;
- an optional workflow-local movement target;
- an optional workflow-local additive movement adjustment;
- whether the decision may inherit terminal credit;
- the assigned policy tier and selection metadata; and
- the discounted reward-to-go calculated after the game.

Recording the full reward vector matters when completing a route awards a point to a non-acting city controller. That point remains in the receiving player's reward stream and can reinforce their earlier decisions; it is not reassigned to the player who completed the route.

After terminal rewards are known, training processes the global decision sequence backward while retaining a separate running return for every player. Terminal rewards are added to the final environment decision, then reward-to-go is calculated with:

```text
return = immediate reward + gamma × later return for the same player
```

All interactions within one player turn have the same discount distance to later
rewards. Gamma is applied only when that player begins another turn. Decisions by
other players do not discount, receive, erase, or replace that return. The initial
discount is `gamma = 0.99`, stored in the training configuration and checkpoint.

After normal reward-to-go is calculated, small Move-efficiency penalties are
added only to the offending grouped workflow's target. Explicitly flagged hard
movement mistakes instead replace that workflow's calculated target. Neither
kind enters the player's chronological reward stream, so earlier productive
non-Move decisions keep their score and terminal credit. Normal Move still uses
the narrower Move-to-claim terminal-credit rule above.

## Worked Route Example

Red completes a route without taking an office, upgrade, marker, or immediate
prestige. Blue controls an endpoint city and therefore gains one prestige point.
The choice also blocks Green's longer-term plan. Red later wins the game.

For that decision, Red receives the `+50` route-completion signal, pays `-100`
for the point awarded to Blue, and receives no separate reward merely for
blocking Green. Blue's `+100` projected-score change belongs to Blue's reward
stream; it is never credited to Red. Red's eventual winner reward flows back to
Red's earlier decisions through reward-to-go, so this initially negative decision
can still receive positive long-term credit when it contributes to the win.

## Known Reward Risks

Shaped rewards make early learning easier, but they can also be exploited. A
model could repeatedly pursue small placement, route, movement, or upgrade
rewards while reducing its chance of winning. Large penalties can similarly make
the model avoid a strategically correct sacrifice. Training therefore evaluates
fixed games by win rate, final score, and game length in addition to loss. A
falling loss alone is not evidence that the policy is improving.

Reward leakage is controlled by recording a reward vector for every interaction
and maintaining a separate return for every player. Opponent rewards never enter
the acting player's return unless an explicit training rule, such as paying for
prestige awarded during a route completion, intentionally does so.

## Safety and Testing

Tests must verify:

- every former `Player.reward` mutation is removed or disabled;
- projected scoring and final scoring share one implementation;
- each scoring category produces its exact `100 × prestige` delta;
- neutral actions remain neutral when projected score does not change;
- negative projected changes produce negative rewards;
- terminal rewards go only to winners;
- the end-game trigger bonus goes only to a triggering winner;
- losing players retain legitimate immediate rewards;
- per-player reward-to-go does not leak between seats;
- interactions within one player turn do not discount each other;
- gamma is configurable and checkpointed; and
- model weights do not change during an active game.
