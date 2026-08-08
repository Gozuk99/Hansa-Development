# Reinforcement-Learning Rewards and Credit Assignment

## Decision

Training uses score-derived rewards after every selected interaction and winner-only rewards after final scoring. The shared model remains frozen while a game is active. Model updates happen only after completed games or batches.

This replaces the legacy collection of hand-written rewards for particular action types. The rules engine remains responsible for scoring; training observes changes in the engine's authoritative projected score instead of trying to duplicate the rules with reward constants.

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

The following have no fixed reward merely for being performed:

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

## Terminal Rewards

After final scoring:

- each eventual winner receives `100 ×` that winner's final total score;
- non-winners receive no terminal reward;
- the player whose decision triggered game end receives another `+150` only if that player is an eventual winner; and
- triggering game end while losing gives no terminal bonus.

Normal immediate rewards are never erased. A losing player keeps legitimate reward from the final decision even though their terminal reward is zero.

## Per-Decision Credit Assignment

Every `TrainingDecision` records:

- the acting player's visible observation;
- the training legal-action mask;
- the selected action index;
- the acting player;
- the projected-score reward delta for every player caused by that interaction;
- the acting player's immediate reward delta;
- the assigned policy tier and selection metadata; and
- the discounted reward-to-go calculated after the game.

Recording the full reward vector matters when completing a route awards a point to a non-acting city controller. That point remains in the receiving player's reward stream and can reinforce their earlier decisions; it is not reassigned to the player who completed the route.

After terminal rewards are known, training processes the global decision sequence backward while retaining a separate running return for every player. Terminal rewards are added to the final environment decision, then reward-to-go is calculated with:

```text
return = immediate reward + gamma × later return for the same player
```

Decisions by other players do not receive, erase, or replace that return. The initial discount is `gamma = 0.99`, stored in the training configuration and checkpoint.

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
- gamma is configurable and checkpointed; and
- model weights do not change during an active game.
