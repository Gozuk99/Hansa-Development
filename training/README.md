# Targeted Self-Play Training

The first training loop starts from exact saved games that are already close to ending. This lets the model receive a useful result quickly instead of spending thousands of early experiments moving pieces without finishing a game.

Run the supplied Yellow-at-19-points example with:

```powershell
python tools/train_self_play.py --episodes 100 --batch-size 8
```

The command saves `hansa_nn_model.pth` in the project root, which is also where the game currently looks for its shared AI model. The checkpoint is ignored by Git because it is a large generated artifact.

Use `--resume` to continue from that checkpoint:

```powershell
python tools/train_self_play.py --resume --episodes 100
```

Training writes a resumable checkpoint every 100 games by default. Change that interval with `--checkpoint-every`. Resume always uses the learning settings stored in the checkpoint, preventing accidental changes halfway through an experiment.

Tiered training uses checkpoint format version 3. Earlier training checkpoints
remain usable for game inference when their action and observation schemas match,
but they cannot be resumed because they do not contain tier configuration or
tier metrics. Start a new tiered run to replace the canonical checkpoint.

## Training policy tiers

Every game randomly assigns a policy tier to each seat. All tiers use the same
shared model; a tier changes only how broadly it explores the model's legal
action rankings.

| Tier | Top-k pool | Epsilon |
| --- | ---: | ---: |
| 1 | 2 | 0.05 |
| 2 | 5 | 0.10 |
| 3 | 10 | 0.20 |
| 4 | 15 | 0.35 |
| 5 | all legal actions | 1.00 |

Three-player games use tiers 1, 3, and 5. Four-player games use tiers 1, 2, 4,
and 5. Five-player games use all five tiers. The seats are reshuffled for every
game, so no color, player number, or turn position permanently receives a
stronger or weaker policy.

An epsilon choice is uniform across every legal action. Otherwise, selection is
uniform within the tier's highest-ranked legal actions. Tier 5 is fully random.
The tier definitions and player-count subsets are part of `TrainingConfig` and
are saved in the checkpoint.

Pass `--state` more than once to rotate through several starting positions:

```powershell
python tools/train_self_play.py --state training_data/first.hansa --state training_data/second.hansa
```

## What "disable Move" means

By default, the training policy removes interactions that would begin the normal Move action by selecting one of the acting player's occupied posts. It does not change the engine's legal-action rules. Mandatory placement steps are never removed, and placing from personal supply, Income, displacement, route completion, bonus markers, and end-turn controls remain available. If Move is the only legal choice, those interactions are restored so this training preference cannot create an artificial stuck state.

For the supplied example, all 34 post interactions would begin Move. Removing them leaves exactly one choice: complete the prepared route and end the game. Use `--allow-move-action` to train without this temporary restriction.

## Current learning signal

Each interaction records `100 ×` the acting player's change in authoritative projected final prestige. After the game, winners receive another `100 ×` their final score, and a player who both triggers game end and wins receives `+150`.

A normal Income action also receives a proportional negative reward when the
player lacks enough pieces in General Stock to use their full finite Bank
capacity. Full-capacity Income and Bank `C/all` receive no penalty. The default
penalty scale is `100` and is stored in `TrainingConfig` and the checkpoint.
Set a different scale for a new run with `--income-penalty-scale`.

Training calculates discounted reward-to-go separately for each player instead of assigning one whole-game total to every decision. Gamma defaults to `0.99` and can be set for a new run with `--gamma`.

Training progress reports games, wins, selection counts, average selected rank,
average immediate reward, and average reward-to-go by assigned tier rather than
by seat.
