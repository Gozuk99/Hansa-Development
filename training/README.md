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

Curriculum training uses checkpoint format version 5. Earlier training checkpoints
remain usable for game inference when their action and observation schemas match,
but they cannot be resumed because they do not contain the complete tier, reward,
and curriculum configuration.

## Standalone curriculum runner

Run one complete local curriculum iteration with:

```powershell
python tools/run_curriculum_training.py
```

The command generates temporary states, collects tiered training games, updates
the shared model between completed games, evaluates without updates on newly
generated positions at the same difficulty, saves resumable training state to
`training_output/curriculum/training_checkpoint.pth`, exports the successfully
trained playable model to `hansa_nn_model.pth`, and appends graphable results to
`training_output/curriculum/results.csv`. It does not call Codex, OpenAI, or any
other external AI service.

Progress is printed as each training and evaluation game finishes, followed by
the latest loss and the locations of the playable model and results file.
Evaluation uses newly generated maps, player counts, options, and positions from
the current training stage. It uses the same top-2, top-5, top-10, top-15, and
top-20 tiers as training, but disables epsilon exploration so the benchmark
measures the model without extra exploratory moves.

Each learning row records that game's loss before its per-game update and the
rolling average of the latest five learning-game losses. Test-only rows leave
both loss columns blank because evaluation never updates the model.

By default it runs five learning games followed by one test-only game. Set the
number of learning games with `--iterations`; one test-only game runs after all
of them. Use `--batch` to repeat that entire learning-and-test cycle.
The model learns after every completed game. Each game receives one equally
weighted update using at most 256 representative decisions; the final decision
and the strongest immediate rewards are always retained. `--batch-size` controls how
many completed games pass between disk saves. After each save group (five
learning games by default), the runner saves its
recovery data, exports the playable model, and adds those games to the CSV. It
saves the final recovery information and CSV row after the test-only game, but
does not rewrite the unchanged playable model.
The command automatically resumes interrupted training when recovery information
exists; otherwise it starts from the current playable model.

Use `--fresh` to begin a new experiment with untrained model weights. This removes
the previous curriculum recovery file and CSV results. The root playable model is
replaced only after the first successful save group:

```powershell
python tools/run_curriculum_training.py --fresh --iterations 100 --batch 5
```

Nine out of every ten generated training positions begin approximately one full
round before the prepared ending player acts. The tenth is an immediate-finish
lesson. The CSV records these as `one_round_before` and `immediate_finish` so their
results can be compared separately. The fixed evaluation suite uses eighteen
one-round positions and three immediate-finish positions. Its added scoring
positions cover bounded short, medium, and long East-West paths plus Wales,
Scotland, and dual-region Isle of Man contests.

For example, `--batch 10 --iterations 100` runs ten batches. Each batch contains
100 learning games followed by the 21 fixed test-only games, for 1,000 learning
games and 210 test-only games total.

Stages progress from 18–19 points through late-, mid-, early-, and full-game
positions. Promotion considers invalid actions, unfinished games, fixed-state
evaluation completion, Tier 1 performance, and rolling loss. Failed promotion
repeats the current stage. Action-limit failures are retried with a new
deterministic state; diagnostic bundles are preserved under
`training_data/failures/`.

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
| 5 | 20 | 0.35 |

Three-player games use tiers 1, 3, and 5. Four-player games use tiers 1, 2, 4,
and 5. Five-player games use all five tiers. The seats are reshuffled for every
game, so no color, player number, or turn position permanently receives a
stronger or weaker policy.

An epsilon choice is uniform across every legal action. Otherwise, selection is
uniform within the tier's highest-ranked legal actions. Tier 5 explores more
broadly than the stronger tiers without choosing uniformly from every legal action.
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

Training calculates discounted reward-to-go separately for each player instead of assigning one whole-game total to every decision. Gamma is applied once when that player begins another turn, not between interactions within the same turn. Gamma defaults to `0.99` and can be set for a new run with `--gamma`.

Training progress reports games, wins, selection counts, average selected rank,
average immediate reward, and average reward-to-go by assigned tier rather than
by seat.
