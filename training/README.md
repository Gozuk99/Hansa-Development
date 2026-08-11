# Curriculum Self-Play Training

Curriculum training uses checkpoint format version 5. Earlier training checkpoints
remain usable for game inference when their action and observation schemas match,
but they cannot be resumed because they do not contain the complete tier, reward,
and curriculum configuration.

Run one complete local curriculum iteration with:

```powershell
python tools/run_curriculum_training.py
```

The command generates temporary states, collects tiered training games, updates
the shared model between collected learning trajectories, evaluates without updates on newly
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
rolling average of the latest five learning-game losses. Evaluation rows record
their measured loss in `latest_loss` but leave `rolling_mean_loss` blank because
evaluation never updates the model.

By default it runs five learning games followed by the complete fixed evaluation
suite. Set the learning games per batch with `--iterations`; use `--batch` to
repeat that entire learning-and-evaluation cycle.
The model learns after every collected learning trajectory, including penalized
no-replacement-route failures. Each trajectory receives one equally weighted update
using at most 256 representative decisions; the final decision and the strongest
immediate rewards are always retained. `--batch-size` controls how many collected
learning trajectories pass between disk saves. After each save group (five
learning games by default), the runner saves its
recovery data, exports the playable model, and adds those games to the CSV. It
saves the final recovery information and evaluation rows after the suite, but
does not rewrite the unchanged playable model.
The command automatically resumes interrupted training when recovery information
exists; otherwise it starts from the current playable model.

Use `--fresh` to begin a new experiment with untrained model weights. This removes
the previous curriculum recovery file and CSV results. The root playable model is
replaced only after the first successful save group:

```powershell
python tools/run_curriculum_training.py --fresh --iterations 100 --batch 5
```

Most generated positions begin approximately one full round before the prepared
ending player acts. Five percent begin roughly two decisions before the ending.
The CSV records the starting distance so the results can be compared separately.
The fixed evaluation suite contains one position for every combination of three
maps, three supported player counts, and three end conditions: 27 positions total.

For example, `--batch 10 --iterations 100` runs ten batches. Each batch contains
100 learning games followed by the 27 fixed test-only games, for 1,000 learning
games and 270 test-only games total.

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
