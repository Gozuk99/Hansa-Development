# Curriculum Self-Play Training

Curriculum training uses checkpoint format version 5. Earlier training checkpoints
remain usable for game inference when their action and observation schemas match,
but they cannot be resumed because they do not contain the complete tier, reward,
and curriculum configuration.

Observation-version-1 model and format-5 training checkpoints have an explicit
transfer into observation version 2 because the tensor shape is unchanged. This
preserves learned weights and optimizer progress while hiding opponents'
face-down used-marker identities. The next save records version 2; unrelated
schema mismatches remain rejected.

Run one complete local curriculum iteration with:

```powershell
python tools/run_curriculum_training.py
```

The command generates temporary states, collects tiered training games, updates
the shared model between collected learning trajectories, evaluates without updates on a fixed
suite of positions, saves resumable training state to
`training_output/curriculum/training_checkpoint.pth`, exports the successfully
trained playable model to `hansa_nn_model.pth`, and appends graphable results to
`training_output/curriculum/results.csv`. It does not call Codex, OpenAI, or any
other external AI service.

Progress is printed as each training and evaluation game finishes, followed by
the latest loss and the locations of the playable model and results file.
Evaluation uses the committed fixed suite so results remain comparable between
batches. It uses the same top-2, top-5, top-10, top-15, and
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
no-replacement-route failures. Training uses one 256-decision update per started
block of 256 trajectory decisions, up to four non-overlapping updates and 1,024
sampled decisions per game. The final decision and the strongest immediate rewards
or penalties are retained across those samples, giving long
looping games additional teaching weight. Normal Move and permanent Move Any 2
pickups and placements are sampled together, so selecting a movement penalty also
retains the choices that produced it. `--batch-size` controls how many collected
learning trajectories pass between disk saves. After each save group (five
learning games by default), the runner saves its
recovery data, exports the playable model, and adds those games to the CSV. It
saves the final recovery information and evaluation rows after the suite, but
does not rewrite the unchanged playable model.
The command automatically resumes interrupted training when recovery information
exists; otherwise it starts from the current playable model.
The command has no option that deletes or resets the checkpoint, playable model,
or CSV history.

Training currently uses a shuffled five-game late/end cycle: four late-game
positions and one end-game position. The fresh, early, and mid-game profiles
remain in the curriculum source as commented configuration so they can be
restored after the model has relearned decisive late-game play. The CSV records
the maturity in the curriculum-stage label so results can be compared separately.
Evaluation suite version 3 contains one position for every combination of three
maps, three supported player counts, and three end conditions: 27 positions total.
Three positions test an immediate ending, one per end condition. The other 24
begin farther from the end with prepared routes one post short.

For example, `--batch 10 --iterations 100` runs ten batches. Each batch contains
100 learning games followed by the 27 fixed test-only games, for 1,000 learning
games and 270 test-only games total.

The active late/end curriculum uses a 10,000-interaction limit at every
maturity. Promotion considers invalid actions, unfinished games, fixed-state
evaluation completion, Tier 1 performance, and rolling loss. Action-limit
failures retain their accumulated training signal; diagnostic bundles are
preserved under `training_data/failures/`.

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

An epsilon choice is uniform across every legal interaction. Otherwise, selection
is uniform within the tier's highest-ranked legal interactions. During normal Move
exploration, pickup and placement clicks receive equal weight when both are
available. A pickup selects one concrete owned post. A placement selects one
non-maritime route and piece shape, then one equivalent empty post on that route.
Ranked choices still compare and select specific posts. Different piece shapes,
maritime posts, occupied posts, and finish controls remain distinct.
Tier 5 explores more broadly than the stronger tiers without choosing uniformly
from every legal interaction.
After a staged workflow begins, follow-up interactions use bounded exploration
instead of the normal tier epsilon. With two legal choices, the best and second
choices receive 60% and 40%. With three or more, selection is 40% best, 20%
second, 15% third, and 25% uniformly random among legal choices. A forced single
choice is selected directly. This applies to Move, displacement, bonus-marker
resolution, route completion, and other multi-interaction workflows.
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
