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

Every training and evaluation row also records paid-action movement diagnostics:
Move count and ratio, pointless Move workflows, repeated-Move and all-Move-turn
penalty counts, route-creating Moves, and immediate Move-to-Claim conversions.
Rates remain blank when their denominator is zero. The dashboard aggregates the
fixed evaluation games by batch and applies its existing map and player filters.
Early-game evaluation is reported separately from the established fixed suite;
these values are diagnostics and do not add or change any reward.

By default it runs five learning games followed by the complete fixed evaluation
suite. Set the learning games per batch with `--iterations`; use `--batch` to
repeat that entire learning-and-evaluation cycle.
The model learns after every collected learning trajectory, including penalized
no-replacement-route failures. Training uses one 256-decision update per sampled
block: mid, late, end, and mixed games use at most four blocks (1,024 decisions),
while generated early and early-mixed games use at most 4,096 decisions. CSV coverage columns record the
trajectory decision count, sampled decision count, and sampled fraction. Early
samples are spread across eight chronological sections, targeting 512 decisions
from each section; eight additional CSV columns report the resulting per-section
counts. Unused section capacity is redistributed without splitting movement
workflows or exceeding 4,096 decisions. The final decision and the strongest
immediate rewards or workflow-local penalties are retained across those samples.
Normal
Move and permanent Move Any 2
pickups and placements are sampled together. An exact no-change workflow and the
equivalent rearrangement within one non-maritime route receive a local negative
target, as does the third consecutive normal Move. A turn that spends at least
two paid actions entirely on normal Move gives each of those Move workflows a
local negative target. The small same-route concentration reward can be earned
only once per player and route until that route is claimed. Paid normal-Move
workflows do not inherit terminal credit by default. When the next non-Move
paid action claims a route completed by an uninterrupted same-turn Move chain,
every Move in that chain that contributed to the claimed route regains terminal
credit. Immediate movement rewards still apply; small normal-Move efficiency
penalties adjust only their offending grouped workflow target. Permanent
Move Any 2 is unaffected.
`--batch-size` controls how many collected
learning trajectories pass between disk saves. After each save group (five
learning games by default), the runner saves its
recovery data, exports the playable model, and adds those games to the CSV. It
saves the final recovery information and evaluation rows after the suite, but
does not rewrite the unchanged playable model.
The command automatically resumes interrupted training when recovery information
exists; otherwise it starts from the current playable model.
The command has no option that deletes or resets the checkpoint, playable model,
or CSV history.

Training uses an exact shuffled 160-game curriculum cycle: 64 mixed (40%), 48
early-mixed (30%), 24 early (15%), nine mid, nine late, and six end positions.
Mid, late, and end therefore total 15% while retaining their existing 3:3:2
relative weighting. Mixed positions keep their established broad asymmetric
development. Early-mixed positions use shuffled 2-6 point and 3-5 development
roles, then place three or four conserved pieces per player across two or three
random non-Britannia routes. Every selected route retains an open post; route
selection may create contests but never creates a completed route or guaranteed
Claim. Development roles are shuffled independently of policy tiers.
The fresh-game profile remains commented out.
The CSV records the maturity in the curriculum-stage label so results can be
compared separately.
Early positions retain nine to twelve bonus markers in supply while all three
route markers remain in play.
Early training uses two reproducible variants: 70% scaffold exactly two unique,
randomly selected completed routes per player, while 30% keeps the original
sparse early board. Scaffold selection is without replacement and does not
prefer bonus-marker, upgrade, scoring, or otherwise valuable routes. All pieces
come from the normal conserved player pools. CSV rows record the variant plus
the selected route IDs and route lengths by seat. Fixed early evaluation boards
remain unscaffolded.
Across generated training positions, bonus-marker modules use the default
supply 50% of the time, every promotional marker plus shuffled defaults 25% of
the time, and a random mixture of default and promotional markers 25% of the
time. Map 1 enables mission cards 40% of the time and leaves them disabled 60%
of the time.
Early-game rows use the single `early` label and early-mixed rows use
`early_mixed`; neither prepares a score, bonus-marker, or completed-city ending
condition. Existing mixed rows use `mixed`. Midgame rows use `score_focus`,
`bonus_marker_focus`, or `completed_city_focus`; only late/end rows use `near_*`
condition labels. Generated mid/late/end learning positions prepare routes
exactly one post short: one of two,
two of three, or three of four posts. Automatic training focuses do not add an
opponent blocker to that final post, so the model must place the missing piece
before it can claim the route.
Evaluation suite version 8 preserves the 27 established fixed positions and the
27 deterministic early-game positions: one for every map, supported player
count, and bonus-marker setup. The early set has low scores, modest development, zero completed cities,
three active route markers, varied remaining marker supply, and deterministic
optional modules. It contains no East-West, regional-control, immediate-finish,
or fresh-game setup. CSV rows identify `mid_late_end` versus `early`, and the
dashboard gives early evaluation its own filtered section and timeout reporting.
Nine additional fixed mixed-development positions cover every map and player
count. Their dashboard section compares policy tiers by shuffled starting role,
including win, completion, timeout, final-score, score-gain, interaction, and
movement results. Across all generated states, a positive starting score is
accepted only when that player controls a city or occupies a spent special
prestige/bonus-VP circle.
Every map/player-count combination deliberately covers default,
all-promotional, and mixed bonus-marker supplies. Map 1's nine early positions
include four with mission cards and five without.
The CSV `starting_position` column remains blank for early, early-mixed, mixed,
mid, and late rows.
End-relative labels such as `two_decisions_before`, `one_round_before`, and
`immediate_finish` are reserved for end-focused training or fixed evaluation
positions.

For example, `--batch 10 --iterations 100` runs ten batches. Each batch contains
100 learning games followed by the 63 fixed test-only games, for 1,000 learning
games and 630 test-only games total.

Generated early and early-mixed training games use a 15,000-interaction limit;
mixed, mid, late, and end training games remain at 10,000. Every fixed evaluation game, including the
early benchmark, remains capped at 10,000 so timeout rates stay comparable.
Early training rows identify games that finished by 10,000, finished during the
extra 5,000 interactions, or still timed out at 15,000. Promotion considers
invalid actions, unfinished games, fixed-state evaluation completion, Tier 1
performance, and rolling loss. Action-limit failures retain their accumulated
training signal; diagnostic bundles are preserved under
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

An epsilon choice is uniform across every legal interaction. Otherwise, the tier's
Top-K semantic choices are sampled by model rank with normalized `1 / sqrt(rank)`
weights. For normal piece
placement and Move placement, equivalent empty posts on one non-maritime route are
treated as one choice for both ranked and random selection. After that route and
piece shape are chosen, one of its equivalent posts is selected. Training applies
the result to the whole equivalent group. Matching pieces on one non-maritime
route are likewise one pickup or displacement-target choice. Different owners,
piece shapes, displacement replacement shapes, maritime posts, and finish
controls remain distinct. The engine's legal-action mask and 768 action numbers
are unchanged. Move 3 uses the same grouping for matching opponent pieces and
for equivalent destination posts.
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
