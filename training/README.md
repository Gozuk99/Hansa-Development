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

Each learning and evaluation row records its raw measured loss in `latest_loss`.
The dashboard calculates rolling loss statistics from that history instead of
storing a derived value in every row.

Every training and evaluation row also records the raw paid-action movement
counters: Move and spent-action counts, pointless Move workflows, repeated-Move
and all-Move-turn penalty counts, route-creating Moves, and immediate
Move-to-Claim conversions. The dashboard derives ratios safely from those raw
counters. It aggregates the
fixed evaluation games by batch and applies its existing map and player filters.
Early-game evaluation is reported separately from the established fixed suite;
these values are diagnostics and do not add or change any reward.

By default it runs five learning games followed by the complete fixed evaluation
suite. Set the learning games per batch with `--iterations`; use `--batch` to
repeat that entire learning-and-evaluation cycle.
The model learns after every collected learning trajectory, including penalized
no-replacement-route failures. Training uses one 256-decision update per sampled
block: mid, late, and end games use at most four blocks (1,024 decisions),
while generated early games use at most 4,096 decisions. CSV coverage columns record the
trajectory and sampled decision counts. Early
samples are spread across eight chronological sections, targeting 512 decisions
from each section. Unused section capacity is redistributed without splitting movement
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

Training uses an exact shuffled twenty-game curriculum cycle: ten fresh (50%),
five early (25%), two mid (10%), two late (10%), and one end position (5%). Independently,
5% of training games disable only the broad epsilon-random branch for every
seat; each tier retains its configured Top-K and `1 / sqrt(rank)` weighting.
The zero-epsilon percentage is configurable from the curriculum runner CLI.
Fresh positions use canonical new-game setup, including seeded optional modules,
marker supplies, and randomized legal locations for the three fixed starter-marker
types. The CSV records each maturity so results can be compared separately.
Its `run_mode` identifies normal or zero-epsilon training and the active fixed
evaluation set without mutually exclusive columns.
Early positions retain nine to twelve bonus markers in supply while all three
route markers remain in play.
Fresh positions choose standard, full-promo, and random-mix draw supplies with
equal probability. Fresh Map 1 positions enable mission cards 50% of the time,
and every fresh position independently enables Emperor's Favour 50% of the time.
Developed positions retain their existing 50/25/25 marker-supply split and Map 1
40/60 mission-card split.
Early-game rows use the single `early` label and do not prepare a score,
bonus-marker, or completed-city ending condition. Midgame rows use `score_focus`,
`bonus_marker_focus`, or `completed_city_focus`; only late/end rows use `near_*`
condition labels. Generated mid/late/end learning positions prepare routes
exactly one post short: one of two,
two of three, or three of four posts. Automatic training focuses do not add an
opponent blocker to that final post, so the model must place the missing piece
before it can claim the route.
Evaluation suite version 9 preserves the 27 established fixed positions and the
27 deterministic early-game positions: one for every map, supported player
count, and bonus-marker setup. The early set has low scores, modest development, zero completed cities,
three active route markers, varied remaining marker supply, and deterministic
optional modules. It contains no East-West, regional-control, immediate-finish,
or fresh-game setup. CSV `run_mode` values identify `evaluation_mid_late_end`
versus `evaluation_early`, and the
dashboard gives early evaluation its own filtered section and timeout reporting.
Across all generated states, a positive starting score is
accepted only when that player controls a city or occupies a spent special
prestige/bonus-VP circle.
Every map/player-count combination deliberately covers default,
all-promotional, and mixed bonus-marker supplies. Map 1's nine early positions
include four with mission cards and five without.
End-relative labels such as `two_decisions_before`, `one_round_before`, and
`immediate_finish` are reserved for end-focused training or fixed evaluation
positions.

For example, `--batch 10 --iterations 100` runs ten batches. Each batch contains
100 learning games followed by the 54 fixed test-only games, for 1,000 learning
games and 540 test-only games total.

Generated early training games use a 15,000-interaction limit; mid, late, and
end training games remain at 10,000. Every fixed evaluation game, including the
early benchmark, remains capped at 10,000 so timeout rates stay comparable.
Action counts and completion reasons show whether an early game finished during
the extra 5,000 interactions or still timed out at 15,000. Promotion considers
invalid actions, unfinished games, fixed-state evaluation completion, Tier 1
performance, and rolling loss. Action-limit failures retain their accumulated
training signal; diagnostic bundles are preserved under
`training_data/failures/`.

Normal runs retain only high-level generation, play, and learning timers. Pass
`--detailed-profiling` to collect and write the nine fine-grained hot-loop timing
metrics for a diagnostic run; detailed profiling is disabled by default.

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

Three-player training games use tiers 1 and 2 plus one uniformly selected tier
from 3, 4, and 5. Four-player training games use tiers 1, 2, 4, and 5.
Five-player training games use all five tiers. The selected roster is reshuffled
across seats for every game, so no color, player number, or turn position
permanently receives a stronger or weaker policy. Fixed evaluation retains its
historical three-player 1/3/5 roster.

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
