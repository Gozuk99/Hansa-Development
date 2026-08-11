# Training States

Curriculum training generates deterministic, playable positions covering all three maps, supported player counts, end conditions, strategic focuses, and optional modules. These are deliberately varied board positions, not reconstructed game histories.

Temporary learning positions are written beneath `training_output/curriculum/states/`. The runner removes old batches automatically so these files do not accumulate indefinitely.

The permanent evaluation suite is organized as:

```text
generated/evaluation/<map-player-scenario>/<scenario>/map_<number>/<player-count>_players/
```

Each position has an exact `.hansa` save and a searchable `.json` summary. The evaluation manifest records the fixed suite used to compare model versions consistently.

Generate the fixed evaluation suite in an empty evaluation directory with:

```powershell
python tools/generate_training_states.py --eval --seed 1
```

Run training with:

```powershell
python tools/run_curriculum_training.py --iterations 100 --batch 5
```

The same generation request and seed produce the same state identity. Every saved state is checked with the engine and save-file validators, loaded again, and revalidated before use.
