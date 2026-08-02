# Generated Training States

`tools/generate_training_states.py` creates deterministic, playable positions near one of Hansa Teutonica's three end conditions. These are deliberately varied board positions, not reconstructed game histories.

Generated files are organized as:

```text
generated/<scenario>/map_<number>/<player-count>_players/
```

Each position has an exact `.hansa` save and a searchable `.json` summary. Generated files are ignored by Git because datasets can become large; this README remains tracked.

Example:

```powershell
python tools/generate_training_states.py --count 30 --seed 1000
```

The same request and seed produce the same state identity. Every state is checked with the normal engine and save-file validators, saved, loaded again, and revalidated before the command reports success.
