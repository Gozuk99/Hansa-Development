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

Pass `--state` more than once to rotate through several starting positions:

```powershell
python tools/train_self_play.py --state training_data/first.hansa --state training_data/second.hansa
```

## What "disable Move" means

By default, the training policy removes interactions that would begin the normal Move action by selecting one of the acting player's occupied posts. It does not change the engine's legal-action rules. Mandatory placement steps are never removed, and placing from personal supply, Income, displacement, route completion, bonus markers, and end-turn controls remain available.

For the supplied example, all 34 post interactions would begin Move. Removing them leaves exactly one choice: complete the prepared route and end the game. Use `--allow-move-action` to train without this temporary restriction.

## Current learning signal

Each interaction records `100 ×` the acting player's change in authoritative projected final prestige. After the game, winners receive another `100 ×` their final score, and a player who both triggers game end and wins receives `+150`.

Training calculates discounted reward-to-go separately for each player instead of assigning one whole-game total to every decision. Gamma defaults to `0.99` and can be set for a new run with `--gamma`.
