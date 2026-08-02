"""Train the shared Hansa model from exact near-end-game saves."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.self_play import SelfPlayTrainer, TrainingConfig  # noqa: E402


DEFAULT_STATE = ROOT / "training_data/5p-map2_YELLOW_19_points_1_turn_from_winning.hansa"
DEFAULT_CHECKPOINT = ROOT / "hansa_nn_model.pth"


@dataclass(frozen=True)
class TrainingRunSummary:
    completed_games: int
    decisions: int
    acting_player_wins: int


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        action="append",
        dest="states",
        help="Exact .hansa starting state; repeat to train from several positions",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-move-action",
        action="store_true",
        default=None,
        help="Allow normal Move interactions (disabled by default)",
    )
    return parser.parse_args()


def train_with_periodic_checkpoints(
    trainer, states, *, episodes, batch_size, checkpoint_every, checkpoint_path
):
    """Train in resumable chunks and save after every completed chunk."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    completed_games = 0
    decisions = 0
    acting_player_wins = 0
    remaining = episodes
    while remaining:
        count = min(remaining, checkpoint_every)
        trajectories = trainer.train(states, count, batch_size=batch_size)
        completed_games += len(trajectories)
        decisions += sum(len(trajectory.decisions) for trajectory in trajectories)
        acting_player_wins += sum(
            bool(trajectory.decisions)
            and trajectory.decisions[0].acting_player_index in trajectory.winner_indices
            for trajectory in trajectories
        )
        trainer.save_checkpoint(checkpoint_path, states)
        remaining -= count
    return TrainingRunSummary(completed_games, decisions, acting_player_wins)


def main():
    args = parse_args()
    states = tuple(args.states or (DEFAULT_STATE,))
    missing = [str(path) for path in states if not path.is_file()]
    if missing:
        raise SystemExit("Starting state does not exist: " + ", ".join(missing))
    if args.episodes < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        raise SystemExit("episodes, batch size, and checkpoint interval must be positive")

    if args.resume:
        if not args.checkpoint.is_file():
            raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")
        overrides = {
            "--epsilon": args.epsilon,
            "--learning-rate": args.learning_rate,
            "--max-actions": args.max_actions,
            "--seed": args.seed,
            "--gamma": args.gamma,
            "--allow-move-action": args.allow_move_action,
        }
        supplied = [name for name, value in overrides.items() if value is not None]
        if supplied:
            raise SystemExit(
                "Resume uses the checkpoint's saved training configuration; remove: "
                + ", ".join(supplied)
            )
        trainer = SelfPlayTrainer.from_checkpoint(args.checkpoint)
    else:
        trainer = SelfPlayTrainer(
            config=TrainingConfig(
                epsilon=0.20 if args.epsilon is None else args.epsilon,
                learning_rate=(0.0001 if args.learning_rate is None else args.learning_rate),
                max_actions=500 if args.max_actions is None else args.max_actions,
                disable_move_action=not bool(args.allow_move_action),
                seed=124 if args.seed is None else args.seed,
                gamma=0.99 if args.gamma is None else args.gamma,
            )
        )

    summary = train_with_periodic_checkpoints(
        trainer,
        states,
        episodes=args.episodes,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=args.checkpoint,
    )

    print(f"Completed games: {summary.completed_games}")
    print(f"Training updates: {trainer.progress.training_updates}")
    print(f"Recorded decisions: {summary.decisions}")
    print(f"Acting-player wins: {summary.acting_player_wins}/{summary.completed_games}")
    print(f"Latest loss: {trainer.progress.last_loss:.3f}")
    print(f"Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
