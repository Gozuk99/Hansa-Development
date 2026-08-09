"""Run local curriculum generation, self-play training, evaluation, and logging."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.ai_model import HansaNN  # noqa: E402
from training.curriculum import (  # noqa: E402
    CurriculumConfig,
    CurriculumRunner,
    PromotionCriteria,
)
from training.self_play import SelfPlayTrainer, TrainingConfig  # noqa: E402


DEFAULT_DIRECTORY = ROOT / "training_output/curriculum"
DEFAULT_CHECKPOINT = DEFAULT_DIRECTORY / "training_checkpoint.pth"
DEFAULT_PLAYABLE_MODEL = ROOT / "hansa_nn_model.pth"
DEFAULT_CSV = DEFAULT_DIRECTORY / "results.csv"
DEFAULT_EVALUATION_SUITE = ROOT / "training_data/generated/evaluation"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of learning games to run before one test-only game",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Number of learning-and-evaluation batches to run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Completed learning games between model/checkpoint/CSV saves",
    )
    parser.add_argument("--retry-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--playable-model", type=Path, default=DEFAULT_PLAYABLE_MODEL)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--evaluation-suite", type=Path, default=DEFAULT_EVALUATION_SUITE)
    parser.add_argument("--maximum-unfinished-rate", type=float, default=0.05)
    parser.add_argument("--minimum-evaluation-completion", type=float, default=0.95)
    parser.add_argument("--loss-tolerance", type=float, default=0.10)
    parser.add_argument("--rolling-loss-window", type=int, default=5)
    parser.add_argument("--skip-tier-one-promotion-check", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start with new model weights and replace prior recovery/CSV progress",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.fresh:
        args.checkpoint.unlink(missing_ok=True)
        args.csv.unlink(missing_ok=True)
        trainer = SelfPlayTrainer(
            model=HansaNN(),
            config=TrainingConfig(seed=args.seed),
        )
    elif args.checkpoint.exists():
        trainer = SelfPlayTrainer.from_checkpoint(args.checkpoint)
    else:
        trainer = SelfPlayTrainer(
            model=HansaNN(model_file=args.playable_model),
            config=TrainingConfig(seed=args.seed),
        )

    config = CurriculumConfig(
        iterations=args.batch,
        training_games_per_batch=args.iterations,
        evaluation_games_per_batch=1,
        update_batch_size=args.batch_size,
        retry_limit=args.retry_limit,
        seed=args.seed,
        promotion=PromotionCriteria(
            maximum_unfinished_rate=args.maximum_unfinished_rate,
            minimum_evaluation_completion_rate=args.minimum_evaluation_completion,
            require_tier_one_advantage=not args.skip_tier_one_promotion_check,
            loss_tolerance=args.loss_tolerance,
            rolling_loss_window=args.rolling_loss_window,
        ),
    )
    runner = CurriculumRunner(
        trainer,
        config,
        checkpoint_path=args.checkpoint,
        playable_model_path=args.playable_model,
        csv_path=args.csv,
        evaluation_suite_directory=args.evaluation_suite,
        progress_callback=print,
    )
    state = runner.run()
    print(
        f"Training complete: {args.batch} batch(es), "
        f"{args.batch * args.iterations} learning game(s), and "
        f"evaluation suite after each batch.\n"
        f"Current stage: {config.stages[state['stage_index']].name}.\n"
        f"Latest loss: {trainer.progress.last_loss}.\n"
        f"Replacement-route deadlocks: {trainer.progress.replacement_route_deadlocks}.\n"
        f"Playable model: {args.playable_model}.\n"
        f"Results: {args.csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
