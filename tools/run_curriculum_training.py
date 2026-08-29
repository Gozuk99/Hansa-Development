"""Run local curriculum generation, self-play training, evaluation, and logging."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.ai_model import HansaNN  # noqa: E402
from training.balanced_curriculum import BalancedCurriculumRunner  # noqa: E402
from training.curriculum import (  # noqa: E402
    CurriculumConfig,
    DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS,
    PromotionCriteria,
)
from training.self_play import SelfPlayTrainer, TrainingConfig  # noqa: E402


DEFAULT_DIRECTORY = ROOT / "training_output/curriculum"
DEFAULT_CHECKPOINT = DEFAULT_DIRECTORY / "training_checkpoint.pth"
DEFAULT_PLAYABLE_MODEL = ROOT / "hansa_nn_model.pth"
DEFAULT_CSV = DEFAULT_DIRECTORY / "results.csv"
DEFAULT_EVALUATION_SUITE = ROOT / "training_data/generated/evaluation"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Learning games per batch before the complete evaluation suite",
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
    parser.add_argument(
        "--zero-epsilon-training-percentage",
        type=float,
        default=None,
        help="Legacy override applying one zero-epsilon percentage to every maturity",
    )
    for maturity, fraction in DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS:
        parser.add_argument(
            f"--{maturity}-zero-epsilon-training-percentage",
            type=float,
            default=fraction * 100,
            help=f"Zero-epsilon percentage for {maturity} training games",
        )
    parser.add_argument(
        "--detailed-profiling",
        action="store_true",
        help="Collect fine-grained action-loop timings (disabled by default)",
    )
    parser.add_argument(
        "--shadow-filter-audit",
        action="store_true",
        help="Collect the expensive per-action shadow-filter audit (disabled by default)",
    )
    parser.add_argument("--skip-tier-one-promotion-check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stale_states = DEFAULT_DIRECTORY / "states"
    if stale_states.is_dir():
        shutil.rmtree(stale_states)
    if args.checkpoint.exists():
        trainer = SelfPlayTrainer.from_checkpoint(args.checkpoint)
        trainer.config = replace(
            trainer.config,
            detailed_profiling=args.detailed_profiling,
            shadow_filter_audit_enabled=args.shadow_filter_audit,
        )
    else:
        trainer = SelfPlayTrainer(
            model=HansaNN(model_file=args.playable_model),
            config=TrainingConfig(
                seed=args.seed,
                detailed_profiling=args.detailed_profiling,
                shadow_filter_audit_enabled=args.shadow_filter_audit,
            ),
        )

    zero_epsilon_fractions = tuple(
        (
            maturity,
            (
                args.zero_epsilon_training_percentage
                if args.zero_epsilon_training_percentage is not None
                else getattr(args, f"{maturity}_zero_epsilon_training_percentage")
            )
            / 100,
        )
        for maturity, _fraction in DEFAULT_ZERO_EPSILON_TRAINING_FRACTIONS
    )
    config = CurriculumConfig(
        iterations=args.batch,
        training_games_per_batch=args.iterations,
        evaluation_games_per_batch=1,
        update_batch_size=args.batch_size,
        retry_limit=args.retry_limit,
        seed=args.seed,
        zero_epsilon_training_fractions=zero_epsilon_fractions,
        promotion=PromotionCriteria(
            maximum_unfinished_rate=args.maximum_unfinished_rate,
            minimum_evaluation_completion_rate=args.minimum_evaluation_completion,
            require_tier_one_advantage=not args.skip_tier_one_promotion_check,
            loss_tolerance=args.loss_tolerance,
            rolling_loss_window=args.rolling_loss_window,
        ),
    )
    runner = BalancedCurriculumRunner(
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
        "Training mix: 50% fresh, 0% early, 15% mid, 20% late, 15% end.\n"
        "Zero-epsilon by maturity: "
        + ", ".join(
            f"{maturity} {fraction * 100:g}%" for maturity, fraction in zero_epsilon_fractions
        )
        + ".\n"
        f"Latest loss: {trainer.progress.last_loss}.\n"
        f"Replacement-route deadlocks: {trainer.progress.replacement_route_deadlocks}.\n"
        f"Playable model: {args.playable_model}.\n"
        f"Results: {args.csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
