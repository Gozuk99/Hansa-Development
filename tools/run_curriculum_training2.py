"""Run the isolated experimental balanced curriculum trainer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.ai_model import HansaNN  # noqa: E402
from training.balanced_curriculum import BalancedCurriculumRunner  # noqa: E402
from training.curriculum import CurriculumConfig, PromotionCriteria  # noqa: E402
from training.self_play import SelfPlayTrainer, TrainingConfig  # noqa: E402


DEFAULT_DIRECTORY = ROOT / "training_output/curriculum2"
DEFAULT_CHECKPOINT = DEFAULT_DIRECTORY / "training_checkpoint.pth"
DEFAULT_PLAYABLE_MODEL = DEFAULT_DIRECTORY / "hansa_nn_model.pth"
DEFAULT_STARTING_MODEL = ROOT / "hansa_nn_model.pth"
DEFAULT_CSV = DEFAULT_DIRECTORY / "results.csv"
DEFAULT_EVALUATION_SUITE = ROOT / "training_data/generated/evaluation"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--retry-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--playable-model", type=Path, default=DEFAULT_PLAYABLE_MODEL)
    parser.add_argument("--starting-model", type=Path, default=DEFAULT_STARTING_MODEL)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--evaluation-suite", type=Path, default=DEFAULT_EVALUATION_SUITE)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.fresh:
        args.checkpoint.unlink(missing_ok=True)
        args.csv.unlink(missing_ok=True)
        trainer = SelfPlayTrainer(model=HansaNN(), config=TrainingConfig(seed=args.seed))
    elif args.checkpoint.exists():
        trainer = SelfPlayTrainer.from_checkpoint(args.checkpoint)
    else:
        trainer = SelfPlayTrainer(
            model=HansaNN(model_file=args.starting_model),
            config=TrainingConfig(seed=args.seed),
        )
    config = CurriculumConfig(
        iterations=args.batch,
        training_games_per_batch=args.iterations,
        evaluation_games_per_batch=1,
        update_batch_size=args.batch_size,
        retry_limit=args.retry_limit,
        seed=args.seed,
        promotion=PromotionCriteria(),
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
    runner.run()
    print(f"Balanced training complete. Model: {args.playable_model}. Results: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
