"""Command-line entry point for targeted near-end-game state generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.targeted_state_generator import (  # noqa: E402
    DEFAULT_OUTPUT_DIRECTORY,
    EndGameScenario,
    GenerationRequest,
    generate_state,
    save_generated_state,
)


def _optional_bool(value):
    return {"auto": None, "on": True, "off": False}[value]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--scenario",
        choices=["random", *(item.value for item in EndGameScenario)],
        default="random",
    )
    parser.add_argument(
        "--map", dest="map_num", choices=("random", "1", "2", "3"), default="random"
    )
    parser.add_argument("--players", choices=("random", "3", "4", "5"), default="random")
    for option in ("mission-cards", "emperors-favour", "promo-markers"):
        parser.add_argument(f"--{option}", choices=("auto", "on", "off"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    for offset in range(args.count):
        request = GenerationRequest(
            seed=args.seed + offset,
            scenario=None if args.scenario == "random" else EndGameScenario(args.scenario),
            map_num=None if args.map_num == "random" else int(args.map_num),
            player_count=None if args.players == "random" else int(args.players),
            use_mission_cards=_optional_bool(args.mission_cards),
            use_emperors_favour=_optional_bool(args.emperors_favour),
            use_promo_markers=_optional_bool(args.promo_markers),
        )
        generated = generate_state(request)
        save_path, _metadata_path = save_generated_state(generated, args.output)
        print(f"{generated.scenario.value}: {save_path}")


if __name__ == "__main__":
    main()
