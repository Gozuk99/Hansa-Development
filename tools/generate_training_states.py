"""Command-line entry point for targeted near-end-game state generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
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


def _route_readiness(value):
    return {"auto": None, "full": True, "one-short": False}[value]


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    map_num: int
    player_count: int
    scenario: EndGameScenario = EndGameScenario.NEAR_SCORE
    score_range: tuple[int, int] = (16, 16)
    mission_cards: bool = False
    emperors_favour: bool = False
    promo_markers: bool = False
    immediate_finish: bool = False
    east_west_path_length: str | None = None
    prepared_route_full: bool | None = None


EVALUATION_SPECS = tuple(
    EvaluationSpec(f"balanced_map{map_num}_{players}p", map_num, players)
    for map_num in (1, 2, 3)
    for players in (3, 4, 5)
) + (
    EvaluationSpec(
        "missions_map1_3p",
        1,
        3,
        score_range=(10, 17),
        mission_cards=True,
        immediate_finish=True,
    ),
    EvaluationSpec(
        "promo_markers_map2_4p",
        2,
        4,
        scenario=EndGameScenario.NEAR_BONUS_MARKERS,
        promo_markers=True,
        immediate_finish=True,
    ),
    EvaluationSpec(
        "emperors_favour_map3_5p",
        3,
        5,
        scenario=EndGameScenario.NEAR_COMPLETED_CITIES,
        emperors_favour=True,
        immediate_finish=True,
    ),
    EvaluationSpec(
        "all_options_map1_5p",
        1,
        5,
        score_range=(10, 17),
        mission_cards=True,
        emperors_favour=True,
        promo_markers=True,
    ),
    EvaluationSpec("city_limit_map2_5p", 2, 5, scenario=EndGameScenario.NEAR_COMPLETED_CITIES),
    EvaluationSpec(
        "marker_limit_map3_4p",
        3,
        4,
        scenario=EndGameScenario.NEAR_BONUS_MARKERS,
        emperors_favour=True,
        promo_markers=True,
    ),
    EvaluationSpec(
        "east_west_short_map1_4p",
        1,
        4,
        scenario=EndGameScenario.EAST_WEST,
        east_west_path_length="short",
        prepared_route_full=True,
    ),
    EvaluationSpec(
        "east_west_medium_map2_4p",
        2,
        4,
        scenario=EndGameScenario.EAST_WEST,
        east_west_path_length="medium",
        prepared_route_full=True,
    ),
    EvaluationSpec(
        "east_west_long_map3_5p",
        3,
        5,
        scenario=EndGameScenario.EAST_WEST,
        east_west_path_length="long",
        prepared_route_full=True,
    ),
    EvaluationSpec(
        "wales_control_map3_3p",
        3,
        3,
        scenario=EndGameScenario.BRITANNIA_WALES,
        prepared_route_full=True,
    ),
    EvaluationSpec(
        "scotland_control_map3_4p",
        3,
        4,
        scenario=EndGameScenario.BRITANNIA_SCOTLAND,
        prepared_route_full=True,
    ),
    EvaluationSpec(
        "isle_of_man_dual_control_map3_5p",
        3,
        5,
        scenario=EndGameScenario.BRITANNIA_ISLE_OF_MAN,
        prepared_route_full=True,
    ),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--eval",
        action="store_true",
        help=f"create the permanent {len(EVALUATION_SPECS)}-state evaluation suite",
    )
    parser.add_argument(
        "--scenario",
        choices=["random", *(item.value for item in EndGameScenario)],
        default="random",
    )
    parser.add_argument(
        "--immediate-finish",
        action="store_true",
        help="make the prepared player act first instead of after the other players",
    )
    parser.add_argument(
        "--east-west-path-length",
        choices=("short", "medium", "long"),
        help="choose an East-West path group when --scenario east_west is used",
    )
    parser.add_argument(
        "--prepared-route",
        choices=("auto", "full", "one-short"),
        default="auto",
        help="choose whether a targeted scoring route is full or one post short",
    )
    parser.add_argument(
        "--map", dest="map_num", choices=("random", "1", "2", "3"), default="random"
    )
    parser.add_argument("--players", choices=("random", "3", "4", "5"), default="random")
    for option in ("mission-cards", "emperors-favour", "promo-markers"):
        parser.add_argument(f"--{option}", choices=("auto", "on", "off"), default="auto")
    return parser.parse_args(argv)


def _generate_evaluation_suite(args):
    evaluation_directory = args.output / "evaluation"
    if evaluation_directory.exists() and any(evaluation_directory.iterdir()):
        raise SystemExit(
            f"Evaluation suite already exists at {evaluation_directory}; it was not overwritten"
        )

    manifest = []
    for index, spec in enumerate(EVALUATION_SPECS):
        generated = generate_state(
            GenerationRequest(
                seed=args.seed + index,
                scenario=spec.scenario,
                map_num=spec.map_num,
                player_count=spec.player_count,
                use_mission_cards=spec.mission_cards,
                use_emperors_favour=spec.emperors_favour,
                use_promo_markers=spec.promo_markers,
                score_range=spec.score_range,
                immediate_finish=spec.immediate_finish,
                east_west_path_length=spec.east_west_path_length,
                prepared_route_full=spec.prepared_route_full,
            )
        )
        save_path, metadata_path = save_generated_state(generated, evaluation_directory / spec.name)
        manifest.append(
            {
                **asdict(spec),
                "scenario": spec.scenario.value,
                "seed": args.seed + index,
                "save_file": str(save_path.relative_to(evaluation_directory)),
                "metadata_file": str(metadata_path.relative_to(evaluation_directory)),
            }
        )
        print(f"{index + 1}/{len(EVALUATION_SPECS)} {spec.name}: {save_path}")

    evaluation_directory.mkdir(parents=True, exist_ok=True)
    (evaluation_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {len(EVALUATION_SPECS)} fixed evaluation states in {evaluation_directory}")


def main(argv=None):
    args = parse_args(argv)
    if args.eval:
        _generate_evaluation_suite(args)
        return
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
            immediate_finish=args.immediate_finish,
            east_west_path_length=args.east_west_path_length,
            prepared_route_full=_route_readiness(args.prepared_route),
        )
        generated = generate_state(request)
        save_path, _metadata_path = save_generated_state(generated, args.output)
        print(f"{generated.scenario.value}: {save_path}")


if __name__ == "__main__":
    main()
