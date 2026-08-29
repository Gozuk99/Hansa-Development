"""Command-line entry point for targeted near-end-game state generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.balanced_state_generator import (  # noqa: E402
    BalancedGenerationRequest,
    BonusMarkerSetup,
    EndingCondition,
    RegionalFocus,
    StartingPosition,
    StrategicFocus,
    bonus_marker_configuration,
    generate_balanced_state,
    save_balanced_state,
)
from game.game_config import GameConfiguration, human_players  # noqa: E402
from game.persistence import save_game  # noqa: E402
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
    ending_condition: EndingCondition | None
    score_range: tuple[int, int] = (16, 17)
    east_west: bool = False
    regional_focus: RegionalFocus | None = None
    mission_cards: bool = False
    emperors_favour: bool = False
    promo_markers: bool = False
    bonus_marker_setup: BonusMarkerSetup | None = None
    immediate_finish: bool = False
    bonus_markers_remaining: int = 2
    completed_cities_below_limit: int = 2
    prepared_routes_one_short: bool = True
    development_range: tuple[int, int] | None = None
    evaluation_set: str = "mid_late_end"
    prepare_ending_condition: bool = True
    round_range: tuple[int, int] = (8, 20)


EVALUATION_SUITE_VERSION = 10
RETIRED_EARLY_EVALUATION_SEED_COUNT = 27


def _regional_focus(player_count, ending_index):
    if player_count == 3:
        return RegionalFocus.WALES if ending_index == 1 else None
    focuses = (
        (RegionalFocus.WALES, RegionalFocus.ISLE_OF_MAN, RegionalFocus.SCOTLAND)
        if player_count == 4
        else (RegionalFocus.WALES, RegionalFocus.SCOTLAND, RegionalFocus.ISLE_OF_MAN)
    )
    return focuses[ending_index]


def _evaluation_specs():
    specs = []
    for map_num in (1, 2, 3):
        for players in (3, 4, 5):
            for ending_index, ending in enumerate(EndingCondition):
                immediate_finish = (map_num, players, ending) in {
                    (1, 3, EndingCondition.NEAR_SCORE),
                    (2, 4, EndingCondition.NEAR_BONUS_MARKERS),
                    (3, 5, EndingCondition.NEAR_COMPLETED_CITIES),
                }
                if (map_num, players) == (3, 4):
                    east_west_index = 0
                elif (map_num, players) == (3, 5):
                    east_west_index = 1
                else:
                    east_west_index = (map_num + players) % 3
                east_west = ending_index == east_west_index
                region = _regional_focus(players, ending_index) if map_num == 3 else None
                specs.append(
                    EvaluationSpec(
                        f"map{map_num}_{players}p_{ending.value}",
                        map_num,
                        players,
                        ending,
                        east_west=east_west,
                        regional_focus=region,
                        mission_cards=map_num == 1 and (players + ending_index) % 2 == 0,
                        emperors_favour=(map_num + players + ending_index) % 2 == 0,
                        promo_markers=(map_num + players + ending_index) % 2 == 1,
                        immediate_finish=immediate_finish,
                        score_range=(17, 18) if immediate_finish else (15, 16),
                        bonus_markers_remaining=0 if immediate_finish else 2,
                        completed_cities_below_limit=1 if immediate_finish else 2,
                        prepared_routes_one_short=not immediate_finish,
                    )
                )
    # Fresh evaluation is an untouched initial game position.
    for map_num in (1, 2, 3):
        for players in (3, 4, 5):
            for setup_index, marker_setup in enumerate(BonusMarkerSetup):
                configuration_index = (map_num - 1) * 9 + (players - 3) * 3 + setup_index
                specs.append(
                    EvaluationSpec(
                        f"map{map_num}_{players}p_fresh_{marker_setup.value}",
                        map_num,
                        players,
                        None,
                        score_range=(0, 0),
                        mission_cards=(map_num == 1 and configuration_index in {1, 3, 6, 8}),
                        emperors_favour=configuration_index % 2 == 0,
                        promo_markers=marker_setup is not BonusMarkerSetup.DEFAULT,
                        bonus_marker_setup=marker_setup,
                        prepared_routes_one_short=False,
                        development_range=(0, 0),
                        evaluation_set="fresh",
                        prepare_ending_condition=False,
                        round_range=(1, 1),
                    )
                )
    return tuple(specs)


EVALUATION_SPECS = _evaluation_specs()


def _evaluation_seed(base_seed, spec_index, spec):
    """Preserve the existing fixed Fresh boards after retiring Early evaluation."""
    retired_offset = RETIRED_EARLY_EVALUATION_SEED_COUNT if spec.evaluation_set == "fresh" else 0
    return base_seed + spec_index + retired_offset


def evaluation_request(spec, seed):
    if spec.ending_condition is None:
        raise ValueError("Fresh evaluation uses fresh_evaluation_game(), not balanced generation")
    return BalancedGenerationRequest(
        seed=seed,
        map_num=spec.map_num,
        player_count=spec.player_count,
        ending_condition=spec.ending_condition,
        score_range=spec.score_range,
        strategic_focus=(StrategicFocus.EAST_WEST if spec.east_west else StrategicFocus.NONE),
        regional_focus=spec.regional_focus,
        use_mission_cards=spec.mission_cards,
        use_emperors_favour=spec.emperors_favour,
        use_promo_markers=spec.promo_markers,
        bonus_marker_setup=spec.bonus_marker_setup,
        bonus_markers_remaining=spec.bonus_markers_remaining,
        completed_cities_below_limit=spec.completed_cities_below_limit,
        prepared_routes_one_short=spec.prepared_routes_one_short,
        development_range=spec.development_range,
        prepare_ending_condition=spec.prepare_ending_condition,
        round_range=spec.round_range,
        starting_position=(
            StartingPosition.IMMEDIATE_FINISH
            if spec.immediate_finish
            else StartingPosition.ONE_ROUND_BEFORE
        ),
    )


def fresh_evaluation_game(spec, seed):
    """Create one deterministic, untouched initial evaluation game."""
    if spec.evaluation_set != "fresh" or spec.bonus_marker_setup is None:
        raise ValueError("A Fresh evaluation spec with a marker setup is required")
    use_promos, promo_mode, promo_markers = bonus_marker_configuration(
        spec.bonus_marker_setup,
        seed,
    )
    return GameConfiguration(
        map_num=spec.map_num,
        player_count=spec.player_count,
        player_controls=human_players(spec.player_count),
        use_mission_cards=spec.mission_cards,
        use_emperors_favour=spec.emperors_favour,
        use_promo_markers=use_promos,
        promo_marker_mode=promo_mode,
        promo_markers=promo_markers,
        randomize_starting_bonus_marker_locations=True,
        seed=seed,
    ).create_game()


def _starting_bonus_marker_routes(game):
    return tuple(
        (route.bonus_marker.type, *(city.name for city in route.cities))
        for route in game.selected_map.routes
        if route.bonus_marker is not None
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
        "--fresh-eval",
        action="store_true",
        help="append the inactive Fresh fixed-evaluation states without replacing the suite",
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
        seed = _evaluation_seed(args.seed, index, spec)
        if spec.evaluation_set == "fresh":
            game = fresh_evaluation_game(spec, seed)
            save_path, metadata_path = _save_fresh_evaluation_state(
                game,
                spec,
                seed,
                evaluation_directory,
            )
            manifest.append(
                _fresh_manifest_entry(
                    game,
                    spec,
                    seed,
                    save_path,
                    metadata_path,
                    evaluation_directory,
                )
            )
            print(f"{index + 1}/{len(EVALUATION_SPECS)} {spec.name}: {save_path}")
            continue
        generated = generate_balanced_state(evaluation_request(spec, seed), max_attempts=10_000)
        save_path, metadata_path = save_balanced_state(
            generated,
            evaluation_directory / spec.name,
        )
        focuses = ["east_west"] if spec.east_west else []
        if spec.regional_focus is not None:
            focuses.append(spec.regional_focus.value)
        manifest.append(
            {
                **asdict(spec),
                "suite_version": EVALUATION_SUITE_VERSION,
                "scenario": "+".join((spec.ending_condition.value, *focuses)),
                "ending_condition": spec.ending_condition.value,
                "regional_focus": (
                    None if spec.regional_focus is None else spec.regional_focus.value
                ),
                "starting_score_by_seat": generated.starting_scores_by_seat,
                "starting_development_by_seat": generated.starting_development_by_seat,
                "seed": seed,
                "save_file": save_path.relative_to(evaluation_directory).as_posix(),
                "metadata_file": metadata_path.relative_to(evaluation_directory).as_posix(),
            }
        )
        print(f"{index + 1}/{len(EVALUATION_SPECS)} {spec.name}: {save_path}")

    evaluation_directory.mkdir(parents=True, exist_ok=True)
    (evaluation_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {len(EVALUATION_SPECS)} fixed evaluation states in {evaluation_directory}")


def _save_fresh_evaluation_state(game, spec, seed, evaluation_directory):
    identity = {
        "suite_version": EVALUATION_SUITE_VERSION,
        "evaluation_set": "fresh",
        "seed": seed,
        "map_num": spec.map_num,
        "player_count": spec.player_count,
        "mission_cards": spec.mission_cards,
        "emperors_favour": spec.emperors_favour,
        "bonus_marker_setup": spec.bonus_marker_setup.value,
    }
    state_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = (
        evaluation_directory / "fresh_game" / f"map_{spec.map_num}" / f"{spec.player_count}_players"
    )
    save_path = save_game(game, directory / f"state-{state_id}.hansa")
    metadata_path = directory / f"state-{state_id}.json"
    metadata_path.write_text(
        json.dumps(
            {
                **identity,
                "state_id": f"state-{state_id}",
                "starting_score_by_seat": [player.score for player in game.players],
                "starting_development_by_seat": [0 for _player in game.players],
                "emperor_tiles": list(game.tile_pool),
                "bonus_marker_draw_supply": list(game.selected_map.bonus_marker_pool),
                "starting_bonus_marker_routes": _starting_bonus_marker_routes(game),
                "save_file": save_path.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return save_path, metadata_path


def _fresh_manifest_entry(
    game,
    spec,
    seed,
    save_path,
    metadata_path,
    evaluation_directory,
):
    return {
        **asdict(spec),
        "suite_version": EVALUATION_SUITE_VERSION,
        "scenario": "fresh_game",
        "ending_condition": None,
        "regional_focus": None,
        "starting_score_by_seat": tuple(player.score for player in game.players),
        "starting_development_by_seat": tuple(0 for _player in game.players),
        "seed": seed,
        "save_file": save_path.relative_to(evaluation_directory).as_posix(),
        "metadata_file": metadata_path.relative_to(evaluation_directory).as_posix(),
    }


def _append_fresh_evaluation_suite(args):
    evaluation_directory = args.output / "evaluation"
    manifest_path = evaluation_directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Evaluation manifest does not exist at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(entry.get("evaluation_set") == "fresh" for entry in manifest):
        raise SystemExit("Fresh evaluation states already exist; nothing was overwritten")

    fresh_specs = [spec for spec in EVALUATION_SPECS if spec.evaluation_set == "fresh"]
    spec_indices = {spec.name: index for index, spec in enumerate(EVALUATION_SPECS)}
    for position, spec in enumerate(fresh_specs, 1):
        spec_index = spec_indices[spec.name]
        seed = _evaluation_seed(args.seed, spec_index, spec)
        game = fresh_evaluation_game(spec, seed)
        save_path, metadata_path = _save_fresh_evaluation_state(
            game,
            spec,
            seed,
            evaluation_directory,
        )
        manifest.append(
            _fresh_manifest_entry(
                game,
                spec,
                seed,
                save_path,
                metadata_path,
                evaluation_directory,
            )
        )
        print(f"{position}/{len(fresh_specs)} {spec.name}: {save_path}")

    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(f"Appended {len(fresh_specs)} Fresh states; suite now contains {len(manifest)} states")


def main(argv=None):
    args = parse_args(argv)
    if args.fresh_eval:
        _append_fresh_evaluation_suite(args)
        return
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
