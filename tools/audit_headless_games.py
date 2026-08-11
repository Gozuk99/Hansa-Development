"""Audit legal-action agreement and deterministic headless games across configurations."""

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.action_schema import ACTION_RANGES, action_schema_metadata
from game.action_validation import validate_action_state
from game.game_runner import create_headless_game, run_game


def validate_fresh_state(map_num, num_players, seed):
    started = time.time()
    game = create_headless_game(map_num, num_players, seed=seed)
    try:
        result = validate_action_state(game, quiet=True)
        return {
            "ok": result.legal_action_count == result.enabled_index_count,
            "map": map_num,
            "players": num_players,
            "seed": seed,
            "legal_action_count": result.legal_action_count,
            "enabled_index_count": result.enabled_index_count,
            "duration_seconds": round(time.time() - started, 3),
        }
    except Exception as error:
        return {
            "ok": False,
            "map": map_num,
            "players": num_players,
            "seed": seed,
            "error": f"{type(error).__name__}: {error}",
        }


def validate_complete_game(map_num, num_players, seed):
    started = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_game(map_num, num_players, seed)
            second = run_game(map_num, num_players, seed)
        deterministic = (
            first.action_trace == second.action_trace and first.final_scores == second.final_scores
        )
        trace_bytes = json.dumps(first.action_trace).encode()
        return {
            "ok": first.terminal_reason == "game_end" and deterministic,
            "map": map_num,
            "players": num_players,
            "seed": seed,
            "actions": first.action_count,
            "final_scores": first.final_scores,
            "deterministic": deterministic,
            "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "observed_action_indices": sorted(set(first.action_trace)),
            "duration_seconds": round(time.time() - started, 3),
        }
    except Exception as error:
        return {
            "ok": False,
            "map": map_num,
            "players": num_players,
            "seed": seed,
            "error": f"{type(error).__name__}: {error}",
        }


def run_parallel(function, tasks):
    workers = min(4, os.cpu_count() or 2)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, *task) for task in tasks]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--players", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[124, 125])
    parser.add_argument("--full-game-seeds", nargs="+", type=int, default=[124])
    parser.add_argument("--out", default="audit_results.json")
    args = parser.parse_args()

    fresh_tasks = [
        (map_num, players, seed)
        for map_num in args.maps
        for players in args.players
        for seed in args.seeds
    ]
    game_tasks = [
        (map_num, players, seed)
        for map_num in args.maps
        for players in args.players
        for seed in args.full_game_seeds
    ]
    fresh_results = run_parallel(validate_fresh_state, fresh_tasks)
    game_results = run_parallel(validate_complete_game, game_tasks)

    active_count = sum(action_range.active_capacity for action_range in ACTION_RANGES)
    reserved_count = sum(action_range.reserved_capacity for action_range in ACTION_RANGES)
    observed = sorted(
        {index for result in game_results for index in result.get("observed_action_indices", ())}
    )
    assigned = {
        index
        for action_range in ACTION_RANGES
        for index in range(action_range.start, action_range.active_stop)
    }
    report = {
        **action_schema_metadata(),
        "active_index_count": active_count,
        "reserved_index_count": reserved_count,
        "fresh_state_results": sorted(
            fresh_results, key=lambda item: (item["map"], item["players"], item["seed"])
        ),
        "complete_game_results": sorted(
            game_results, key=lambda item: (item["map"], item["players"], item["seed"])
        ),
        "observed_complete_game_indices": observed,
        "unobserved_complete_game_indices": sorted(assigned - set(observed)),
        "proven_unreachable_indices": [],
    }
    with open(args.out, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)

    failures = [result for result in fresh_results + game_results if not result["ok"]]
    print(
        f"Fresh states: {len(fresh_results) - len([r for r in fresh_results if not r['ok']])}"
        f"/{len(fresh_results)} passed"
    )
    print(
        f"Complete deterministic games: "
        f"{len(game_results) - len([r for r in game_results if not r['ok']])}"
        f"/{len(game_results)} passed"
    )
    print(f"Wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
