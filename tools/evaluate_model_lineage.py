"""Compare archived Hansa models with fixed evaluation and balanced head-to-head games."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.ai_model import HansaNN, SHARED_MODEL_FILE  # noqa: E402
from training.balanced_curriculum import BalancedCurriculumRunner  # noqa: E402
from training.curriculum import (  # noqa: E402
    ACTION_SEED_OFFSET,
    ACTIVE_EVALUATION_SETS,
    EVALUATION_RETRY_LIMIT,
    CurriculumConfig,
)
from training.self_play import (  # noqa: E402
    IncompleteGameError,
    PolicyTier,
    SelfPlayTrainer,
    TrainingConfig,
)


DEFAULT_ARCHIVE_DIRECTORY = ROOT / "training_output/archive"
DEFAULT_CURRENT_MODEL = ROOT / SHARED_MODEL_FILE
DEFAULT_EVALUATION_SUITE = ROOT / "training_data/generated/evaluation"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "training_output/model_lineage_evaluation"
MODEL_GLOB = f"{Path(SHARED_MODEL_FILE).stem}*{Path(SHARED_MODEL_FILE).suffix}"
T0_SELECTION_MODE = "T0"
EVALUATION_ACTION_LIMIT = 10_000

DISCOVERY_FIELDS = ("model", "model_path", "model_hash", "status", "detail")
SUMMARY_FIELDS = (
    "model",
    "model_path",
    "model_hash",
    "standard_evaluation_games",
    "early_t1_wins",
    "early_t1_win_rate",
    "mid_late_end_t1_wins",
    "mid_late_end_t1_win_rate",
    "overall_t1_wins",
    "overall_t1_win_rate",
    "average_actions",
    "timeouts",
    "evaluation_runtime_seconds",
)
HEAD_TO_HEAD_FIELDS = (
    "selection_mode",
    "archived_model",
    "current_model",
    "games",
    "archived_controlled_seats",
    "current_controlled_seats",
    "archived_wins",
    "current_wins",
    "archived_win_rate_per_controlled_seat",
    "current_win_rate_per_controlled_seat",
    "draws_or_ties",
    "archived_average_final_score",
    "current_average_final_score",
    "average_actions",
    "timeouts",
    "failed_games",
)
GAME_DETAIL_FIELDS = (
    "comparison",
    "selection_mode",
    "model",
    "archived_model",
    "current_model",
    "evaluation_set",
    "evaluation_state",
    "scenario",
    "map",
    "player_count",
    "state_seed",
    "action_seed",
    "seat_ownership",
    "seat_tiers",
    "selection_configuration",
    "winner_players",
    "winner_models",
    "final_scores",
    "completion_reason",
    "action_count",
    "retry_count",
    "failure_detail",
)


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    path: Path
    sha256: str


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(path):
    return HansaNN(model_file=path)


def discover_models(archive_directory, current_model, *, model_loader=_load_model):
    """Discover one unambiguous, loadable model per immediate archive directory."""
    archive_directory = Path(archive_directory)
    current_model = Path(current_model)
    artifacts = []
    rows = []

    candidates_by_name = []
    if archive_directory.is_dir():
        for directory in sorted(
            (path for path in archive_directory.iterdir() if path.is_dir()),
            key=lambda path: path.name.lower(),
        ):
            candidates_by_name.append((directory.name, tuple(sorted(directory.rglob(MODEL_GLOB)))))
    candidates_by_name.append(("current", (current_model,) if current_model.is_file() else ()))

    for name, candidates in candidates_by_name:
        if not candidates:
            rows.append(
                {
                    "model": name,
                    "model_path": "",
                    "model_hash": "",
                    "status": "missing",
                    "detail": f"No {MODEL_GLOB} model file found",
                }
            )
            continue
        if len(candidates) > 1:
            rows.append(
                {
                    "model": name,
                    "model_path": "",
                    "model_hash": "",
                    "status": "ambiguous",
                    "detail": "; ".join(str(path) for path in candidates),
                }
            )
            continue
        path = candidates[0].resolve()
        try:
            loaded_model = model_loader(path)
            del loaded_model
        except Exception as error:
            rows.append(
                {
                    "model": name,
                    "model_path": str(path),
                    "model_hash": "",
                    "status": "invalid",
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
            continue
        artifact = ModelArtifact(name, path, _sha256(path))
        artifacts.append(artifact)
        rows.append(
            {
                "model": name,
                "model_path": str(path),
                "model_hash": artifact.sha256,
                "status": "ready",
                "detail": "",
            }
        )
    return tuple(artifacts), tuple(rows)


def load_evaluation_manifest(directory, *, limit=None):
    manifest_path = Path(directory) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = tuple(
        entry
        for entry in manifest
        if entry.get("evaluation_set", "mid_late_end") in ACTIVE_EVALUATION_SETS
    )
    return entries if limit is None else entries[:limit]


def _materialize_evaluation_subset(source_directory, entries, destination):
    source_directory = Path(source_directory)
    destination = Path(destination)
    for entry in entries:
        for field in ("save_file", "metadata_file"):
            relative = Path(entry[field])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_directory / relative, target)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )


def _evaluation_trainer(model):
    return SelfPlayTrainer(
        model=model,
        config=TrainingConfig(
            max_actions=EVALUATION_ACTION_LIMIT,
            disable_move_action=False,
            shadow_filter_audit_enabled=False,
        ),
    )


def evaluate_standard_model(
    artifact,
    evaluation_suite,
    work_directory,
    *,
    evaluation_rotation=0,
    limit=None,
    progress_callback=None,
):
    """Run the canonical curriculum fixed evaluator without entering its training loop."""
    entries = load_evaluation_manifest(evaluation_suite, limit=limit)
    subset = Path(work_directory) / "fixed_suite"
    _materialize_evaluation_subset(evaluation_suite, entries, subset)
    trainer = _evaluation_trainer(_load_model(artifact.path))
    config = CurriculumConfig(iterations=1, training_games_per_batch=1)
    runner = BalancedCurriculumRunner(
        trainer,
        config,
        checkpoint_path=Path(work_directory) / "unused_checkpoint.pth",
        playable_model_path=artifact.path,
        csv_path=Path(work_directory) / "unused_results.csv",
        temporary_directory=Path(work_directory) / "states",
        failure_directory=Path(work_directory) / "failures",
        evaluation_suite_directory=subset,
        backup_directory=Path(work_directory) / "backups",
        progress_callback=progress_callback,
    )
    runner.batch_number = evaluation_rotation + 1
    runner.report_batch_number = 1
    started = perf_counter()
    rows, _trajectories, incomplete = runner._collect_evaluation(  # noqa: SLF001
        config.stages[runner.stage_index], Path(work_directory) / "states"
    )
    return tuple(rows), incomplete, perf_counter() - started


def _json_values(value):
    return tuple(json.loads(value)) if isinstance(value, str) else tuple(value)


def _t1_win_share(row):
    winner_tiers = _json_values(row["winner_tier"])
    return winner_tiers.count(1) / len(winner_tiers) if winner_tiers else 0.0


def summarize_standard(artifact, rows, runtime):
    def results_for(run):
        selected = [row for row in rows if row["run"] == run]
        wins = sum(_t1_win_share(row) for row in selected)
        return wins, wins / len(selected) if selected else None

    early_wins, early_rate = results_for("evaluation_early")
    developed_wins, developed_rate = results_for("evaluation_mid_late_end")
    overall_wins = sum(_t1_win_share(row) for row in rows)
    actions = [int(row["action_count"]) for row in rows]
    return {
        "model": artifact.name,
        "model_path": str(artifact.path),
        "model_hash": artifact.sha256,
        "standard_evaluation_games": len(rows),
        "early_t1_wins": early_wins,
        "early_t1_win_rate": early_rate,
        "mid_late_end_t1_wins": developed_wins,
        "mid_late_end_t1_win_rate": developed_rate,
        "overall_t1_wins": overall_wins,
        "overall_t1_win_rate": overall_wins / len(rows) if rows else None,
        "average_actions": sum(actions) / len(actions) if actions else None,
        "timeouts": sum(row["completion_reason"] == "action_limit" for row in rows),
        "evaluation_runtime_seconds": runtime,
    }


def standard_game_details(artifact, rows, entries):
    entries_by_seed = {entry["seed"]: entry for entry in entries}
    details = []
    for row in rows:
        entry = entries_by_seed[int(row["state_seed"])]
        details.append(
            {
                "comparison": "standard",
                "selection_mode": "",
                "model": artifact.name,
                "archived_model": "",
                "current_model": "",
                "evaluation_set": entry.get("evaluation_set", "mid_late_end"),
                "evaluation_state": entry["name"],
                "scenario": entry.get("scenario", ""),
                "map": row["map"],
                "player_count": row["player_count"],
                "state_seed": row["state_seed"],
                "action_seed": row["action_seed"],
                "seat_ownership": json.dumps([artifact.name] * int(row["player_count"])),
                "seat_tiers": row["tier_to_seat_assignments"],
                "selection_configuration": "fixed evaluation tiers; epsilon=0",
                "winner_players": row["winner_player"],
                "winner_models": json.dumps(
                    [artifact.name] * len(_json_values(row["winner_player"]))
                ),
                "final_scores": row["final_player_scores"],
                "completion_reason": row["completion_reason"],
                "action_count": row["action_count"],
                "retry_count": row["retry_count"],
                "failure_detail": "",
            }
        )
    return tuple(details)


class GreedyHeadToHeadTrainer(SelfPlayTrainer):
    """Use greedy semantic Q selection for every independent player seat."""

    _greedy_tier = PolicyTier(number=0, top_k=1, epsilon=0.0)

    def _assign_evaluation_tiers(self, player_count, _rotation):
        return (self._greedy_tier,) * player_count

    def _select_workflow_action(self, scores, legal_indices, exploration_categories=None):
        equivalent_groups = (
            None
            if exploration_categories is None
            else tuple(group for category in exploration_categories for group in category)
        )
        return self._select_action(
            scores,
            legal_indices,
            self._greedy_tier,
            equivalent_groups,
        )


def mirrored_seat_ownership(player_count, current_name="current", archived_name="archive"):
    first = tuple(
        current_name if index % 2 == 0 else archived_name for index in range(player_count)
    )
    second = tuple(archived_name if owner == current_name else current_name for owner in first)
    return first, second


def evaluate_head_to_head_pair(
    current_artifact,
    archived_artifact,
    evaluation_suite,
    *,
    limit=None,
    progress_callback=None,
    completed_pair_callback=None,
):
    entries = load_evaluation_manifest(evaluation_suite, limit=limit)
    current_model = _load_model(current_artifact.path)
    archived_model = _load_model(archived_artifact.path)
    trainer = GreedyHeadToHeadTrainer(
        model=current_model,
        config=TrainingConfig(
            max_actions=EVALUATION_ACTION_LIMIT,
            disable_move_action=False,
            shadow_filter_audit_enabled=False,
        ),
    )
    details = []
    models = {current_artifact.name: current_model, archived_artifact.name: archived_model}
    for entry in entries:
        state_path = Path(evaluation_suite) / entry["save_file"]
        ownership_pairs = mirrored_seat_ownership(
            entry["player_count"], current_artifact.name, archived_artifact.name
        )
        completed_pair = None
        failure_reason = "engine_dead_end"
        failure_detail = ""
        retry_count = 0
        for retry_count in range(EVALUATION_RETRY_LIMIT + 1):
            action_seed = int(entry["seed"]) + ACTION_SEED_OFFSET + retry_count
            pair = []
            for mirror_index, ownership in enumerate(ownership_pairs, start=1):
                trainer.rng.seed(action_seed)
                if progress_callback is not None:
                    retry = "" if retry_count == 0 else f", retry {retry_count}"
                    progress_callback(
                        f"Head-to-head {T0_SELECTION_MODE} {archived_artifact.name}, "
                        f"{entry['name']}, "
                        f"mirror {mirror_index}/2{retry}"
                    )
                try:
                    trajectory = trainer.collect_game(
                        state_path,
                        evaluation=True,
                        capture_action_limit=True,
                        evaluation_models_by_seat=tuple(models[owner] for owner in ownership),
                    )
                except IncompleteGameError as error:
                    failure_reason = "engine_dead_end"
                    failure_detail = str(error)
                    pair = []
                    break
                if trajectory.completion_reason == "no_replacement_route":
                    failure_reason = "no_replacement_route"
                    failure_detail = "No legal replacement route remained"
                    pair = []
                    break
                pair.append((ownership, trajectory))
            if len(pair) == len(ownership_pairs):
                completed_pair = pair
                break
            if progress_callback is not None and retry_count < EVALUATION_RETRY_LIMIT:
                detail = f": {failure_detail}" if failure_detail else ""
                progress_callback(
                    f"Head-to-head {T0_SELECTION_MODE} {archived_artifact.name}, "
                    f"{entry['name']}: "
                    f"{failure_reason}{detail}; retrying both mirrors"
                )

        pair_details = []
        if completed_pair is not None:
            for ownership, trajectory in completed_pair:
                winner_players = tuple(index + 1 for index in trajectory.winner_indices)
                winner_models = tuple(ownership[index] for index in trajectory.winner_indices)
                pair_details.append(
                    _head_to_head_detail(
                        current_artifact,
                        archived_artifact,
                        entry,
                        ownership,
                        action_seed,
                        retry_count,
                        trajectory=trajectory,
                        winner_players=winner_players,
                        winner_models=winner_models,
                    )
                )
        else:
            if progress_callback is not None:
                detail = f": {failure_detail}" if failure_detail else ""
                progress_callback(
                    f"Head-to-head {T0_SELECTION_MODE} {archived_artifact.name}, "
                    f"{entry['name']}: "
                    f"recording {failure_reason}{detail} after {retry_count} retries"
                )
            pair_details.extend(
                _head_to_head_detail(
                    current_artifact,
                    archived_artifact,
                    entry,
                    ownership,
                    action_seed,
                    retry_count,
                    failure_reason=failure_reason,
                    failure_detail=failure_detail,
                )
                for ownership in ownership_pairs
            )
        details.extend(pair_details)
        if completed_pair_callback is not None:
            completed_pair_callback(tuple(pair_details))
    return tuple(details)


def _head_to_head_detail(
    current_artifact,
    archived_artifact,
    entry,
    ownership,
    action_seed,
    retry_count,
    *,
    trajectory=None,
    winner_players=(),
    winner_models=(),
    failure_reason="",
    failure_detail="",
):
    return {
        "comparison": "head_to_head",
        "selection_mode": T0_SELECTION_MODE,
        "model": "",
        "archived_model": archived_artifact.name,
        "current_model": current_artifact.name,
        "evaluation_set": entry.get("evaluation_set", "mid_late_end"),
        "evaluation_state": entry["name"],
        "scenario": entry.get("scenario", ""),
        "map": entry["map_num"],
        "player_count": entry["player_count"],
        "state_seed": entry["seed"],
        "action_seed": action_seed,
        "seat_ownership": json.dumps(ownership),
        "seat_tiers": json.dumps(
            trajectory.seat_tiers if trajectory is not None else (0,) * entry["player_count"]
        ),
        "selection_configuration": "all seats T0 greedy semantic Q Top-1; epsilon=0; policy unused",
        "winner_players": json.dumps(winner_players),
        "winner_models": json.dumps(winner_models),
        "final_scores": json.dumps(trajectory.final_scores if trajectory is not None else ()),
        "completion_reason": (
            trajectory.completion_reason if trajectory is not None else failure_reason
        ),
        "action_count": len(trajectory.action_trace) if trajectory is not None else "",
        "retry_count": retry_count,
        "failure_detail": failure_detail,
    }


def summarize_head_to_head(current_artifact, archived_artifact, details):
    names = (archived_artifact.name, current_artifact.name)
    seats = {name: 0 for name in names}
    wins = {name: 0.0 for name in names}
    score_totals = {name: 0 for name in names}
    scored_seats = {name: 0 for name in names}
    draws = 0
    for row in details:
        ownership = _json_values(row["seat_ownership"])
        scores = _json_values(row["final_scores"])
        winners = _json_values(row["winner_models"])
        for owner, score in zip(ownership, scores, strict=True):
            score_totals[owner] += score
            scored_seats[owner] += 1
        for owner in ownership:
            seats[owner] += 1
        if len(winners) > 1:
            draws += 1
        for winner in winners:
            wins[winner] += 1 / len(winners)
    action_counts = [int(row["action_count"]) for row in details if row["action_count"] != ""]
    return {
        "selection_mode": details[0]["selection_mode"] if details else "",
        "archived_model": archived_artifact.name,
        "current_model": current_artifact.name,
        "games": len(details),
        "archived_controlled_seats": seats[archived_artifact.name],
        "current_controlled_seats": seats[current_artifact.name],
        "archived_wins": wins[archived_artifact.name],
        "current_wins": wins[current_artifact.name],
        "archived_win_rate_per_controlled_seat": (
            wins[archived_artifact.name] / seats[archived_artifact.name]
        ),
        "current_win_rate_per_controlled_seat": (
            wins[current_artifact.name] / seats[current_artifact.name]
        ),
        "draws_or_ties": draws,
        "archived_average_final_score": (
            score_totals[archived_artifact.name] / scored_seats[archived_artifact.name]
            if scored_seats[archived_artifact.name]
            else None
        ),
        "current_average_final_score": (
            score_totals[current_artifact.name] / scored_seats[current_artifact.name]
            if scored_seats[current_artifact.name]
            else None
        ),
        "average_actions": sum(action_counts) / len(action_counts) if action_counts else None,
        "timeouts": sum(row["completion_reason"] == "action_limit" for row in details),
        "failed_games": sum(
            row["completion_reason"] in {"engine_dead_end", "no_replacement_route"}
            for row in details
        ),
    }


def _write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            delete=False,
            newline="",
            encoding="utf-8",
        ) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_lineage_evaluation(
    *,
    archive_directory=DEFAULT_ARCHIVE_DIRECTORY,
    current_model=DEFAULT_CURRENT_MODEL,
    evaluation_suite=DEFAULT_EVALUATION_SUITE,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    evaluation_rotation=0,
    standard_limit=None,
    head_to_head_limit=None,
    progress_callback=print,
):
    artifacts, discovery_rows = discover_models(archive_directory, current_model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = Path(output_directory) / timestamp
    run_directory.mkdir(parents=True, exist_ok=False)
    _write_csv(run_directory / "model_discovery.csv", DISCOVERY_FIELDS, discovery_rows)
    _write_csv(run_directory / "model_summary.csv", SUMMARY_FIELDS, ())
    _write_csv(run_directory / "head_to_head.csv", HEAD_TO_HEAD_FIELDS, ())
    _write_csv(run_directory / "game_details.csv", GAME_DETAIL_FIELDS, ())
    current = next((artifact for artifact in artifacts if artifact.name == "current"), None)
    if current is None:
        raise RuntimeError(
            "The current root model is missing, ambiguous, or invalid; "
            f"see {run_directory / 'model_discovery.csv'}"
        )

    entries = load_evaluation_manifest(evaluation_suite, limit=standard_limit)
    summary_rows = []
    game_details = []
    for artifact in artifacts:
        if progress_callback is not None:
            progress_callback(f"Standard fixed evaluation: {artifact.name}")
        with tempfile.TemporaryDirectory(dir=run_directory) as temporary:
            rows, _incomplete, runtime = evaluate_standard_model(
                artifact,
                evaluation_suite,
                temporary,
                evaluation_rotation=evaluation_rotation,
                limit=standard_limit,
                progress_callback=None,
            )
        summary_rows.append(summarize_standard(artifact, rows, runtime))
        game_details.extend(standard_game_details(artifact, rows, entries))
        _write_csv(run_directory / "model_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(run_directory / "game_details.csv", GAME_DETAIL_FIELDS, game_details)

    head_to_head_rows = []
    for archived in (artifact for artifact in artifacts if artifact.name != "current"):

        def save_completed_pair(pair_details):
            game_details.extend(pair_details)
            _write_csv(run_directory / "game_details.csv", GAME_DETAIL_FIELDS, game_details)

        details = evaluate_head_to_head_pair(
            current,
            archived,
            evaluation_suite,
            limit=head_to_head_limit,
            progress_callback=progress_callback,
            completed_pair_callback=save_completed_pair,
        )
        head_to_head_rows.append(summarize_head_to_head(current, archived, details))
        _write_csv(
            run_directory / "head_to_head.csv",
            HEAD_TO_HEAD_FIELDS,
            head_to_head_rows,
        )

    return run_directory, artifacts, discovery_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_DIRECTORY)
    parser.add_argument("--current-model", type=Path, default=DEFAULT_CURRENT_MODEL)
    parser.add_argument("--evaluation-suite", type=Path, default=DEFAULT_EVALUATION_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--evaluation-rotation", type=int, default=0)
    parser.add_argument(
        "--standard-limit",
        type=int,
        default=None,
        help="Limit fixed states per model for a smoke test (default: complete suite)",
    )
    parser.add_argument(
        "--head-to-head-limit",
        type=int,
        default=None,
        help="Limit fixed states per archive comparison for a smoke test (default: complete suite)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_directory, artifacts, discovery_rows = run_lineage_evaluation(
        archive_directory=args.archive,
        current_model=args.current_model,
        evaluation_suite=args.evaluation_suite,
        output_directory=args.output,
        evaluation_rotation=args.evaluation_rotation,
        standard_limit=args.standard_limit,
        head_to_head_limit=args.head_to_head_limit,
    )
    ready = ", ".join(artifact.name for artifact in artifacts)
    unavailable = [row for row in discovery_rows if row["status"] != "ready"]
    print(f"Models evaluated: {ready}")
    for row in unavailable:
        print(f"Model {row['model']}: {row['status']} ({row['detail']})")
    print(f"Results: {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
