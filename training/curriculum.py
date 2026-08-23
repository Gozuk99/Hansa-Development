"""Standalone curriculum orchestration for local Hansa self-play training."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
import traceback

from game.persistence import save_game
from training.self_play import (
    ActionLimitExceeded,
    IncompleteGameError,
    SelfPlayTrainer,
)
from training.targeted_state_generator import StateGenerationError


CSV_FIELDS = (
    "game#",
    "batch#",
    "curriculum_stage",
    "starting_position",
    "run_type",
    "evaluation_set",
    "map",
    "player_count",
    "starting_score_by_seat",
    "starting_development_by_seat",
    "development_role_by_seat",
    "early_route_scaffold",
    "scaffolded_route_ids_by_seat",
    "scaffolded_route_lengths_by_seat",
    "state_seed",
    "action_seed",
    "winner_player",
    "winner_tier",
    "tier_to_seat_assignments",
    "final_player_scores",
    "completion_reason",
    "action_count",
    "early_training_action_limit_outcome",
    "trajectory_decision_count",
    "sampled_training_decision_count",
    "sampled_training_fraction",
    "sampled_octile_1",
    "sampled_octile_2",
    "sampled_octile_3",
    "sampled_octile_4",
    "sampled_octile_5",
    "sampled_octile_6",
    "sampled_octile_7",
    "sampled_octile_8",
    "move_action_count",
    "spent_action_count",
    "move_ratio",
    "pointless_move_workflows",
    "repeated_move_penalties",
    "all_move_turn_penalties",
    "moves_creating_claimable_route",
    "move_claim_conversions",
    "move_claim_conversion_rate",
    "retry_count",
    "latest_loss",
    "rolling_mean_loss",
    "evaluation_suite_size",
    "evaluation_suite_version",
    "generation_seconds",
    "play_seconds",
    "inference_seconds",
    "scoring_seconds",
    "execution_seconds",
    "validation_seconds",
    "observation_seconds",
    "legality_seconds",
    "selection_seconds",
    "context_seconds",
    "reward_seconds",
    "learning_seconds",
)


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    action_limit: int
    score_range: tuple[int, int] | None = None
    full_game: bool = False


DEFAULT_STAGES = (
    CurriculumStage("near_end_18_19", 10_000, (10, 17)),
    CurriculumStage("late_game_15_17", 10_000, (15, 17)),
    CurriculumStage("mid_game", 6_000, (8, 14)),
    CurriculumStage("early_game", 8_000, (0, 7)),
    CurriculumStage("full_game", 10_000, full_game=True),
)
ACTION_SEED_OFFSET = 1_000_000_007
EVALUATION_RETRY_LIMIT = 2
EVALUATION_CONFIGURATIONS = tuple(
    (map_num, player_count) for map_num in (1, 2, 3) for player_count in (3, 4, 5)
)


@dataclass(frozen=True)
class PromotionCriteria:
    maximum_unfinished_rate: float = 0.05
    minimum_evaluation_completion_rate: float = 0.95
    require_tier_one_advantage: bool = True
    loss_tolerance: float = 0.10
    rolling_loss_window: int = 5

    def __post_init__(self):
        for name, value in (
            ("maximum unfinished rate", self.maximum_unfinished_rate),
            ("minimum evaluation completion rate", self.minimum_evaluation_completion_rate),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.loss_tolerance < 0:
            raise ValueError("loss tolerance cannot be negative")
        if self.rolling_loss_window < 1:
            raise ValueError("rolling loss window must be positive")


@dataclass(frozen=True)
class CurriculumConfig:
    iterations: int = 1
    training_games_per_batch: int = 5
    evaluation_games_per_batch: int = 1
    update_batch_size: int = 5
    retry_limit: int = 5
    seed: int = 124
    evaluation_seed: int = 10_000
    stages: tuple[CurriculumStage, ...] = DEFAULT_STAGES
    promotion: PromotionCriteria = field(default_factory=PromotionCriteria)

    def __post_init__(self):
        for name, value in (
            ("iterations", self.iterations),
            ("training games", self.training_games_per_batch),
            ("evaluation games", self.evaluation_games_per_batch),
            ("update batch size", self.update_batch_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.retry_limit < 0:
            raise ValueError("retry limit cannot be negative")
        if not self.stages:
            raise ValueError("at least one curriculum stage is required")


@dataclass(frozen=True)
class StateDescriptor:
    path: Path
    metadata_path: Path | None
    map_num: int
    player_count: int
    seed: int
    scenario: str | None = None
    starting_position: str = "full_game"
    evaluation_set: str | None = None
    starting_scores_by_seat: tuple[int, ...] = ()
    starting_development_by_seat: tuple[int, ...] = ()
    development_roles_by_seat: tuple[str, ...] = ()
    early_route_scaffold: bool | None = None
    scaffolded_route_ids_by_seat: tuple[tuple[int, ...], ...] = ()
    scaffolded_route_lengths_by_seat: tuple[tuple[int, ...], ...] = ()

    @property
    def action_seed(self):
        return self.seed + ACTION_SEED_OFFSET


class CurriculumRunError(RuntimeError):
    """Raised when the standalone curriculum cannot safely continue."""


def _play_timing_breakdown(trajectory):
    measured = sum(
        (
            trajectory.observation_seconds,
            trajectory.legality_seconds,
            trajectory.inference_seconds,
            trajectory.selection_seconds,
            trajectory.context_seconds,
            trajectory.scoring_seconds,
            trajectory.execution_seconds,
            trajectory.validation_seconds,
            trajectory.reward_seconds,
        )
    )
    loop_seconds = max(trajectory.play_seconds - measured, 0.0)
    timings = (
        ("observation", trajectory.observation_seconds),
        ("legality", trajectory.legality_seconds),
        ("inference", trajectory.inference_seconds),
        ("selection", trajectory.selection_seconds),
        ("context/threats", trajectory.context_seconds),
        ("scoring", trajectory.scoring_seconds),
        ("execution", trajectory.execution_seconds),
        ("validation", trajectory.validation_seconds),
        ("rewards", trajectory.reward_seconds),
        ("loop/control", loop_seconds),
    )
    return ", ".join(f"{name} {seconds:.2f}s" for name, seconds in timings if seconds >= 1)


def _format_game_numbers(game_numbers):
    numbers = list(dict.fromkeys(game_numbers))
    if len(numbers) > 1 and numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"{numbers[0]}-{numbers[-1]}"
    if len(numbers) < 2:
        return str(numbers[0])
    if len(numbers) == 2:
        return f"{numbers[0]} and {numbers[1]}"
    return f"{', '.join(map(str, numbers[:-1]))}, and {numbers[-1]}"


def _rounded_seconds(value):
    return round(float(value), 2)


class CurriculumRunner:
    def __init__(
        self,
        trainer: SelfPlayTrainer,
        config: CurriculumConfig,
        *,
        checkpoint_path,
        playable_model_path,
        csv_path,
        temporary_directory="training_output/curriculum/states",
        failure_directory="training_data/failures",
        evaluation_suite_directory="training_data/generated/evaluation",
        progress_callback=None,
    ):
        self.trainer = trainer
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.playable_model_path = Path(playable_model_path)
        self.csv_path = Path(csv_path)
        self.temporary_directory = Path(temporary_directory)
        self.failure_directory = Path(failure_directory)
        self.evaluation_suite_directory = Path(evaluation_suite_directory)
        self.progress_callback = progress_callback
        self._captured_errors = set()
        self._latest_descriptor = None
        saved = trainer.curriculum_state or {}
        signature = self._configuration_signature()
        compatible_signatures = {
            signature,
            self._configuration_signature(late_game_action_limit=4_000),
            self._configuration_signature(near_end_score_range=(10, 18)),
            self._configuration_signature(near_end_action_limit=6_000),
            self._configuration_signature(
                near_end_action_limit=6_000, near_end_score_range=(10, 18)
            ),
            self._configuration_signature(near_end_action_limit=6_000, retry_limit=2),
            self._configuration_signature(
                near_end_action_limit=6_000,
                near_end_score_range=(10, 18),
                retry_limit=2,
            ),
            self._configuration_signature(retry_limit=2),
            self._configuration_signature(
                near_end_action_limit=3_000, near_end_score_range=(18, 19)
            ),
            self._configuration_signature(
                near_end_action_limit=3_000,
                near_end_score_range=(18, 19),
                retry_limit=2,
            ),
            self._configuration_signature(near_end_action_limit=2_000, retry_limit=2),
        }
        if (
            saved.get("configuration_version", 1) >= 4
            and saved.get("configuration_signature") not in compatible_signatures
        ):
            raise ValueError("Checkpoint curriculum configuration is incompatible")
        self.stage_index = saved.get("stage_index", 0)
        self.batch_number = saved.get("batch_number", 0)
        self.game_number = saved.get("game_number", 0)
        self.rolling_losses = list(saved.get("rolling_losses", ()))
        self.report_game_number = 0
        self.report_batch_number = 0
        if self.csv_path.is_file():
            with self.csv_path.open(newline="", encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
            if rows:
                self.report_game_number = max(int(row["game#"]) for row in rows)
                self.report_batch_number = max(int(row["batch#"]) for row in rows)
        if self.stage_index >= len(config.stages):
            raise ValueError("Checkpoint curriculum stage is not configured")

    def _report(self, message):
        if self.progress_callback is not None:
            if hasattr(self, "run_batch_number"):
                message = f"[Batch {self.run_batch_number}/{self.config.iterations}] {message}"
            self.progress_callback(message)

    def _evaluation_game_count(self):
        manifest_path = self.evaluation_suite_directory / "manifest.json"
        if not manifest_path.is_file():
            return self.config.evaluation_games_per_batch
        return len(json.loads(manifest_path.read_text(encoding="utf-8")))

    def _configuration_signature(
        self,
        *,
        near_end_action_limit=None,
        near_end_score_range=None,
        late_game_action_limit=None,
        retry_limit=None,
    ):
        stages = [
            replace(
                stage,
                action_limit=(
                    near_end_action_limit
                    if stage.name == "near_end_18_19" and near_end_action_limit is not None
                    else late_game_action_limit
                    if stage.name == "late_game_15_17" and late_game_action_limit is not None
                    else stage.action_limit
                ),
                score_range=(
                    near_end_score_range
                    if stage.name == "near_end_18_19" and near_end_score_range is not None
                    else stage.score_range
                ),
            )
            if (
                stage.name == "near_end_18_19"
                and (near_end_action_limit is not None or near_end_score_range is not None)
            )
            or (stage.name == "late_game_15_17" and late_game_action_limit is not None)
            else stage
            for stage in self.config.stages
        ]
        data = {
            "update_batch_size": self.config.update_batch_size,
            "retry_limit": self.config.retry_limit if retry_limit is None else retry_limit,
            "seed": self.config.seed,
            "evaluation_seed": self.config.evaluation_seed,
            "stages": [asdict(stage) for stage in stages],
            "promotion": asdict(self.config.promotion),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _curriculum_state(self):
        return {
            "configuration_version": 4,
            "configuration_signature": self._configuration_signature(),
            "stage_index": self.stage_index,
            "batch_number": self.batch_number,
            "game_number": self.game_number,
            "rolling_losses": self.rolling_losses,
        }

    def _generate_state(self, stage, seed, directory, *, map_num=None, player_count=None):
        raise NotImplementedError

    @staticmethod
    def _stage_label(stage):
        return "full_game" if stage.full_game else "mixed_end_game"

    @staticmethod
    def _stage_action_limit(stage):
        return stage.action_limit

    def _training_action_limit(self, stage, _descriptor):
        return self._stage_action_limit(stage)

    def _evaluation_action_limit(self, stage, _descriptor):
        return self._stage_action_limit(stage)

    def _failure_callback(self, stage, descriptor, retry_count, run_type):
        def capture(game, action_trace, seat_tiers, error):
            directory = self._save_failure(
                stage,
                descriptor,
                retry_count,
                run_type,
                error,
                game=game,
                action_trace=action_trace,
                seat_tiers=seat_tiers,
            )
            self._captured_errors.add(id(error))
            if isinstance(error, IncompleteGameError) and not isinstance(
                error, ActionLimitExceeded
            ):
                self._report(f"Saved no-legal-interaction state: {directory}")

        return capture

    def _save_failure(
        self,
        stage,
        descriptor,
        retry_count,
        run_type,
        error,
        *,
        game=None,
        action_trace=(),
        seat_tiers=(),
    ):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.failure_directory.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"{timestamp}_", dir=self.failure_directory))
        if descriptor is not None and descriptor.path.is_file():
            shutil.copy2(descriptor.path, directory / "source_state.hansa")
        latest_state_error = None
        if game is not None:
            try:
                save_game(game, directory / "latest_state.hansa")
            except Exception as save_error:
                latest_state_error = repr(save_error)
        (directory / "action_trace.json").write_text(
            json.dumps(list(action_trace), indent=2) + "\n", encoding="utf-8"
        )
        details = {
            "curriculum_stage": descriptor.scenario or stage.name,
            "starting_position": descriptor.starting_position,
            "run_type": run_type,
            "evaluation_set": None if descriptor is None else descriptor.evaluation_set,
            "state_seed": None if descriptor is None else descriptor.seed,
            "action_seed": None if descriptor is None else descriptor.action_seed,
            "map": None if descriptor is None else descriptor.map_num,
            "player_count": None if descriptor is None else descriptor.player_count,
            "tier_assignments": [tier.number for tier in seat_tiers],
            "source_state": None if descriptor is None else str(descriptor.path),
            "checkpoint_path": str(self.checkpoint_path),
            "retry_count": retry_count,
            "exception": repr(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "latest_state_save_error": latest_state_error,
        }
        (directory / "diagnostics.json").write_text(
            json.dumps(details, indent=2) + "\n", encoding="utf-8"
        )
        return directory

    def _trajectory_row(
        self,
        trajectory,
        descriptor,
        stage,
        run_type,
        retry_count,
        latest_loss,
        rolling_mean_loss,
        game_number,
        *,
        generation_seconds=0.0,
        learning_seconds=0.0,
        action_seed=None,
        evaluation_suite_size=None,
        evaluation_suite_version=None,
        training_sample_coverage=None,
    ):
        winner_players = [index + 1 for index in trajectory.winner_indices]
        winner_tiers = [trajectory.seat_tiers[index] for index in trajectory.winner_indices]
        move_action_count = getattr(trajectory, "move_action_count", 0)
        spent_action_count = getattr(trajectory, "spent_action_count", 0)
        move_ratio = getattr(trajectory, "move_ratio", None)
        if move_ratio is None and spent_action_count:
            move_ratio = move_action_count / spent_action_count
        moves_creating_claimable_route = getattr(trajectory, "moves_creating_claimable_route", 0)
        move_claim_conversions = getattr(trajectory, "move_claim_conversions", 0)
        move_claim_conversion_rate = getattr(trajectory, "move_claim_conversion_rate", None)
        if move_claim_conversion_rate is None and moves_creating_claimable_route:
            move_claim_conversion_rate = move_claim_conversions / moves_creating_claimable_route
        action_count = len(trajectory.action_trace)
        completion_reason = getattr(trajectory, "completion_reason", "normal")
        maturity = (descriptor.scenario or stage.name).partition("+")[0]
        early_training_action_limit_outcome = None
        if run_type in {"training", "training_timeout"} and maturity in {
            "early",
            "early_mixed",
        }:
            if completion_reason == "action_limit":
                early_training_action_limit_outcome = "timed_out_at_15000"
            elif action_count <= 10_000:
                early_training_action_limit_outcome = "finished_by_10000"
            else:
                early_training_action_limit_outcome = "finished_10001_to_15000"
        sampled_octiles = (
            getattr(training_sample_coverage, "sampled_octiles", ())
            if training_sample_coverage
            else ()
        )
        return {
            "game#": game_number,
            "batch#": self.report_batch_number,
            "curriculum_stage": descriptor.scenario or stage.name,
            "starting_position": descriptor.starting_position,
            "run_type": run_type,
            "evaluation_set": descriptor.evaluation_set,
            "map": descriptor.map_num,
            "player_count": descriptor.player_count,
            "starting_score_by_seat": json.dumps(descriptor.starting_scores_by_seat),
            "starting_development_by_seat": json.dumps(descriptor.starting_development_by_seat),
            "development_role_by_seat": json.dumps(descriptor.development_roles_by_seat),
            "early_route_scaffold": (
                str(descriptor.early_route_scaffold).lower()
                if descriptor.early_route_scaffold is not None
                else None
            ),
            "scaffolded_route_ids_by_seat": json.dumps(descriptor.scaffolded_route_ids_by_seat),
            "scaffolded_route_lengths_by_seat": json.dumps(
                descriptor.scaffolded_route_lengths_by_seat
            ),
            "state_seed": descriptor.seed,
            "action_seed": descriptor.action_seed if action_seed is None else action_seed,
            "winner_player": json.dumps(winner_players),
            "winner_tier": json.dumps(winner_tiers),
            "tier_to_seat_assignments": json.dumps(trajectory.seat_tiers),
            "final_player_scores": json.dumps(trajectory.final_scores),
            "completion_reason": completion_reason,
            "action_count": action_count,
            "early_training_action_limit_outcome": early_training_action_limit_outcome,
            "trajectory_decision_count": (
                training_sample_coverage.total_decisions if training_sample_coverage else None
            ),
            "sampled_training_decision_count": (
                training_sample_coverage.sampled_decisions if training_sample_coverage else None
            ),
            "sampled_training_fraction": (
                training_sample_coverage.sampled_fraction if training_sample_coverage else None
            ),
            **{
                f"sampled_octile_{index + 1}": (
                    sampled_octiles[index] if len(sampled_octiles) == 8 else None
                )
                for index in range(8)
            },
            "move_action_count": move_action_count,
            "spent_action_count": spent_action_count,
            "move_ratio": move_ratio,
            "pointless_move_workflows": getattr(trajectory, "pointless_move_workflows", 0),
            "repeated_move_penalties": getattr(trajectory, "repeated_move_penalties", 0),
            "all_move_turn_penalties": getattr(trajectory, "all_move_turn_penalties", 0),
            "moves_creating_claimable_route": moves_creating_claimable_route,
            "move_claim_conversions": move_claim_conversions,
            "move_claim_conversion_rate": move_claim_conversion_rate,
            "retry_count": retry_count,
            "latest_loss": latest_loss,
            "rolling_mean_loss": rolling_mean_loss,
            "evaluation_suite_size": evaluation_suite_size,
            "evaluation_suite_version": evaluation_suite_version,
            "generation_seconds": _rounded_seconds(generation_seconds),
            "play_seconds": _rounded_seconds(trajectory.play_seconds),
            "inference_seconds": _rounded_seconds(trajectory.inference_seconds),
            "scoring_seconds": _rounded_seconds(trajectory.scoring_seconds),
            "execution_seconds": _rounded_seconds(trajectory.execution_seconds),
            "validation_seconds": _rounded_seconds(trajectory.validation_seconds),
            "observation_seconds": _rounded_seconds(trajectory.observation_seconds),
            "legality_seconds": _rounded_seconds(trajectory.legality_seconds),
            "selection_seconds": _rounded_seconds(trajectory.selection_seconds),
            "context_seconds": _rounded_seconds(trajectory.context_seconds),
            "reward_seconds": _rounded_seconds(trajectory.reward_seconds),
            "learning_seconds": _rounded_seconds(learning_seconds),
        }

    def _append_csv(self, rows):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        replacement = None
        if self.csv_path.is_file():
            with self.csv_path.open(newline="", encoding="utf-8-sig") as source:
                reader = csv.DictReader(source)
                existing_fields = tuple(reader.fieldnames or ())
                if existing_fields != CSV_FIELDS:
                    with tempfile.NamedTemporaryFile(
                        "w",
                        dir=self.csv_path.parent,
                        delete=False,
                        newline="",
                        encoding="utf-8",
                    ) as output:
                        replacement = Path(output.name)
                        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
                        writer.writeheader()
                        for existing in reader:
                            writer.writerow(
                                {field: existing.get(field, "") for field in CSV_FIELDS}
                            )
            if replacement is not None:
                replacement.replace(self.csv_path)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _collect_training(self, stage, directory):
        descriptors = []
        pending_update = []
        pending_rows = []
        pending_game_numbers = []
        rows = []
        recent_losses = []
        unfinished = 0

        def save_completed_group():
            if not pending_update:
                return
            self.trainer.save_checkpoint(
                self.checkpoint_path,
                [descriptor.path for descriptor in descriptors],
                curriculum_state=self._curriculum_state(),
            )
            self.trainer.model.save_model(self.playable_model_path)
            self._append_csv(pending_rows)
            self._report(f"Saved training games {_format_game_numbers(pending_game_numbers)}")
            pending_update.clear()
            pending_rows.clear()
            pending_game_numbers.clear()

        total_games = self.config.training_games_per_batch
        game_index = 0
        while game_index < total_games:
            completed_game = False
            retry_reason = None
            for retry_count in range(self.config.retry_limit + 1):
                retry = (
                    ""
                    if retry_count == 0
                    else f" (retry {retry_count}: {retry_reason or 'unknown'})"
                )
                self._report(f"Training game {game_index + 1}/{total_games}{retry}...")
                seed = self.config.seed + self.game_number * 10_007 + retry_count
                generation_started = perf_counter()
                try:
                    descriptor = self._generate_state(stage, seed, directory)
                except StateGenerationError as error:
                    unfinished += 1
                    if retry_count == self.config.retry_limit:
                        self.game_number += 1
                        self._report(
                            f"Discarded generator seed after {retry_count} retries; "
                            f"continuing training game {game_index + 1}/{total_games}"
                        )
                        break
                    retry_reason = f"generation constraints: {error}"
                    continue
                generation_seconds = perf_counter() - generation_started
                self.trainer.rng.seed(descriptor.action_seed)
                self.trainer.config = replace(
                    self.trainer.config,
                    max_actions=self._training_action_limit(stage, descriptor),
                )
                try:
                    trajectory = self.trainer.collect_game(
                        descriptor.path,
                        failure_callback=self._failure_callback(
                            stage, descriptor, retry_count, "training"
                        ),
                    )
                except IncompleteGameError as error:
                    unfinished += 1
                    reason = (
                        "action_limit"
                        if isinstance(error, ActionLimitExceeded)
                        else "no_legal_interaction"
                    )
                    if retry_count == self.config.retry_limit:
                        self.game_number += 1
                        self._report(
                            f"Discarded starting position after {retry_count} retries "
                            f"({reason}); "
                            f"continuing training game {game_index + 1}/{total_games}"
                        )
                        break
                    retry_reason = (
                        f"{reason}: {error}" if reason == "no_legal_interaction" else reason
                    )
                    continue
                game_loss = self.trainer.trajectory_loss(trajectory)
                maturity = (descriptor.scenario or stage.name).partition("+")[0]
                result = getattr(trajectory, "completion_reason", "normal")
                descriptors.append(descriptor)
                learning_started = perf_counter()
                self.trainer.update_model((trajectory,), curriculum_maturities=(maturity,))
                learning_seconds = perf_counter() - learning_started
                sample_coverage = getattr(self.trainer, "last_training_sample_coverage", ())
                training_sample_coverage = sample_coverage[0] if sample_coverage else None
                loss = f"; loss {game_loss:.2f}" if game_loss is not None else ""
                slow_timings = _play_timing_breakdown(trajectory)
                timing = f" ({slow_timings})" if slow_timings else ""
                generation = (
                    f"; generate {generation_seconds:.2f}s" if generation_seconds >= 1 else ""
                )
                learning = f"; learn {learning_seconds:.2f}s" if learning_seconds >= 1 else ""
                self._report(
                    f"Training game {game_index + 1}/{total_games}: {result}; "
                    f"{len(trajectory.action_trace)} actions; "
                    f"play {trajectory.play_seconds:.2f}s{loss}{timing}{generation}{learning}"
                )
                pending_update.append(trajectory)
                pending_game_numbers.append(game_index + 1)
                completed = result not in {"no_replacement_route", "action_limit"}
                if not completed:
                    unfinished += 1
                    if result == "action_limit":
                        self.game_number += 1
                        self.report_game_number += 1
                        pending_rows.append(
                            self._trajectory_row(
                                trajectory,
                                descriptor,
                                stage,
                                "training_timeout",
                                retry_count,
                                game_loss,
                                None,
                                self.report_game_number,
                                generation_seconds=generation_seconds,
                                learning_seconds=learning_seconds,
                                training_sample_coverage=training_sample_coverage,
                            )
                        )
                        if len(pending_update) == self.config.update_batch_size:
                            save_completed_group()
                        completed_game = True
                        break
                    if len(pending_update) == self.config.update_batch_size:
                        save_completed_group()
                    if retry_count < self.config.retry_limit:
                        retry_reason = "no_replacement_route"
                        continue
                    save_completed_group()
                    self.game_number += 1
                    self._report(
                        f"Discarded starting position after {retry_count} retries; "
                        f"continuing training game {game_index + 1}/{total_games}"
                    )
                    break

                self.game_number += 1
                self.report_game_number += 1
                completed_game = True
                if game_loss is not None:
                    recent_losses.append(game_loss)
                rolling_mean = (
                    sum(recent_losses[-5:]) / len(recent_losses[-5:]) if recent_losses else None
                )
                row = self._trajectory_row(
                    trajectory,
                    descriptor,
                    stage,
                    "training",
                    retry_count,
                    game_loss,
                    rolling_mean,
                    self.report_game_number,
                    generation_seconds=generation_seconds,
                    learning_seconds=learning_seconds,
                    training_sample_coverage=training_sample_coverage,
                )
                rows.append(row)
                pending_rows.append(row)
                if len(pending_update) == self.config.update_batch_size:
                    save_completed_group()
                break
            if completed_game:
                game_index += 1
        save_completed_group()
        return rows, descriptors, unfinished

    def _collect_evaluation(self, stage, directory):
        rows = []
        trajectories = []
        incomplete = 0
        progress = deepcopy(self.trainer.progress)
        rng_state = self.trainer.rng.getstate()
        try:
            manifest_path = self.evaluation_suite_directory / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                evaluation_set_sizes = {}
                for item in manifest:
                    evaluation_set = item.get("evaluation_set", "mid_late_end")
                    evaluation_set_sizes[evaluation_set] = (
                        evaluation_set_sizes.get(evaluation_set, 0) + 1
                    )
                evaluation_suite_version = max(
                    (int(item.get("suite_version", 1)) for item in manifest), default=1
                )
                evaluation_states = [
                    StateDescriptor(
                        self.evaluation_suite_directory / item["save_file"],
                        self.evaluation_suite_directory / item["metadata_file"],
                        item["map_num"],
                        item["player_count"],
                        item["seed"],
                        item.get("scenario"),
                        (
                            ""
                            if item.get("evaluation_set") in {"early", "mixed_development"}
                            else "immediate_finish"
                            if item.get("immediate_finish", False)
                            else "one_round_before"
                        ),
                        item.get("evaluation_set", "mid_late_end"),
                        tuple(item.get("starting_score_by_seat", ())),
                        tuple(item.get("starting_development_by_seat", ())),
                        tuple(item.get("development_role_by_seat", ())),
                        item.get("early_route_scaffold"),
                        tuple(
                            tuple(route_ids)
                            for route_ids in item.get("scaffolded_route_ids_by_seat", ())
                        ),
                        tuple(
                            tuple(lengths)
                            for lengths in item.get("scaffolded_route_lengths_by_seat", ())
                        ),
                    )
                    for item in manifest
                ]
            else:
                evaluation_states = []
                evaluation_set_sizes = {"mid_late_end": self.config.evaluation_games_per_batch}
                evaluation_suite_version = 0
                total_games = self.config.evaluation_games_per_batch
                for index in range(total_games):
                    evaluation_index = (self.batch_number - 1) * total_games + index
                    map_num, player_count = EVALUATION_CONFIGURATIONS[
                        evaluation_index % len(EVALUATION_CONFIGURATIONS)
                    ]
                    evaluation_states.append(
                        self._generate_state(
                            stage,
                            self.config.evaluation_seed + evaluation_index,
                            directory / "evaluation",
                            map_num=map_num,
                            player_count=player_count,
                        )
                    )

            total_games = len(evaluation_states)
            for index, descriptor in enumerate(evaluation_states):
                trajectory = None
                action_seed = descriptor.action_seed
                failure_reason = None
                evaluation_retry_limit = min(self.config.retry_limit, EVALUATION_RETRY_LIMIT)
                for retry_count in range(evaluation_retry_limit + 1):
                    retry = "" if retry_count == 0 else f" (retry {retry_count})"
                    self._report(f"Evaluation game {index + 1}/{total_games}{retry}...")
                    action_seed = descriptor.action_seed + retry_count
                    self.trainer.rng.seed(action_seed)
                    self.trainer.config = replace(
                        self.trainer.config,
                        max_actions=self._evaluation_action_limit(stage, descriptor),
                    )
                    try:
                        candidate = self.trainer.collect_game(
                            descriptor.path,
                            failure_callback=self._failure_callback(
                                stage, descriptor, retry_count, "evaluation"
                            ),
                            evaluation=True,
                            evaluation_tier_rotation=self.batch_number - 1,
                            capture_action_limit=descriptor.evaluation_set
                            in {"early", "mixed_development"},
                        )
                    except ActionLimitExceeded:
                        failure_reason = "action_limit"
                        incomplete += 1
                        self._report(
                            f"Evaluation game {index + 1}/{total_games}: action_limit; retrying"
                        )
                        continue
                    except IncompleteGameError:
                        failure_reason = "engine_dead_end"
                        incomplete += 1
                        self._report(
                            f"Evaluation game {index + 1}/{total_games}: engine_dead_end; retrying"
                        )
                        continue

                    candidate_reason = getattr(candidate, "completion_reason", "normal")
                    if candidate_reason == "action_limit":
                        incomplete += 1
                    if candidate_reason != "no_replacement_route":
                        trajectory = candidate
                        break
                    failure_reason = "no_replacement_route"
                    incomplete += 1
                    self._report(
                        f"Evaluation game {index + 1}/{total_games}: no_replacement_route; retrying"
                    )

                if trajectory is None:
                    self._report(
                        f"Evaluation game {index + 1}/{total_games} repeatedly failed "
                        f"with {failure_reason}; keeping the fixed state for the next batch"
                    )
                    continue
                trajectories.append(trajectory)
                evaluation_loss = self.trainer.trajectory_loss(trajectory)
                result = getattr(trajectory, "completion_reason", "normal")
                slow_timings = _play_timing_breakdown(trajectory)
                timing = f" ({slow_timings})" if slow_timings else ""
                self._report(
                    f"Evaluation game {index + 1}/{total_games}: {result}; "
                    f"{len(trajectory.action_trace)} actions; "
                    f"play {trajectory.play_seconds:.2f}s{timing}"
                )
                self.game_number += 1
                self.report_game_number += 1
                rows.append(
                    self._trajectory_row(
                        trajectory,
                        descriptor,
                        stage,
                        "evaluation",
                        retry_count,
                        evaluation_loss,
                        None,
                        self.report_game_number,
                        action_seed=action_seed,
                        evaluation_suite_size=evaluation_set_sizes.get(
                            descriptor.evaluation_set or "mid_late_end", total_games
                        ),
                        evaluation_suite_version=evaluation_suite_version,
                    )
                )
        finally:
            self.trainer.progress = progress
            self.trainer.rng.setstate(rng_state)
        return rows, trajectories, incomplete

    def _should_promote(
        self,
        training_rows,
        evaluation_rows,
        trajectories,
        unfinished,
        evaluation_unfinished=0,
    ):
        criteria = self.config.promotion
        if evaluation_unfinished or not evaluation_rows:
            return False
        if self.trainer.progress.invalid_action_attempts:
            return False
        if any(
            row.get("completion_reason") == "no_replacement_route"
            for row in (*training_rows, *evaluation_rows)
        ):
            return False
        attempts = len(training_rows) + unfinished
        if attempts and unfinished / attempts > criteria.maximum_unfinished_rate:
            return False
        if criteria.require_tier_one_advantage:
            games = {}
            wins = {}
            for trajectory in trajectories:
                for tier in trajectory.seat_tiers:
                    games[tier] = games.get(tier, 0) + 1
                for winner in trajectory.winner_indices:
                    tier = trajectory.seat_tiers[winner]
                    wins[tier] = wins.get(tier, 0) + 1
            tier_one_rate = wins.get(1, 0) / games.get(1, 1)
            lower_rates = [wins.get(tier, 0) / count for tier, count in games.items() if tier != 1]
            if lower_rates and tier_one_rate <= sum(lower_rates) / len(lower_rates):
                return False
        if self.rolling_losses:
            baseline = sum(self.rolling_losses) / len(self.rolling_losses)
            if self.trainer.progress.last_loss > baseline * (1 + criteria.loss_tolerance):
                return False
        return True

    def run(self):
        for run_batch_index in range(self.config.iterations):
            self.run_batch_number = run_batch_index + 1
            stage = self.config.stages[self.stage_index]
            stage_label = self._stage_label(stage)
            self._report(
                f"Starting stage '{stage_label}': "
                f"{self.config.training_games_per_batch} training games and "
                f"{self._evaluation_game_count()} evaluation game(s)"
            )
            self.batch_number += 1
            self.report_batch_number += 1
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            batch_directory = self.temporary_directory / f"batch_{self.batch_number}_{run_id}"
            batch_directory.mkdir(parents=True, exist_ok=False)
            self.trainer.config = replace(
                self.trainer.config,
                max_actions=self._stage_action_limit(stage),
                disable_move_action=False,
            )
            try:
                training_rows, descriptors, unfinished = self._collect_training(
                    stage, batch_directory
                )
                (
                    evaluation_rows,
                    evaluation_trajectories,
                    evaluation_unfinished,
                ) = self._collect_evaluation(stage, batch_directory)
                if self._should_promote(
                    training_rows,
                    evaluation_rows,
                    evaluation_trajectories,
                    unfinished,
                    evaluation_unfinished,
                ):
                    self.stage_index = min(self.stage_index + 1, len(self.config.stages) - 1)
                latest_loss = self.trainer.progress.last_loss
                if latest_loss is not None:
                    self.rolling_losses.append(latest_loss)
                    self.rolling_losses = self.rolling_losses[
                        -self.config.promotion.rolling_loss_window :
                    ]
                source_paths = [descriptor.path for descriptor in descriptors]
                self.trainer.save_checkpoint(
                    self.checkpoint_path,
                    source_paths,
                    curriculum_state=self._curriculum_state(),
                )
                self._append_csv(evaluation_rows)
                self._report("Saved final batch progress and evaluation result")
            except Exception as error:
                if id(error) not in self._captured_errors and not isinstance(
                    error, (ActionLimitExceeded, CurriculumRunError)
                ):
                    self._save_failure(stage, self._latest_descriptor, 0, "runner", error)
                raise
            else:
                shutil.rmtree(batch_directory)
        return self._curriculum_state()
