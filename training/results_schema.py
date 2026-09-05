"""Read-only interpretation of current and historical training-results rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


TRAINING_MATURITIES = frozenset(("fresh", "early", "mid", "late", "end"))


def maturity_from_stage_name(stage_name):
    """Map current and historical stage labels to one training maturity."""
    value = (stage_name or "").strip().lower()
    if value in TRAINING_MATURITIES:
        return value
    if value.startswith("near_end") or value.startswith("end"):
        return "end"
    if value.startswith("late"):
        return "late"
    if value.startswith("mid"):
        return "mid"
    if value.startswith("early"):
        return "early"
    if value.startswith("fresh") or value == "full_game":
        return "fresh"
    if value.startswith("near_"):
        return "end"
    return "end"


def canonical_run(row: Mapping[str, object]) -> str:
    """Return the canonical run label while accepting historical schemas."""
    current = str(row.get("run") or "").strip().lower()
    legacy_mode = str(row.get("run_mode") or "").strip().lower()
    run_type = str(row.get("run_type") or "").strip().lower()
    if current.startswith("evaluation_"):
        return current
    if legacy_mode.startswith("evaluation_"):
        return legacy_mode
    if run_type == "evaluation":
        evaluation_set = str(row.get("evaluation_set") or "mid_late_end").strip().lower()
        return f"evaluation_{evaluation_set}"

    stage = str(row.get("training_stage") or "").strip().lower()
    if stage not in TRAINING_MATURITIES and current.startswith("training_"):
        current_stage = current.removeprefix("training_").removesuffix("_zero_epsilon")
        if current_stage in TRAINING_MATURITIES:
            stage = current_stage
    if stage not in TRAINING_MATURITIES:
        curriculum_stage = str(row.get("curriculum_stage") or "").partition("+")[0]
        stage = maturity_from_stage_name(curriculum_stage)

    exploration = str(row.get("training_exploration_mode") or "").strip().lower()
    zero_epsilon = (
        current.endswith("_zero_epsilon")
        or legacy_mode.endswith("_zero_epsilon")
        or exploration == "zero_epsilon"
    )
    suffix = "_zero_epsilon" if zero_epsilon else ""
    return f"training_{stage}{suffix}"


def scenario(row: Mapping[str, object]) -> str:
    """Preserve scenario detail that is not already represented by ``run``."""
    current = str(row.get("scenario") or "").strip()
    if current:
        return current
    value = str(row.get("curriculum_stage") or "").strip()
    if not value:
        return ""
    first, separator, detail = value.partition("+")
    if first.lower() in TRAINING_MATURITIES:
        return detail if separator else ""
    if first.lower() in {"early_game", "mid_game", "late_game", "full_game"}:
        return detail if separator else ""
    return value


def _legacy_alias(row: Mapping[str, object], current: str, legacy: str):
    return row.get(current, row.get(legacy, ""))


@dataclass(frozen=True)
class ResultsRowInterpretation:
    """Canonical read-only view over one current or historical results row."""

    run: str
    scenario: str
    evaluation_set: str | None
    completion_reason: object
    total_training_decisions: object
    sampled_training_decisions: object

    @property
    def run_type(self):
        return "evaluation" if self.run.startswith("evaluation_") else "training"


def interpret_results_row(row: Mapping[str, object]) -> ResultsRowInterpretation:
    """Interpret shared results fields without mutating the supplied row."""
    run = canonical_run(row)
    evaluation_set = run.removeprefix("evaluation_") if run.startswith("evaluation_") else None
    completion_reason = row.get("completion_reason", "")
    if (
        str(row.get("run_type") or "").strip().lower() == "training_timeout"
        and not completion_reason
    ):
        completion_reason = "action_limit"
    return ResultsRowInterpretation(
        run=run,
        scenario=scenario(row),
        evaluation_set=evaluation_set,
        completion_reason=completion_reason,
        total_training_decisions=_legacy_alias(
            row, "total_training_decisions", "trajectory_decision_count"
        ),
        sampled_training_decisions=_legacy_alias(
            row, "sampled_training_decisions", "sampled_training_decision_count"
        ),
    )
