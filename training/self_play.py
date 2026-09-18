"""Shared-model training from exact Hansa starting positions."""

from __future__ import annotations

import hashlib
import io
import math
import random
import tempfile
from collections import Counter
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, field, replace
from functools import cache
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as functional

from ai.ai_model import (
    LEGACY_MODEL_CHECKPOINT_VERSION,
    MODEL_CHECKPOINT_FORMAT,
    MODEL_CHECKPOINT_VERSION,
    HansaNN,
    HansaNNOutput,
    device,
)
from ai.observation_encoder import ObservationEncoder
from ai.observation_schema import (
    LEGACY_OBSERVATION_SIZE,
    LEGACY_OBSERVATION_SIZE_V3,
    LEGACY_OBSERVATION_SIZE_V4,
    observation_schema_metadata,
    validate_model_observation_schema_metadata,
)
from game.action_codec import DEFAULT_ACTION_CODEC
from game.action_schema import (
    ACTION_SPACE_SIZE,
    action_schema_metadata,
    validate_action_schema_metadata,
)
from game.invariants import validate_game
from game.persistence import load_game
from game.structured_actions import (
    IncomeInteraction,
    PieceShape,
    PostInteraction,
    RouteInteraction,
)
from game.turn_state import TurnPhase
from map_data.constants import ACTIONS_MAX_VALUES, DARK_GREEN, UPGRADE_MAX_VALUES

TRAINING_CHECKPOINT_FORMAT = "hansa-shared-q-training"
TRAINING_CHECKPOINT_VERSION = 7
LEGACY_DUAL_HEAD_CHECKPOINT_VERSION = 6
LEGACY_Q_ONLY_CHECKPOINT_VERSION = 5
DEFAULT_LEARNING_RATE = 0.0001
TRAJECTORY_LOSS_CHUNK_SIZE = 512
LEGACY_LEARNING_RATE = 0.00001
LEGACY_EARLY_MAX_TRAINING_DECISIONS = 2_048
PRESTIGE_REWARD_MULTIPLIER = 100
END_GAME_WINNER_BONUS = 150
NO_REPLACEMENT_ROUTE_PENALTY = -500
MOVE_ROUTE_FOCUS_REWARD = 10
MOVE_BLOCK_REWARD = 25
ROUTE_COMPLETION_REWARD = 50
MOVE_COMPLETED_ROUTE_REWARD = 50
MOVE_CLAIM_COMBO_REWARD = 250
MOVE1_UTILIZATION_LOCAL_TARGET = -500
CASE_A_FAMILY_RANK_WEIGHT = 1.00
CASE_A_FAMILY_RANK_MARGIN = 1.0
MOVE_PICKUP_TRAINING_SCAFFOLD_ENABLED = True
MOVE_CONTINUATION_FAMILY_RANK_WEIGHT = CASE_A_FAMILY_RANK_WEIGHT
MOVE_CONTINUATION_FAMILY_RANK_MARGIN = 1.0
POINTLESS_ROUTE_CLAIM_PENALTY = -250
ALL_MOVE_TURN_LOCAL_TARGET = -500
ROUTE_BUILDING_PLACEMENT_REWARD = 5
ROUTE_BUILDING_DISPLACEMENT_REWARD = 3
INTERMEDIATE_ABILITY_UPGRADE_REWARD = 250
FIRST_ACTIONS_UPGRADE_REWARD = 400
INTERMEDIATE_REWARDED_ABILITIES = ("privilege", "book", "actions", "bank")
REPEATED_MOVE_LOCAL_TARGET = -1500
CONSECUTIVE_HIGH_CAPACITY_MOVE_PENALTY = -200
POINTLESS_MOVEMENT_LOCAL_TARGET = -1000
POINTLESS_FINAL_PLACEMENT_RANK_WEIGHT = 0.25
POINTLESS_FINAL_PLACEMENT_RANK_MARGIN = 1.0
MOVE_PICKUP_RANKING_DEPTH_DIAGNOSTIC_METRICS = (
    "worst_pickup_q_mean",
    "best_pickup_q_mean",
    "best_placement_q_mean",
    "selected_pickup_q_mean",
    "ordinary_target_mean",
    "selected_q_minus_ordinary_target_mean",
    "base_q_would_push_selected_up_fraction",
    "base_q_would_push_selected_down_fraction",
    "selected_pickup_is_worst_fraction",
    "ranking_hinge_active_fraction",
    "base_q_ranking_direct_opposition_fraction",
    "ordinary_reward_to_go_samples",
    "repeated_move_hard_target_samples",
    "all_move_turn_hard_target_samples",
    "additive_adjustment_target_samples",
    "other_override_target_samples",
)
_CURRICULUM_STATE_UNSET = object()
DEFAULT_TIER_TOP_K = (2, 4, 6, 8, 10)
DEFAULT_TIER_EPSILONS = (0.05, 0.10, 0.20, 0.35, 0.35)
FRESH_OPTIMIZER_UPDATES_PER_TRAJECTORY = 4
NORMAL_EXPLORATION_MODE = "normal"
ZERO_EPSILON_EXPLORATION_MODE = "zero_epsilon"
SHADOW_FILTER_POLICY_TOP_K = 10
SHADOW_FILTER_Q_TOP_K = 20
NORMAL_MOVE_CAPACITY_TELEMETRY_FIELDS = tuple(
    [f"normal_move_effective_capacity_{capacity}_moves" for capacity in range(1, 6)]
    + [
        f"normal_move_capacity_{capacity}_moved_{pieces_moved}"
        for capacity in range(1, 6)
        for pieces_moved in range(1, capacity + 1)
    ]
)
POINTLESS_MOVE_ATTRIBUTION_FIELDS = (
    "pointless_audit_all_ranked_top_k_moves",
    "pointless_audit_all_epsilon_random_moves",
    "pointless_audit_all_other_moves",
    "pointless_audit_ranked_top_k_moves",
    "pointless_audit_epsilon_random_moves",
    "pointless_audit_other_moves",
    "pointless_audit_q1_moves",
    "pointless_audit_q2_moves",
    "pointless_audit_q3_moves",
    "pointless_audit_q4_plus_moves",
    "pointless_audit_exploration_unranked_moves",
    "pointless_audit_exact_restoration_moves",
    "pointless_audit_equivalent_rearrangement_moves",
    *(f"pointless_audit_move{depth}_moves" for depth in range(1, 6)),
    *(f"pointless_audit_effective_capacity_{capacity}_moves" for capacity in range(1, 6)),
    "pointless_audit_immediate_q_undo_moves",
    "pointless_audit_repeated_penalty_moves",
    "pointless_audit_all_move_turn_moves",
    "pointless_audit_avoidable_extra_move_actions",
    "pointless_audit_consecutive_move1_moves",
)
LEGACY_TIER_TOP_K = (2, 5, 10, 15, None)
PREVIOUS_TIER_TOP_K = (2, 5, 10, 15, 20)
LEGACY_TIER_EPSILONS = (0.05, 0.10, 0.20, 0.35, 1.00)
_ACTIONS_BY_INDEX = tuple(
    None if DEFAULT_ACTION_CODEC.is_reserved(index) else DEFAULT_ACTION_CODEC.decode(index)
    for index in range(ACTION_SPACE_SIZE)
)


@cache
def inverse_sqrt_rank_weights(count):
    """Return unnormalized 1/sqrt(rank) weights for one-indexed ranks."""
    if count < 1:
        raise ValueError("Rank weight count must be positive")
    return tuple(1.0 / math.sqrt(rank) for rank in range(1, count + 1))


@cache
def normalized_rank_weights(count):
    """Return normalized inverse-square-root probabilities for ranked choices."""
    weights = inverse_sqrt_rank_weights(count)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


class TrainingRunError(RuntimeError):
    """Raised when a training game cannot safely continue."""


class IncompleteGameError(TrainingRunError):
    """Raised when a generated training game cannot reach normal completion."""


class ActionLimitExceeded(IncompleteGameError):
    """Raised when a game remains unfinished at its configured interaction limit."""


@dataclass(frozen=True)
class TrainingRosterPolicy:
    """One training roster: fixed tiers plus one uniformly selected opponent tier."""

    fixed_tiers: tuple[int, ...]
    random_tier_pool: tuple[int, ...] = ()

    @classmethod
    def from_serialized(cls, value):
        if isinstance(value, cls):
            return value
        return cls(
            fixed_tiers=tuple(value["fixed_tiers"]),
            random_tier_pool=tuple(value.get("random_tier_pool", ())),
        )


@dataclass(frozen=True)
class TierRosterConfig:
    """Own the distinct training policies and fixed evaluation rosters."""

    evaluation_three_player: tuple[int, ...] = (1, 3, 5)
    evaluation_four_player: tuple[int, ...] = (1, 2, 4, 5)
    evaluation_five_player: tuple[int, ...] = (1, 2, 3, 4, 5)
    training_three_player: TrainingRosterPolicy = field(
        default_factory=lambda: TrainingRosterPolicy((1, 2), (3, 4, 5))
    )
    training_four_player: TrainingRosterPolicy = field(
        default_factory=lambda: TrainingRosterPolicy((1, 2, 3), (4, 5))
    )
    training_five_player: TrainingRosterPolicy = field(
        default_factory=lambda: TrainingRosterPolicy((1, 2, 3, 4, 5))
    )

    @classmethod
    def from_serialized(cls, value):
        if isinstance(value, cls):
            return value
        return cls(
            evaluation_three_player=tuple(value["evaluation_three_player"]),
            evaluation_four_player=tuple(value["evaluation_four_player"]),
            evaluation_five_player=tuple(value["evaluation_five_player"]),
            training_three_player=TrainingRosterPolicy.from_serialized(
                value["training_three_player"]
            ),
            training_four_player=TrainingRosterPolicy.from_serialized(
                value["training_four_player"]
            ),
            training_five_player=TrainingRosterPolicy.from_serialized(
                value["training_five_player"]
            ),
        )

    def evaluation_rosters(self):
        return {
            3: self.evaluation_three_player,
            4: self.evaluation_four_player,
            5: self.evaluation_five_player,
        }

    def training_policies(self):
        return {
            3: self.training_three_player,
            4: self.training_four_player,
            5: self.training_five_player,
        }


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_gradient_norm: float = 1.0
    max_actions: int = 500
    disable_move_action: bool = True
    move_general_stock_threshold: int = 3
    seed: int = 124
    gamma: float = 0.99
    decision_batch_size: int = 256
    normal_max_training_decisions: int = 1_024
    fresh_max_training_decisions: int = 4_096
    early_max_training_decisions: int = 4_096
    full_validation_interval: int = 50
    detailed_profiling: bool = False
    shadow_filter_audit_enabled: bool = False
    pointless_move_attribution_audit_enabled: bool = True
    income_penalty_scale: float = 100.0
    policy_loss_weight: float = 1.0
    policy_head_lr_multiplier: float = 1.0
    policy_return_scale: float = 1_000.0
    tier_top_k: tuple[int | None, ...] = DEFAULT_TIER_TOP_K
    tier_epsilons: tuple[float, ...] = DEFAULT_TIER_EPSILONS
    tier_rosters: TierRosterConfig = field(default_factory=TierRosterConfig)

    def __post_init__(self):
        if not isinstance(self.tier_rosters, TierRosterConfig):
            object.__setattr__(
                self,
                "tier_rosters",
                TierRosterConfig.from_serialized(self.tier_rosters),
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_gradient_norm <= 0:
            raise ValueError("maximum gradient norm must be positive")
        if self.max_actions < 1:
            raise ValueError("max_actions must be positive")
        if self.move_general_stock_threshold < 0:
            raise ValueError("move general-stock threshold cannot be negative")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.decision_batch_size < 1:
            raise ValueError("decision batch size must be positive")
        if self.normal_max_training_decisions < 1:
            raise ValueError("normal maximum training decisions must be positive")
        if self.fresh_max_training_decisions < 1:
            raise ValueError("fresh maximum training decisions must be positive")
        if self.early_max_training_decisions < 1:
            raise ValueError("early maximum training decisions must be positive")
        if self.full_validation_interval < 1:
            raise ValueError("full validation interval must be positive")
        if self.income_penalty_scale < 0:
            raise ValueError("income penalty scale cannot be negative")
        if self.policy_loss_weight < 0:
            raise ValueError("policy loss weight cannot be negative")
        if self.policy_head_lr_multiplier <= 0:
            raise ValueError("policy-head learning-rate multiplier must be positive")
        if self.policy_return_scale <= 0:
            raise ValueError("policy return scale must be positive")
        if len(self.tier_top_k) != len(self.tier_epsilons):
            raise ValueError("tier top-k and epsilon settings must have equal lengths")
        if any(top_k is not None and top_k < 1 for top_k in self.tier_top_k):
            raise ValueError("tier top-k values must be positive")
        if any(not 0 <= epsilon <= 1 for epsilon in self.tier_epsilons):
            raise ValueError("tier epsilon values must be between 0 and 1")
        for player_count, tiers in self.tier_rosters.evaluation_rosters().items():
            if len(tiers) != player_count or len(set(tiers)) != player_count:
                raise ValueError(f"{player_count}-player evaluation tiers must be unique")
            if any(tier < 1 or tier > len(self.tier_top_k) for tier in tiers):
                raise ValueError(f"{player_count}-player evaluation tier is undefined")
        for player_count, policy in self.tier_rosters.training_policies().items():
            selected_count = len(policy.fixed_tiers) + bool(policy.random_tier_pool)
            if selected_count != player_count:
                raise ValueError(
                    f"{player_count}-player training policy must select {player_count} tiers"
                )
            all_tiers = policy.fixed_tiers + policy.random_tier_pool
            if len(set(policy.fixed_tiers)) != len(policy.fixed_tiers) or set(
                policy.fixed_tiers
            ) & set(policy.random_tier_pool):
                raise ValueError(f"{player_count}-player training tiers must be unique")
            if len(set(policy.random_tier_pool)) != len(policy.random_tier_pool):
                raise ValueError(f"{player_count}-player random tier pool must be unique")
            if any(tier < 1 or tier > len(self.tier_top_k) for tier in all_tiers):
                raise ValueError(f"{player_count}-player training tier is undefined")


@dataclass(frozen=True)
class PolicyTier:
    number: int
    top_k: int | None
    epsilon: float


@dataclass(frozen=True)
class ActionSelection:
    action_index: int
    used_epsilon: bool
    model_rank: int
    legal_action_count: int
    equivalent_action_indices: tuple[int, ...] = ()
    semantic_q_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class TrainingDecision:
    observation: torch.Tensor
    legal_action_mask: torch.Tensor
    action_index: int
    acting_player_index: int
    player_reward_deltas: tuple[float, ...]
    immediate_reward: float
    policy_tier: int
    epsilon: float
    top_k: int | None
    used_epsilon: bool
    model_rank: int
    legal_action_count: int
    game_turn_number: int = 0
    movement_workflow_id: int | None = None
    reward_to_go: float | None = None
    local_training_target: float | None = None
    local_training_adjustment: float = 0.0
    equivalent_action_indices: tuple[int, ...] = ()
    equivalent_action_groups: tuple[tuple[int, ...], ...] = ()
    receives_terminal_credit: bool = True
    move1_utilization_penalty_role: str | None = None
    case_a_pickup_action_groups: tuple[tuple[int, ...], ...] = ()
    case_a_placement_action_groups: tuple[tuple[int, ...], ...] = ()
    move_continuation_pickup_action_groups: tuple[tuple[int, ...], ...] = ()
    move_continuation_placement_action_groups: tuple[tuple[int, ...], ...] = ()
    move_continuation_pickup_depth: int = 0
    pointless_final_restorative_action_group: tuple[int, ...] = ()
    pointless_final_nonpointless_action_groups: tuple[tuple[int, ...], ...] = ()
    pointless_final_placement_type: str | None = None
    pointless_final_placement_piece_count: int = 0


@dataclass(frozen=True)
class ShadowFilterAuditRecord:
    decision_index: int
    action_index: int
    action_type: str
    semantic_action_indices: tuple[int, ...]
    semantic_q_rank: int
    q_value: float
    q_gap_from_best: float
    semantic_policy_rank: int
    policy_probability: float
    immediate_reward: float
    local_training_target: float | None
    local_training_adjustment: float
    reward_to_go: float
    final_training_target: float
    receives_terminal_credit: bool
    terminal_credit_value: float
    acting_player_index: int
    acting_player_final_score: int
    acting_player_won: bool
    policy_tier: int
    used_epsilon: bool


@dataclass(frozen=True)
class CompletedTrajectory:
    decisions: tuple[TrainingDecision, ...]
    terminal_rewards: tuple[float, ...]
    final_scores: tuple[int, ...]
    winner_indices: tuple[int, ...]
    action_trace: tuple[int, ...]
    seat_tiers: tuple[int, ...]
    completion_reason: str = "normal"
    play_seconds: float = 0.0
    inference_seconds: float = 0.0
    scoring_seconds: float = 0.0
    execution_seconds: float = 0.0
    validation_seconds: float = 0.0
    observation_seconds: float = 0.0
    legality_seconds: float = 0.0
    selection_seconds: float = 0.0
    context_seconds: float = 0.0
    reward_seconds: float = 0.0
    move_action_count: int = 0
    spent_action_count: int = 0
    move_ratio: float | None = None
    pointless_move_workflows: int = 0
    repeated_move_penalties: int = 0
    all_move_turn_penalties: int = 0
    moves_creating_claimable_route: int = 0
    move_claim_conversions: int = 0
    move_claim_conversion_rate: float | None = None
    training_exploration_mode: str = NORMAL_EXPLORATION_MODE
    policy_q_top1_agreement: float | None = None
    policy_top1_q_rank: float | None = None
    policy_entropy: float | None = None
    policy_top2_mass: float | None = None
    policy_top5_mass: float | None = None
    policy_top10_mass: float | None = None
    shadow_filter_records: tuple[ShadowFilterAuditRecord, ...] = ()
    shadow_filter_selected_count: int = 0
    shadow_filter_epsilon_selected_count: int = 0
    pointless_normal_move_workflows: int = 0
    pointless_move_any2_workflows: int = 0
    immediate_one_piece_q_undos: int = 0
    immediate_q_undo_epsilon_pickups: int = 0
    immediate_q_undo_ranked_pickups: int = 0
    immediate_q_undo_q1_pickups: int = 0
    immediate_q1_restores: int = 0
    immediate_q1_pickup_q1_restores: int = 0
    immediate_valuable_q1_pickup_q1_restores: int = 0
    full_multi_piece_q_undos: int = 0
    full_multi_piece_q_undo_any_exploration: int = 0
    full_multi_piece_q_undo_entirely_ranked: int = 0
    normal_move_nominal_capacity_total: int = 0
    normal_move_movable_pieces_available_total: int = 0
    normal_move_effective_capacity_total: int = 0
    normal_move_pieces_moved_total: int = 0
    normal_move_unused_capacity_total: int = 0
    single_piece_moves: int = 0
    single_piece_moves_with_multiple_available: int = 0
    single_piece_moves_creating_claimable_route: int = 0
    single_piece_move_claim_conversions: int = 0
    full_effective_capacity_moves: int = 0
    under_effective_capacity_moves: int = 0
    move1_utilization_penalties_applied: int = 0
    move1_penalties_on_placement: int = 0
    move1_penalties_on_single_available_initiation: int = 0
    move_claim_reward_awarded: int = 0
    move_claim_reward_blocked_already_claimable: int = 0
    consecutive_move1_pairs: int = 0
    consecutive_move1_pairs_with_multiple_available: int = 0
    consecutive_move1_pairs_move2_capacity: int = 0
    consecutive_move1_pairs_move3_capacity: int = 0
    consecutive_move1_pairs_move4_capacity: int = 0
    consecutive_move1_pairs_move5_capacity: int = 0
    avoidable_extra_move_actions: int = 0
    move1_scaffold_mask_states: int = 0
    move1_scaffold_masked_placement_semantic_actions: int = 0
    move1_scaffold_legal_pickup_semantic_actions: int = 0
    move1_scaffold_unmasked_q1_pickups: int = 0
    move1_scaffold_unmasked_q1_placements: int = 0
    move1_scaffold_unmasked_q1_pickup_fraction: float | None = None
    move1_scaffold_unmasked_q1_placement_fraction: float | None = None
    move1_scaffold_all_pickups_above_all_placements_fraction: float | None = None
    move1_scaffold_margin_satisfied_pair_fraction: float | None = None
    move1_scaffold_family_ranking_loss: float | None = None
    move1_scaffold_best_pickup_minus_best_placement_mean: float | None = None
    move1_scaffold_best_pickup_minus_best_placement_median: float | None = None
    move1_scaffold_best_pickup_minus_best_placement_p10: float | None = None
    move1_scaffold_best_pickup_minus_best_placement_p90: float | None = None
    move1_scaffold_unmasked_top_k_pickup_fraction: float | None = None
    move1_scaffold_unmasked_top_k_has_pickup_fraction: float | None = None
    normal_move_effective_capacity_1_moves: int = 0
    normal_move_effective_capacity_2_moves: int = 0
    normal_move_effective_capacity_3_moves: int = 0
    normal_move_effective_capacity_4_moves: int = 0
    normal_move_effective_capacity_5_moves: int = 0
    normal_move_capacity_1_moved_1: int = 0
    normal_move_capacity_2_moved_1: int = 0
    normal_move_capacity_2_moved_2: int = 0
    normal_move_capacity_3_moved_1: int = 0
    normal_move_capacity_3_moved_2: int = 0
    normal_move_capacity_3_moved_3: int = 0
    normal_move_capacity_4_moved_1: int = 0
    normal_move_capacity_4_moved_2: int = 0
    normal_move_capacity_4_moved_3: int = 0
    normal_move_capacity_4_moved_4: int = 0
    normal_move_capacity_5_moved_1: int = 0
    normal_move_capacity_5_moved_2: int = 0
    normal_move_capacity_5_moved_3: int = 0
    normal_move_capacity_5_moved_4: int = 0
    normal_move_capacity_5_moved_5: int = 0
    pointless_move_attribution: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSampleCoverage:
    """Decision coverage used by the most recent trajectory update."""

    total_decisions: int
    sampled_decisions: int
    sampled_octiles: tuple[int, ...] = ()

    @property
    def sampled_fraction(self):
        if not self.total_decisions:
            return None
        return self.sampled_decisions / self.total_decisions


@dataclass(frozen=True)
class CaseAFamilyRankingResult:
    """Differentiable loss plus detached-count inputs for Case-A telemetry."""

    loss: torch.Tensor
    sample_count: int
    per_state_losses: torch.Tensor
    violating_sample_count: torch.Tensor
    violating_pair_count: torch.Tensor
    pair_count: torch.Tensor
    violating_pair_fraction_sum: torch.Tensor
    all_pickups_above_all_placements_count: torch.Tensor
    q1_pickup_count: torch.Tensor
    worst_pickup_q_sum: torch.Tensor
    best_placement_q_sum: torch.Tensor
    worst_pickup_minus_best_placement_gap_sum: torch.Tensor
    margin_satisfied_count: torch.Tensor
    depth_diagnostic_sums: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class MoveContinuationFamilyRankingResult:
    """Strict worst-pickup boundary loss and telemetry for pickup-depth 2+ states."""

    loss: torch.Tensor
    sample_count: int
    per_state_losses: torch.Tensor
    violating_sample_count: torch.Tensor
    best_pickup_above_all_placements_count: torch.Tensor
    q1_pickup_count: torch.Tensor
    depth_2_count: int
    depth_3_count: int
    depth_4_count: int
    all_pickups_above_all_placements_count: torch.Tensor
    margin_satisfied_count: torch.Tensor
    worst_pickup_q_sum: torch.Tensor
    best_placement_q_sum: torch.Tensor
    worst_pickup_minus_best_placement_gap_sum: torch.Tensor
    depth_loss_sums: tuple[torch.Tensor, ...]
    depth_q1_pickup_counts: tuple[torch.Tensor, ...]
    depth_all_pickups_above_all_placements_counts: tuple[torch.Tensor, ...]
    depth_margin_satisfied_counts: tuple[torch.Tensor, ...]
    depth_gap_sums: tuple[torch.Tensor, ...]
    depth_diagnostic_sums: tuple[tuple[torch.Tensor, ...], ...] = ()


@dataclass(frozen=True)
class MovePickupRankingLossResult:
    """Pooled Move pickup-ranking loss and each depth's sample-weighted contribution."""

    loss: torch.Tensor
    present_depth_count: int
    depth_contributions: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PointlessFinalPlacementRankingResult:
    """Differentiable final-restoration ranking loss and detached telemetry."""

    loss: torch.Tensor
    sample_count: int
    per_state_losses: torch.Tensor
    violating_sample_count: torch.Tensor
    margin_satisfied_count: torch.Tensor
    q_gap_sum: torch.Tensor
    restorative_q1_count: int
    immediate_q_undo_count: int
    multi_piece_count: int
    exact_restoration_count: int
    equivalent_rearrangement_count: int


@dataclass
class PointlessMoveAttributionAudit:
    """Aggregate-only attribution for completed normal paid Move workflows."""

    counters: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(POINTLESS_MOVE_ATTRIBUTION_FIELDS, 0)
    )
    pointless_workflow_ids: set[int] = field(default_factory=set)
    all_move_turn_workflow_ids: set[int] = field(default_factory=set)
    consecutive_move1_workflow_ids: set[int] = field(default_factory=set)

    @staticmethod
    def _selection_source(selection):
        if selection is None or selection.role != "initial_pickup":
            return "other"
        if selection.used_epsilon:
            return "epsilon_random"
        return "ranked_top_k"

    def record_completed_move(
        self,
        *,
        workflow_id,
        initiation_selection,
        pointless_type,
        pieces_moved,
        effective_capacity,
        immediate_q_undo,
        repeated_penalty,
        avoidable_extra_move,
    ):
        source = self._selection_source(initiation_selection)
        self.counters[f"pointless_audit_all_{source}_moves"] += 1
        if pointless_type is None:
            return

        self.pointless_workflow_ids.add(workflow_id)
        self.counters[f"pointless_audit_{source}_moves"] += 1
        if source == "ranked_top_k":
            rank = initiation_selection.model_rank
            rank_bucket = str(rank) if rank <= 3 else "4_plus"
            self.counters[f"pointless_audit_q{rank_bucket}_moves"] += 1
        else:
            self.counters["pointless_audit_exploration_unranked_moves"] += 1
        self.counters[f"pointless_audit_{pointless_type}_moves"] += 1
        if 1 <= pieces_moved <= 5:
            self.counters[f"pointless_audit_move{pieces_moved}_moves"] += 1
        if 1 <= effective_capacity <= 5:
            self.counters[f"pointless_audit_effective_capacity_{effective_capacity}_moves"] += 1
        self.counters["pointless_audit_immediate_q_undo_moves"] += int(immediate_q_undo)
        self.counters["pointless_audit_repeated_penalty_moves"] += int(repeated_penalty)
        self.counters["pointless_audit_avoidable_extra_move_actions"] += int(avoidable_extra_move)

    def mark_all_move_turn(self, workflow_ids):
        self.all_move_turn_workflow_ids.update(
            self.pointless_workflow_ids.intersection(workflow_ids)
        )

    def mark_consecutive_move1(self, workflow_ids):
        self.consecutive_move1_workflow_ids.update(
            self.pointless_workflow_ids.intersection(workflow_ids)
        )

    def as_dict(self):
        counters = dict(self.counters)
        counters["pointless_audit_all_move_turn_moves"] = len(self.all_move_turn_workflow_ids)
        counters["pointless_audit_consecutive_move1_moves"] = len(
            self.consecutive_move1_workflow_ids
        )
        return counters


@dataclass
class MovementBehaviorMetrics:
    """Per-game counters recorded at existing movement reward/penalty boundaries."""

    move_action_count: int = 0
    spent_action_count: int = 0
    pointless_move_workflows: int = 0
    pointless_normal_move_workflows: int = 0
    pointless_move_any2_workflows: int = 0
    repeated_move_penalties: int = 0
    all_move_turn_penalties: int = 0
    moves_creating_claimable_route: int = 0
    move_claim_conversions: int = 0
    immediate_one_piece_q_undos: int = 0
    immediate_q_undo_epsilon_pickups: int = 0
    immediate_q_undo_ranked_pickups: int = 0
    immediate_q_undo_q1_pickups: int = 0
    immediate_q1_restores: int = 0
    immediate_q1_pickup_q1_restores: int = 0
    immediate_valuable_q1_pickup_q1_restores: int = 0
    full_multi_piece_q_undos: int = 0
    full_multi_piece_q_undo_any_exploration: int = 0
    full_multi_piece_q_undo_entirely_ranked: int = 0
    normal_move_nominal_capacity_total: int = 0
    normal_move_movable_pieces_available_total: int = 0
    normal_move_effective_capacity_total: int = 0
    normal_move_pieces_moved_total: int = 0
    normal_move_unused_capacity_total: int = 0
    single_piece_moves: int = 0
    single_piece_moves_with_multiple_available: int = 0
    single_piece_moves_creating_claimable_route: int = 0
    single_piece_move_claim_conversions: int = 0
    full_effective_capacity_moves: int = 0
    under_effective_capacity_moves: int = 0
    move1_utilization_penalties_applied: int = 0
    move1_penalties_on_placement: int = 0
    move1_penalties_on_single_available_initiation: int = 0
    move_claim_reward_awarded: int = 0
    move_claim_reward_blocked_already_claimable: int = 0
    consecutive_move1_pairs: int = 0
    consecutive_move1_pairs_with_multiple_available: int = 0
    consecutive_move1_pairs_move2_capacity: int = 0
    consecutive_move1_pairs_move3_capacity: int = 0
    consecutive_move1_pairs_move4_capacity: int = 0
    consecutive_move1_pairs_move5_capacity: int = 0
    avoidable_extra_move_actions: int = 0
    move1_scaffold_mask_states: int = 0
    move1_scaffold_masked_placement_semantic_actions: int = 0
    move1_scaffold_legal_pickup_semantic_actions: int = 0
    move1_scaffold_unmasked_q1_pickups: int = 0
    move1_scaffold_unmasked_q1_placements: int = 0
    move1_scaffold_all_pickups_above_all_placements: int = 0
    move1_scaffold_margin_satisfied_pair_fraction_total: float = 0.0
    move1_scaffold_family_ranking_loss_total: float = 0.0
    move1_scaffold_best_pickup_minus_best_placement: list[float] = field(default_factory=list)
    move1_scaffold_unmasked_top_k_pickup_fraction_total: float = 0.0
    move1_scaffold_unmasked_top_k_has_pickup: int = 0
    normal_move_effective_capacity_1_moves: int = 0
    normal_move_effective_capacity_2_moves: int = 0
    normal_move_effective_capacity_3_moves: int = 0
    normal_move_effective_capacity_4_moves: int = 0
    normal_move_effective_capacity_5_moves: int = 0
    normal_move_capacity_1_moved_1: int = 0
    normal_move_capacity_2_moved_1: int = 0
    normal_move_capacity_2_moved_2: int = 0
    normal_move_capacity_3_moved_1: int = 0
    normal_move_capacity_3_moved_2: int = 0
    normal_move_capacity_3_moved_3: int = 0
    normal_move_capacity_4_moved_1: int = 0
    normal_move_capacity_4_moved_2: int = 0
    normal_move_capacity_4_moved_3: int = 0
    normal_move_capacity_4_moved_4: int = 0
    normal_move_capacity_5_moved_1: int = 0
    normal_move_capacity_5_moved_2: int = 0
    normal_move_capacity_5_moved_3: int = 0
    normal_move_capacity_5_moved_4: int = 0
    normal_move_capacity_5_moved_5: int = 0
    pointless_move_attribution: PointlessMoveAttributionAudit = field(
        default_factory=PointlessMoveAttributionAudit
    )

    @property
    def move_ratio(self):
        if not self.spent_action_count:
            return None
        return self.move_action_count / self.spent_action_count

    @property
    def move_claim_conversion_rate(self):
        if not self.moves_creating_claimable_route:
            return None
        return self.move_claim_conversions / self.moves_creating_claimable_route

    @property
    def move1_scaffold_unmasked_q1_pickup_fraction(self):
        if not self.move1_scaffold_mask_states:
            return None
        return self.move1_scaffold_unmasked_q1_pickups / self.move1_scaffold_mask_states

    @property
    def move1_scaffold_unmasked_q1_placement_fraction(self):
        if not self.move1_scaffold_mask_states:
            return None
        return self.move1_scaffold_unmasked_q1_placements / self.move1_scaffold_mask_states

    @property
    def move1_scaffold_all_pickups_above_all_placements_fraction(self):
        if not self.move1_scaffold_mask_states:
            return None
        return (
            self.move1_scaffold_all_pickups_above_all_placements / self.move1_scaffold_mask_states
        )

    @property
    def move1_scaffold_margin_satisfied_pair_fraction(self):
        if not self.move1_scaffold_mask_states:
            return None
        return (
            self.move1_scaffold_margin_satisfied_pair_fraction_total
            / self.move1_scaffold_mask_states
        )

    @property
    def move1_scaffold_family_ranking_loss(self):
        if not self.move1_scaffold_mask_states:
            return None
        return self.move1_scaffold_family_ranking_loss_total / self.move1_scaffold_mask_states

    @property
    def move1_scaffold_unmasked_top_k_pickup_fraction(self):
        if not self.move1_scaffold_mask_states:
            return None
        return (
            self.move1_scaffold_unmasked_top_k_pickup_fraction_total
            / self.move1_scaffold_mask_states
        )

    @property
    def move1_scaffold_unmasked_top_k_has_pickup_fraction(self):
        if not self.move1_scaffold_mask_states:
            return None
        return self.move1_scaffold_unmasked_top_k_has_pickup / self.move1_scaffold_mask_states


def record_pointless_movement_workflow(
    metrics,
    penalty,
    *,
    normal_move_completed=False,
    permanent_move_any2_completed=False,
):
    """Count one detected no-op in both combined and workflow-specific telemetry."""
    if not penalty:
        return
    metrics.pointless_move_workflows += 1
    if normal_move_completed:
        metrics.pointless_normal_move_workflows += 1
    elif permanent_move_any2_completed:
        metrics.pointless_move_any2_workflows += 1


def record_move_capacity_utilization(metrics, nominal_capacity, available_pieces, pieces_moved):
    """Record completed normal-Move capacity without judging partial utilization."""
    effective_capacity = min(nominal_capacity, available_pieces)
    unused_capacity = max(effective_capacity - pieces_moved, 0)
    metrics.normal_move_nominal_capacity_total += nominal_capacity
    metrics.normal_move_movable_pieces_available_total += available_pieces
    metrics.normal_move_effective_capacity_total += effective_capacity
    metrics.normal_move_pieces_moved_total += pieces_moved
    metrics.normal_move_unused_capacity_total += unused_capacity
    metrics.single_piece_moves += int(pieces_moved == 1)
    metrics.single_piece_moves_with_multiple_available += int(
        pieces_moved == 1 and effective_capacity >= 2
    )
    metrics.full_effective_capacity_moves += int(pieces_moved == effective_capacity)
    metrics.under_effective_capacity_moves += int(pieces_moved < effective_capacity)
    if 1 <= effective_capacity <= 5:
        capacity_field = f"normal_move_effective_capacity_{effective_capacity}_moves"
        setattr(metrics, capacity_field, getattr(metrics, capacity_field) + 1)
        if 1 <= pieces_moved <= effective_capacity:
            depth_field = f"normal_move_capacity_{effective_capacity}_moved_{pieces_moved}"
            setattr(metrics, depth_field, getattr(metrics, depth_field) + 1)
    return effective_capacity


@dataclass(frozen=True)
class Move1CompletionTelemetry:
    """Minimal state needed to compare adjacent paid Move1 actions."""

    nominal_capacity: int
    initial_pickup_post_slot: int
    additional_pickup_post_slots: frozenset[int]


def update_consecutive_move1_telemetry(
    metrics,
    previous_move1,
    *,
    normal_move_completed,
    pieces_moved,
    nominal_capacity,
    initial_pickup_post_slot,
    additional_pickup_post_slots,
):
    """Record adjacent paid Move1 actions and return the new pending Move1."""
    if not normal_move_completed or pieces_moved != 1:
        return None
    current_move1 = Move1CompletionTelemetry(
        nominal_capacity,
        initial_pickup_post_slot,
        frozenset(additional_pickup_post_slots),
    )
    if previous_move1 is None:
        return current_move1

    metrics.consecutive_move1_pairs += 1
    if previous_move1.additional_pickup_post_slots:
        metrics.consecutive_move1_pairs_with_multiple_available += 1
    capacity_field = {
        2: "consecutive_move1_pairs_move2_capacity",
        3: "consecutive_move1_pairs_move3_capacity",
        4: "consecutive_move1_pairs_move4_capacity",
        5: "consecutive_move1_pairs_move5_capacity",
    }.get(previous_move1.nominal_capacity)
    if capacity_field is not None:
        setattr(metrics, capacity_field, getattr(metrics, capacity_field) + 1)
    if current_move1.initial_pickup_post_slot in previous_move1.additional_pickup_post_slots:
        metrics.avoidable_extra_move_actions += 1
    return current_move1


@dataclass(frozen=True)
class MoveSelectionTelemetry:
    """Selection metadata retained only until one normal Move workflow completes."""

    role: str
    used_epsilon: bool
    model_rank: int


@dataclass(frozen=True)
class LoadedNormalMoveContext:
    """Reconstructed tracking for a trajectory staged during a normal Move."""

    origin_posts: tuple
    origin_pieces: tuple
    completed_route_slots_before: frozenset[int]
    observed_pickup_route_slots: frozenset[int]
    already_claimable_pickup_route_slots: frozenset[int]
    movable_pieces_at_start: int
    initial_pickup_post_slot: int


def is_immediate_one_piece_q_undo(penalty, selections):
    """Return whether a detected no-op is one pickup immediately restored once."""
    return bool(
        penalty
        and len(selections) == 2
        and selections[0].role == "initial_pickup"
        and selections[1].role == "final_placement"
    )


def record_q_undo_workflow(
    metrics,
    penalty,
    selections,
    *,
    valuable_origin=False,
):
    """Classify exact/semantic full undos without changing their reward treatment."""
    if not penalty or not selections:
        return
    pickups = tuple(
        selection
        for selection in selections
        if selection.role in {"initial_pickup", "additional_pickup"}
    )
    placements = tuple(
        selection
        for selection in selections
        if selection.role in {"intermediate_placement", "final_placement"}
    )
    if len(pickups) != len(placements):
        return

    immediate = is_immediate_one_piece_q_undo(penalty, selections) and len(pickups) == 1
    if immediate:
        pickup, restore = selections
        metrics.immediate_one_piece_q_undos += 1
        if pickup.used_epsilon:
            metrics.immediate_q_undo_epsilon_pickups += 1
        else:
            metrics.immediate_q_undo_ranked_pickups += 1
        if pickup.model_rank == 1:
            metrics.immediate_q_undo_q1_pickups += 1
        if restore.model_rank == 1:
            metrics.immediate_q1_restores += 1
        if pickup.model_rank == restore.model_rank == 1:
            metrics.immediate_q1_pickup_q1_restores += 1
            if valuable_origin:
                metrics.immediate_valuable_q1_pickup_q1_restores += 1
        return

    if len(pickups) > 1:
        metrics.full_multi_piece_q_undos += 1
        if any(selection.used_epsilon for selection in selections):
            metrics.full_multi_piece_q_undo_any_exploration += 1
        else:
            metrics.full_multi_piece_q_undo_entirely_ranked += 1


def should_fully_validate(action_count, interval, turn_before, phase_before, game):
    """Validate periodically and whenever a turn or staged workflow completes."""
    return (
        action_count % interval == 0
        or game.turn_number != turn_before
        or (phase_before is not TurnPhase.ACTIONS and game.turn_phase is TurnPhase.ACTIONS)
        or game.game_end
    )


@dataclass
class TrainingProgress:
    completed_games: int = 0
    training_updates: int = 0
    policy_training_updates: int = 0
    decisions: int = 0
    invalid_action_attempts: int = 0
    game_completion_failures: int = 0
    replacement_route_deadlocks: int = 0
    checkpoint_saves: int = 0
    checkpoint_loads: int = 0
    last_loss: float | None = None
    mean_loss: float | None = None
    last_q_loss: float | None = None
    last_policy_loss: float | None = None
    last_total_loss: float | None = None
    last_case_a_family_ranking_samples: int = 0
    last_case_a_family_ranking_loss: float | None = None
    last_case_a_family_ranking_violating_samples: int = 0
    last_case_a_family_ranking_violation_fraction: float | None = None
    last_case_a_family_ranking_mean_violating_placements: float | None = None
    last_case_a_family_ranking_mean_violating_pairs: float | None = None
    last_case_a_family_ranking_mean_violating_pair_fraction: float | None = None
    last_case_a_family_ranking_all_pickups_above_all_placements_fraction: float | None = None
    last_case_a_family_ranking_q1_pickup_fraction: float | None = None
    last_case_a_family_ranking_worst_pickup_q_mean: float | None = None
    last_case_a_family_ranking_best_placement_q_mean: float | None = None
    last_case_a_family_ranking_worst_pickup_minus_best_placement_gap_mean: float | None = None
    last_case_a_family_ranking_margin_satisfied_fraction: float | None = None
    last_move_continuation_family_ranking_samples: int = 0
    last_move_continuation_family_ranking_after_2_pickups: int = 0
    last_move_continuation_family_ranking_after_3_pickups: int = 0
    last_move_continuation_family_ranking_after_4_pickups: int = 0
    last_move_continuation_family_ranking_loss: float | None = None
    last_move_continuation_family_ranking_violating_samples: int = 0
    last_move_continuation_family_ranking_violation_fraction: float | None = None
    last_move_continuation_best_pickup_above_all_placements_fraction: float | None = None
    last_move_continuation_q1_pickup_fraction: float | None = None
    last_move_pickup_ranking_samples: int = 0
    last_move_pickup_ranking_loss: float | None = None
    last_move_pickup_ranking_eligible_effective_batches: int = 0
    last_move_pickup_ranking_mean_present_depth_count: float | None = None
    last_base_q_included_samples: int = 0
    last_base_q_excluded_samples: int = 0
    last_move_continuation_base_q_excluded_samples: int = 0
    last_move_continuation_base_q_excluded_h2: int = 0
    last_move_continuation_base_q_excluded_h3: int = 0
    last_move_continuation_base_q_excluded_h4: int = 0
    last_move_pickup_ranking_q1_pickup_fraction: float | None = None
    last_move_pickup_ranking_all_pickups_above_all_placements_fraction: float | None = None
    last_move_pickup_ranking_margin_satisfied_fraction: float | None = None
    last_move_pickup_ranking_worst_pickup_minus_best_placement_gap_mean: float | None = None
    last_move_pickup_ranking_holding_1_samples: int = 0
    last_move_pickup_ranking_holding_1_loss: float | None = None
    last_move_pickup_ranking_holding_1_q1_pickup_fraction: float | None = None
    last_move_pickup_ranking_holding_1_all_pickups_above_all_placements_fraction: float | None = (
        None
    )
    last_move_pickup_ranking_holding_1_margin_satisfied_fraction: float | None = None
    last_move_pickup_ranking_holding_1_worst_pickup_minus_best_placement_gap_mean: float | None = (
        None
    )
    last_move_pickup_ranking_holding_1_weighted_contribution: float | None = None
    last_move_pickup_ranking_holding_1_worst_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_1_best_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_1_best_placement_q_mean: float | None = None
    last_move_pickup_ranking_holding_1_selected_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_1_ordinary_target_mean: float | None = None
    last_move_pickup_ranking_holding_1_selected_q_minus_ordinary_target_mean: float | None = None
    last_move_pickup_ranking_holding_1_base_q_would_push_selected_up_fraction: float | None = None
    last_move_pickup_ranking_holding_1_base_q_would_push_selected_down_fraction: float | None = None
    last_move_pickup_ranking_holding_1_selected_pickup_is_worst_fraction: float | None = None
    last_move_pickup_ranking_holding_1_ranking_hinge_active_fraction: float | None = None
    last_move_pickup_ranking_holding_1_base_q_ranking_direct_opposition_fraction: float | None = (
        None
    )
    last_move_pickup_ranking_holding_1_ordinary_reward_to_go_samples: int = 0
    last_move_pickup_ranking_holding_1_repeated_move_hard_target_samples: int = 0
    last_move_pickup_ranking_holding_1_all_move_turn_hard_target_samples: int = 0
    last_move_pickup_ranking_holding_1_additive_adjustment_target_samples: int = 0
    last_move_pickup_ranking_holding_1_other_override_target_samples: int = 0
    last_move_pickup_ranking_holding_2_samples: int = 0
    last_move_pickup_ranking_holding_2_loss: float | None = None
    last_move_pickup_ranking_holding_2_q1_pickup_fraction: float | None = None
    last_move_pickup_ranking_holding_2_all_pickups_above_all_placements_fraction: float | None = (
        None
    )
    last_move_pickup_ranking_holding_2_margin_satisfied_fraction: float | None = None
    last_move_pickup_ranking_holding_2_worst_pickup_minus_best_placement_gap_mean: float | None = (
        None
    )
    last_move_pickup_ranking_holding_2_weighted_contribution: float | None = None
    last_move_pickup_ranking_holding_2_worst_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_2_best_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_2_best_placement_q_mean: float | None = None
    last_move_pickup_ranking_holding_2_selected_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_2_ordinary_target_mean: float | None = None
    last_move_pickup_ranking_holding_2_selected_q_minus_ordinary_target_mean: float | None = None
    last_move_pickup_ranking_holding_2_base_q_would_push_selected_up_fraction: float | None = None
    last_move_pickup_ranking_holding_2_base_q_would_push_selected_down_fraction: float | None = None
    last_move_pickup_ranking_holding_2_selected_pickup_is_worst_fraction: float | None = None
    last_move_pickup_ranking_holding_2_ranking_hinge_active_fraction: float | None = None
    last_move_pickup_ranking_holding_2_base_q_ranking_direct_opposition_fraction: float | None = (
        None
    )
    last_move_pickup_ranking_holding_2_ordinary_reward_to_go_samples: int = 0
    last_move_pickup_ranking_holding_2_repeated_move_hard_target_samples: int = 0
    last_move_pickup_ranking_holding_2_all_move_turn_hard_target_samples: int = 0
    last_move_pickup_ranking_holding_2_additive_adjustment_target_samples: int = 0
    last_move_pickup_ranking_holding_2_other_override_target_samples: int = 0
    last_move_pickup_ranking_holding_3_samples: int = 0
    last_move_pickup_ranking_holding_3_loss: float | None = None
    last_move_pickup_ranking_holding_3_q1_pickup_fraction: float | None = None
    last_move_pickup_ranking_holding_3_all_pickups_above_all_placements_fraction: float | None = (
        None
    )
    last_move_pickup_ranking_holding_3_margin_satisfied_fraction: float | None = None
    last_move_pickup_ranking_holding_3_worst_pickup_minus_best_placement_gap_mean: float | None = (
        None
    )
    last_move_pickup_ranking_holding_3_weighted_contribution: float | None = None
    last_move_pickup_ranking_holding_3_worst_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_3_best_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_3_best_placement_q_mean: float | None = None
    last_move_pickup_ranking_holding_3_selected_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_3_ordinary_target_mean: float | None = None
    last_move_pickup_ranking_holding_3_selected_q_minus_ordinary_target_mean: float | None = None
    last_move_pickup_ranking_holding_3_base_q_would_push_selected_up_fraction: float | None = None
    last_move_pickup_ranking_holding_3_base_q_would_push_selected_down_fraction: float | None = None
    last_move_pickup_ranking_holding_3_selected_pickup_is_worst_fraction: float | None = None
    last_move_pickup_ranking_holding_3_ranking_hinge_active_fraction: float | None = None
    last_move_pickup_ranking_holding_3_base_q_ranking_direct_opposition_fraction: float | None = (
        None
    )
    last_move_pickup_ranking_holding_3_ordinary_reward_to_go_samples: int = 0
    last_move_pickup_ranking_holding_3_repeated_move_hard_target_samples: int = 0
    last_move_pickup_ranking_holding_3_all_move_turn_hard_target_samples: int = 0
    last_move_pickup_ranking_holding_3_additive_adjustment_target_samples: int = 0
    last_move_pickup_ranking_holding_3_other_override_target_samples: int = 0
    last_move_pickup_ranking_holding_4_samples: int = 0
    last_move_pickup_ranking_holding_4_loss: float | None = None
    last_move_pickup_ranking_holding_4_q1_pickup_fraction: float | None = None
    last_move_pickup_ranking_holding_4_all_pickups_above_all_placements_fraction: float | None = (
        None
    )
    last_move_pickup_ranking_holding_4_margin_satisfied_fraction: float | None = None
    last_move_pickup_ranking_holding_4_worst_pickup_minus_best_placement_gap_mean: float | None = (
        None
    )
    last_move_pickup_ranking_holding_4_weighted_contribution: float | None = None
    last_move_pickup_ranking_holding_4_worst_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_4_best_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_4_best_placement_q_mean: float | None = None
    last_move_pickup_ranking_holding_4_selected_pickup_q_mean: float | None = None
    last_move_pickup_ranking_holding_4_ordinary_target_mean: float | None = None
    last_move_pickup_ranking_holding_4_selected_q_minus_ordinary_target_mean: float | None = None
    last_move_pickup_ranking_holding_4_base_q_would_push_selected_up_fraction: float | None = None
    last_move_pickup_ranking_holding_4_base_q_would_push_selected_down_fraction: float | None = None
    last_move_pickup_ranking_holding_4_selected_pickup_is_worst_fraction: float | None = None
    last_move_pickup_ranking_holding_4_ranking_hinge_active_fraction: float | None = None
    last_move_pickup_ranking_holding_4_base_q_ranking_direct_opposition_fraction: float | None = (
        None
    )
    last_move_pickup_ranking_holding_4_ordinary_reward_to_go_samples: int = 0
    last_move_pickup_ranking_holding_4_repeated_move_hard_target_samples: int = 0
    last_move_pickup_ranking_holding_4_all_move_turn_hard_target_samples: int = 0
    last_move_pickup_ranking_holding_4_additive_adjustment_target_samples: int = 0
    last_move_pickup_ranking_holding_4_other_override_target_samples: int = 0
    last_pointless_final_placement_ranking_samples: int = 0
    last_pointless_final_placement_ranking_loss: float | None = None
    last_pointless_final_placement_restorative_q1_fraction: float | None = None
    last_pointless_final_placement_margin_satisfied_fraction: float | None = None
    last_pointless_final_placement_q_gap_mean: float | None = None
    last_pointless_final_placement_immediate_q_undo_samples: int = 0
    last_pointless_final_placement_multi_piece_samples: int = 0
    last_pointless_final_placement_exact_restoration_samples: int = 0
    last_pointless_final_placement_equivalent_rearrangement_samples: int = 0
    mean_policy_loss: float | None = None
    tier_games: dict[int, int] = field(default_factory=dict)
    tier_wins: dict[int, int] = field(default_factory=dict)
    tier_selected_rank_total: dict[int, int] = field(default_factory=dict)
    tier_epsilon_selections: dict[int, int] = field(default_factory=dict)
    tier_top_k_selections: dict[int, int] = field(default_factory=dict)
    tier_immediate_reward_total: dict[int, float] = field(default_factory=dict)
    tier_reward_to_go_total: dict[int, float] = field(default_factory=dict)


@dataclass
class ShadowPolicyMetrics:
    """Defer shadow-policy aggregation to one vectorized device operation per game."""

    decisions: int = 0
    records: list = field(default_factory=list)

    def record(self, q_group_scores, policy_logits, groups):
        if len(q_group_scores) != len(groups):
            raise ValueError("Shadow diagnostics require one Q score per semantic group")
        self.decisions += 1
        self.records.append((q_group_scores, policy_logits, groups))

    def averages(self):
        if not self.decisions:
            return (None,) * 6

        policy_rows = torch.stack([record[1] for record in self.records])
        metric_device = policy_rows.device
        metric_dtype = policy_rows.dtype
        member_rows = []
        member_actions = []
        member_groups = []
        q_scores = []
        group_decisions = []
        group_positions = []
        group_offsets = []
        group_id = 0
        max_group_count = 0
        for decision_index, (decision_q_scores, _policy_logits, groups) in enumerate(self.records):
            group_offsets.append(group_id)
            max_group_count = max(max_group_count, len(groups))
            for group_position, (q_score, group) in enumerate(zip(decision_q_scores, groups)):
                q_scores.append(q_score)
                group_decisions.append(decision_index)
                group_positions.append(group_position)
                member_rows.extend((decision_index,) * len(group))
                member_actions.extend(group)
                member_groups.extend((group_id,) * len(group))
                group_id += 1

        member_rows = torch.tensor(member_rows, dtype=torch.long, device=metric_device)
        member_actions = torch.tensor(member_actions, dtype=torch.long, device=metric_device)
        member_groups = torch.tensor(member_groups, dtype=torch.long, device=metric_device)
        group_decisions = torch.tensor(
            group_decisions,
            dtype=torch.long,
            device=metric_device,
        )
        group_positions = torch.tensor(
            group_positions,
            dtype=torch.long,
            device=metric_device,
        )
        q_scores = torch.tensor(q_scores, dtype=metric_dtype, device=metric_device)
        group_offsets = torch.tensor(group_offsets, dtype=torch.long, device=metric_device)

        member_logits = policy_rows[member_rows, member_actions]
        group_logits = torch.zeros(group_id, dtype=metric_dtype, device=metric_device)
        group_logits.scatter_add_(0, member_groups, member_logits)
        group_counts = torch.zeros(group_id, dtype=metric_dtype, device=metric_device)
        group_counts.scatter_add_(0, member_groups, torch.ones_like(member_logits))
        group_logits /= group_counts

        policy_maxima = torch.full(
            (self.decisions,),
            -torch.inf,
            dtype=metric_dtype,
            device=metric_device,
        )
        policy_maxima.scatter_reduce_(
            0,
            group_decisions,
            group_logits,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(group_logits - policy_maxima[group_decisions])
        denominators = torch.zeros(
            self.decisions,
            dtype=metric_dtype,
            device=metric_device,
        )
        denominators.scatter_add_(0, group_decisions, exponentials)
        probabilities = exponentials / denominators[group_decisions]

        policy_top_positions = self._first_maximum_positions(
            group_logits,
            group_decisions,
            group_positions,
            policy_maxima,
            max_group_count,
        )
        q_maxima = torch.full_like(policy_maxima, -torch.inf)
        q_maxima.scatter_reduce_(
            0,
            group_decisions,
            q_scores,
            reduce="amax",
            include_self=True,
        )
        q_top_positions = self._first_maximum_positions(
            q_scores,
            group_decisions,
            group_positions,
            q_maxima,
            max_group_count,
        )
        policy_top_groups = group_offsets + policy_top_positions
        selected_q_scores = q_scores[policy_top_groups]
        outranks_policy_top = (q_scores > selected_q_scores[group_decisions]) | (
            (q_scores == selected_q_scores[group_decisions])
            & (group_positions < policy_top_positions[group_decisions])
        )
        policy_q_ranks = torch.ones(
            self.decisions,
            dtype=torch.long,
            device=metric_device,
        )
        policy_q_ranks.scatter_add_(
            0,
            group_decisions,
            outranks_policy_top.to(torch.long),
        )

        entropy_by_group = -(probabilities * torch.log(probabilities.clamp_min(1e-12)))
        entropy = torch.zeros_like(policy_maxima)
        entropy.scatter_add_(0, group_decisions, entropy_by_group)
        probability_rows = torch.zeros(
            (self.decisions, max_group_count),
            dtype=metric_dtype,
            device=metric_device,
        )
        probability_rows[group_decisions, group_positions] = probabilities
        top_values = torch.topk(
            probability_rows,
            min(10, max_group_count),
            dim=1,
            sorted=True,
        ).values
        totals = torch.stack(
            (
                (policy_top_positions == q_top_positions).sum().to(metric_dtype),
                policy_q_ranks.sum().to(metric_dtype),
                entropy.sum(),
                top_values[:, :2].sum(),
                top_values[:, :5].sum(),
                top_values.sum(),
            )
        )
        return tuple((totals / self.decisions).cpu().tolist())

    @staticmethod
    def _first_maximum_positions(
        scores,
        group_decisions,
        group_positions,
        maxima,
        missing_position,
    ):
        candidates = torch.where(
            scores == maxima[group_decisions],
            group_positions,
            missing_position,
        )
        positions = torch.full(
            maxima.shape,
            missing_position,
            dtype=torch.long,
            device=scores.device,
        )
        positions.scatter_reduce_(
            0,
            group_decisions,
            candidates,
            reduce="amin",
            include_self=True,
        )
        return positions


@dataclass(frozen=True)
class _ShadowFilterSelection:
    decision_index: int
    action_index: int
    action_type: str
    semantic_action_indices: tuple[int, ...]
    semantic_q_rank: int
    q_value: float
    q_gap_from_best: float
    semantic_policy_rank: torch.Tensor
    policy_probability: torch.Tensor


@dataclass
class ShadowFilterAudit:
    """Collect observational semantic rejection candidates without affecting play."""

    records: list[_ShadowFilterSelection] = field(default_factory=list)

    def record(self, decision_index, selection, policy_logits, groups):
        if len(selection.semantic_q_scores) != len(groups):
            raise ValueError("Shadow filtering requires one Q score per semantic group")
        selected_position = next(
            position for position, group in enumerate(groups) if selection.action_index in group
        )
        group_logits = semantic_group_logits(policy_logits, groups)
        selected_logit = group_logits[selected_position]
        positions = torch.arange(len(groups), device=group_logits.device)
        policy_rank = (
            1
            + (
                (group_logits > selected_logit)
                | ((group_logits == selected_logit) & (positions < selected_position))
            ).sum()
        )
        policy_probability = torch.softmax(group_logits, dim=0)[selected_position]
        q_value = selection.semantic_q_scores[selected_position]
        self.records.append(
            _ShadowFilterSelection(
                decision_index=decision_index,
                action_index=selection.action_index,
                action_type=type(_ACTIONS_BY_INDEX[selection.action_index]).__name__,
                semantic_action_indices=groups[selected_position],
                semantic_q_rank=selection.model_rank,
                q_value=q_value,
                q_gap_from_best=max(selection.semantic_q_scores) - q_value,
                semantic_policy_rank=policy_rank,
                policy_probability=policy_probability,
            )
        )

    def flagged_outcomes(
        self,
        reward_to_go_decisions,
        training_decisions,
        terminal_rewards,
        final_scores,
        winner_indices,
    ):
        if not self.records:
            return ()
        policy_values = torch.stack(
            [
                torch.stack(
                    (
                        record.semantic_policy_rank.to(torch.float32),
                        record.policy_probability.to(torch.float32),
                    )
                )
                for record in self.records
            ]
        ).cpu()
        winners = set(winner_indices)
        outcomes = []
        for record, (policy_rank_value, policy_probability_value) in zip(
            self.records, policy_values.tolist()
        ):
            policy_rank = int(policy_rank_value)
            if not would_shadow_filter(policy_rank, record.semantic_q_rank):
                continue
            reward_decision = reward_to_go_decisions[record.decision_index]
            training_decision = training_decisions[record.decision_index]
            player_index = training_decision.acting_player_index
            receives_terminal_credit = training_decision.receives_terminal_credit
            outcomes.append(
                ShadowFilterAuditRecord(
                    decision_index=record.decision_index,
                    action_index=record.action_index,
                    action_type=record.action_type,
                    semantic_action_indices=record.semantic_action_indices,
                    semantic_q_rank=record.semantic_q_rank,
                    q_value=record.q_value,
                    q_gap_from_best=record.q_gap_from_best,
                    semantic_policy_rank=policy_rank,
                    policy_probability=policy_probability_value,
                    immediate_reward=training_decision.immediate_reward,
                    local_training_target=training_decision.local_training_target,
                    local_training_adjustment=training_decision.local_training_adjustment,
                    reward_to_go=reward_decision.reward_to_go,
                    final_training_target=training_decision.reward_to_go,
                    receives_terminal_credit=receives_terminal_credit,
                    terminal_credit_value=(
                        terminal_rewards[player_index] if receives_terminal_credit else 0.0
                    ),
                    acting_player_index=player_index,
                    acting_player_final_score=final_scores[player_index],
                    acting_player_won=player_index in winners,
                    policy_tier=training_decision.policy_tier,
                    used_epsilon=training_decision.used_epsilon,
                )
            )
        return tuple(outcomes)


def would_shadow_filter(
    policy_rank,
    q_rank,
    *,
    policy_top_k=SHADOW_FILTER_POLICY_TOP_K,
    q_top_k=SHADOW_FILTER_Q_TOP_K,
):
    """Return whether both conservative semantic-rank rejection gates agree."""
    return policy_rank > policy_top_k and q_rank > q_top_k


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _post_contexts_by_slot(game):
    return tuple(
        (route_index, route, post)
        for route_index, route in enumerate(game.selected_map.routes)
        for post in route.posts
    )


def _action_index_tuple(action_indices):
    if isinstance(action_indices, tuple):
        return action_indices
    if isinstance(action_indices, torch.Tensor):
        return tuple(action_indices.tolist())
    return tuple(int(index) for index in action_indices)


def action_phase_selection_groups(game, legal_indices, post_contexts=None):
    """Collapse equivalent non-maritime post interactions for action selection."""
    post_contexts = post_contexts or _post_contexts_by_slot(game)
    groups = {}
    for action_index in _action_index_tuple(legal_indices):
        action = _ACTIONS_BY_INDEX[action_index]
        context = (
            post_contexts[action.post_slot]
            if isinstance(action, PostInteraction) and action.post_slot < len(post_contexts)
            else None
        )
        if context is not None:
            route_index, route, post = context
            if route.required_circles > 0:
                key = ("action", action_index)
            elif post.owner is None:
                key = ("route_placement", route_index, action.shape)
            elif post.owner is game.current_player:
                key = ("move_pickup", route_index, post.owner_piece_shape)
            else:
                key = (
                    "displacement_target",
                    route_index,
                    id(post.owner),
                    post.owner_piece_shape,
                    action.shape,
                )
        else:
            key = ("action", action_index)
        groups.setdefault(key, []).append(action_index)
    grouped = tuple(tuple(group) for group in groups.values())
    return grouped if any(len(group) > 1 for group in grouped) else None


def move_workflow_exploration_categories(
    game,
    legal_indices,
    *,
    opponent_pickups=False,
    any_pickups=False,
    post_contexts=None,
):
    """Group equivalent normal-Move or Move-3 clicks into semantic choices."""
    post_contexts = post_contexts or _post_contexts_by_slot(game)

    pickup_groups = {}
    placement_groups = {}
    for action_index in _action_index_tuple(legal_indices):
        action = _ACTIONS_BY_INDEX[action_index]
        context = (
            post_contexts[action.post_slot]
            if isinstance(action, PostInteraction) and action.post_slot < len(post_contexts)
            else None
        )
        if context is not None:
            route_index, route, post = context
            if any_pickups:
                is_pickup = post.owner is not None
            elif opponent_pickups:
                is_pickup = post.owner is not None and post.owner is not game.current_player
            else:
                is_pickup = post.owner is game.current_player
            if is_pickup and route.required_circles == 0:
                # The occupied post already determines the piece shape.
                key = (
                    "pickup",
                    route_index,
                    id(post.owner),
                    post.owner_piece_shape,
                )
                groups = pickup_groups
            elif is_pickup:
                key = ("pickup", action.post_slot)
                groups = pickup_groups
            elif post.owner is None and route.required_circles == 0:
                # Shape remains a meaningful choice when several differently shaped
                # pieces are held; only equivalent post locations are collapsed.
                key = ("route_destination", route_index, action.shape)
                groups = placement_groups
            else:
                key = ("action", action_index)
                groups = placement_groups
        else:
            key = ("action", action_index)
            groups = placement_groups
        groups.setdefault(key, []).append(action_index)
    return tuple(
        tuple(tuple(group) for group in groups.values())
        for groups in (pickup_groups, placement_groups)
        if groups
    )


def preplacement_move_action_families(
    normal_move_workflow_id,
    picked_up_piece_count,
    placement_has_begun,
    effective_movement_capacity,
    exploration_categories,
):
    """Return true semantic families while an unfinished paid Move can pick up more."""
    if (
        normal_move_workflow_id is None
        or picked_up_piece_count < 1
        or placement_has_begun
        or picked_up_piece_count >= effective_movement_capacity
        or exploration_categories is None
        or len(exploration_categories) != 2
    ):
        return (), ()
    return exploration_categories


def case_a_move_action_families(
    normal_move_workflow_id,
    picked_up_piece_count,
    placement_has_begun,
    effective_movement_capacity,
    exploration_categories,
):
    """Retain the holding-one capture used by the Pickup1 scaffold and telemetry."""
    if picked_up_piece_count != 1:
        return (), ()
    return preplacement_move_action_families(
        normal_move_workflow_id,
        picked_up_piece_count,
        placement_has_begun,
        effective_movement_capacity,
        exploration_categories,
    )


def move1_scaffold_action_mask(
    base_mask,
    pickup_action_groups,
    placement_action_groups,
    *,
    enabled=True,
):
    """Hide only premature placements at a qualifying pickup Move boundary."""
    if not enabled or not pickup_action_groups or not placement_action_groups:
        return base_mask
    scaffold_mask = base_mask.clone()
    for group in placement_action_groups:
        for action_index in group:
            scaffold_mask[action_index] = False
    return scaffold_mask


def record_move1_scaffold_readiness(
    metrics,
    semantic_q_scores,
    pickup_group_count,
    top_k,
    *,
    margin=CASE_A_FAMILY_RANK_MARGIN,
):
    """Record what the unmasked Q ranking would do at one scaffolded state."""
    placement_group_count = len(semantic_q_scores) - pickup_group_count
    if pickup_group_count < 1 or placement_group_count < 1:
        raise ValueError("Scaffold readiness requires pickup and placement semantic actions")
    pickup_scores = semantic_q_scores[:pickup_group_count]
    placement_scores = semantic_q_scores[pickup_group_count:]
    ranked_positions = tuple(
        sorted(
            range(len(semantic_q_scores)),
            key=lambda index: (-semantic_q_scores[index], index),
        )
    )
    q1_is_pickup = ranked_positions[0] < pickup_group_count
    pair_violations = [
        max(0.0, placement_q - pickup_q + float(margin))
        for pickup_q in pickup_scores
        for placement_q in placement_scores
    ]
    satisfied_pairs = sum(value == 0.0 for value in pair_violations)
    effective_k = min(top_k or len(ranked_positions), len(ranked_positions))
    top_k_positions = ranked_positions[:effective_k]
    top_k_pickups = sum(position < pickup_group_count for position in top_k_positions)

    metrics.move1_scaffold_mask_states += 1
    metrics.move1_scaffold_masked_placement_semantic_actions += placement_group_count
    metrics.move1_scaffold_legal_pickup_semantic_actions += pickup_group_count
    metrics.move1_scaffold_unmasked_q1_pickups += int(q1_is_pickup)
    metrics.move1_scaffold_unmasked_q1_placements += int(not q1_is_pickup)
    metrics.move1_scaffold_all_pickups_above_all_placements += int(
        min(pickup_scores) > max(placement_scores)
    )
    metrics.move1_scaffold_margin_satisfied_pair_fraction_total += satisfied_pairs / len(
        pair_violations
    )
    metrics.move1_scaffold_family_ranking_loss_total += max(pair_violations)
    metrics.move1_scaffold_best_pickup_minus_best_placement.append(
        max(pickup_scores) - max(placement_scores)
    )
    metrics.move1_scaffold_unmasked_top_k_pickup_fraction_total += top_k_pickups / effective_k
    metrics.move1_scaffold_unmasked_top_k_has_pickup += int(top_k_pickups > 0)


def _percentile(values, fraction):
    """Return a linearly interpolated percentile for a non-empty numeric sequence."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def move_continuation_action_families(
    normal_move_workflow_id,
    picked_up_piece_count,
    placement_has_begun,
    effective_movement_capacity,
    exploration_categories,
):
    """Retain the depth-2+ capture used by continuation telemetry."""
    if picked_up_piece_count < 2:
        return (), ()
    return preplacement_move_action_families(
        normal_move_workflow_id,
        picked_up_piece_count,
        placement_has_begun,
        effective_movement_capacity,
        exploration_categories,
    )


def training_move_scaffold_action_families(
    normal_move_workflow_id,
    picked_up_piece_count,
    placement_has_begun,
    effective_movement_capacity,
    exploration_categories,
    *,
    evaluation,
):
    """Return original families at every eligible training pickup boundary."""
    if evaluation or not MOVE_PICKUP_TRAINING_SCAFFOLD_ENABLED:
        return (), ()
    return preplacement_move_action_families(
        normal_move_workflow_id,
        picked_up_piece_count,
        placement_has_begun,
        effective_movement_capacity,
        exploration_categories,
    )


def _would_complete_east_west(game, player, route):
    if player in game.players_who_completed_east_west:
        return False
    occupied = {city for city in game.selected_map.cities if city.has_office_owned_by(player)}
    start_name, end_name = game.selected_map.east_west_cities

    def connected(cities):
        start = next((city for city in cities if city.name == start_name), None)
        end = next((city for city in cities if city.name == end_name), None)
        if start is None or end is None:
            return False
        visited = {start}
        pending = [start]
        while pending:
            city = pending.pop()
            for candidate_route in city.routes:
                for neighbor in candidate_route.cities:
                    if neighbor in cities and neighbor not in visited:
                        visited.add(neighbor)
                        pending.append(neighbor)
        return end in visited

    if connected(occupied):
        return False
    for city in route.cities:
        if city.color == DARK_GREEN or not city.has_empty_office():
            continue
        office_color = city.get_next_open_office_color()
        if not player.player_can_claim_office(office_color):
            continue
        if not city.has_required_piece_shape(player, route):
            continue
        if connected(occupied | {city}):
            return True
    return False


def valuable_completed_route_slots(game, player):
    """Return completed routes offering the player an immediate high-value outcome."""
    valuable = set()
    for route_index, route in enumerate(game.selected_map.routes):
        if not route.is_controlled_by(player):
            continue
        has_upgrade = False
        for city in route.cities:
            for upgrade in city.upgrade_city_type:
                if upgrade == "SpecialPrestigePoints":
                    prestige = game.selected_map.specialprestigepoints
                    has_upgrade |= bool(
                        route.contains_a_circle()
                        and prestige is not None
                        and prestige.can_claim_prestige(player)
                    )
                else:
                    has_upgrade |= (
                        getattr(player, upgrade.lower()) != UPGRADE_MAX_VALUES[upgrade.lower()]
                    )
        controls_both_cities = all(city.determine_controller() is player for city in route.cities)
        if (
            route.bonus_marker
            or route.permanent_bonus_marker
            or has_upgrade
            or controls_both_cities
            or _would_complete_east_west(game, player, route)
        ):
            valuable.add(route_index)
    return valuable


def training_action_mask(
    game,
    *,
    disable_move_action: bool,
    move_general_stock_threshold: int = 3,
    base_mask=None,
    post_contexts=None,
) -> torch.Tensor:
    """Prefer non-Move interactions, restoring Move when it is the only legal choice."""
    mask = (
        torch.tensor(game.ai_action_mask(), dtype=torch.bool)
        if base_mask is None
        else base_mask.to(dtype=torch.bool).clone()
    )
    if not disable_move_action or game.turn_phase is not TurnPhase.ACTIONS:
        return mask

    acting_player = game.players[game.active_player]
    general_stock = getattr(acting_player, "general_stock_squares", 0) + getattr(
        acting_player, "general_stock_circles", 0
    )
    if general_stock < move_general_stock_threshold:
        return mask
    original_mask = mask.clone()
    post_contexts = post_contexts or _post_contexts_by_slot(game)
    for index in mask.nonzero(as_tuple=False).flatten().tolist():
        action = _ACTIONS_BY_INDEX[index]
        if isinstance(action, PostInteraction):
            post = (
                post_contexts[action.post_slot][2]
                if action.post_slot < len(post_contexts)
                else None
            )
            if post is not None and post.owner is acting_player:
                mask[index] = False
    return mask if mask.any() else original_mask


def assign_reward_to_go(decisions, terminal_rewards, gamma):
    """Discount reward streams once per player turn, not once per interaction."""
    if not decisions:
        return ()
    running = [float(reward) for reward in terminal_rewards]
    terminal_credit = [float(reward) for reward in terminal_rewards]
    latest_turns = [None] * len(terminal_rewards)
    completed = list(decisions)
    for index in range(len(decisions) - 1, -1, -1):
        decision = decisions[index]
        player_count = len(terminal_rewards)
        for player_index, reward in enumerate(decision.player_reward_deltas):
            turns_started = max(
                0,
                (decision.game_turn_number - (player_index + 1)) // player_count + 1,
            )
            if latest_turns[player_index] is not None:
                discount = gamma ** (latest_turns[player_index] - turns_started)
                running[player_index] *= discount
                terminal_credit[player_index] *= discount
            running[player_index] += reward
            latest_turns[player_index] = turns_started
        player_index = decision.acting_player_index
        reward_to_go = running[player_index]
        if not decision.receives_terminal_credit:
            reward_to_go -= terminal_credit[player_index]
        completed[index] = replace(decision, reward_to_go=reward_to_go)
    return tuple(completed)


def assign_training_targets(decisions, terminal_rewards, gamma):
    """Assign game returns, then override only explicitly local movement mistakes."""
    completed = assign_reward_to_go(decisions, terminal_rewards, gamma)
    return apply_local_training_targets(completed)


def apply_local_training_targets(decisions):
    """Apply local target overrides and adjustments after reward-to-go is complete."""
    return tuple(
        replace(decision, reward_to_go=decision.local_training_target)
        if decision.local_training_target is not None
        else replace(
            decision,
            reward_to_go=decision.reward_to_go + decision.local_training_adjustment,
        )
        if decision.local_training_adjustment
        else decision
        for decision in decisions
    )


def _training_priority_value(decision):
    """Preserve reward/local-mistake priority after local additive adjustments."""
    if decision.local_training_target is not None:
        return abs(decision.local_training_target)
    return abs(decision.immediate_reward + decision.local_training_adjustment)


def mark_movement_workflow_target(decisions, workflow_id, target):
    """Give one completed movement workflow a local target without changing prior play."""
    if workflow_id is None:
        raise ValueError("A local movement target requires a workflow ID")
    found = False
    for index in range(len(decisions) - 1, -1, -1):
        decision = decisions[index]
        if decision.movement_workflow_id == workflow_id:
            local_target = float(target)
            if decision.local_training_target is not None:
                local_target = min(local_target, decision.local_training_target)
            decisions[index] = replace(decision, local_training_target=local_target)
            found = True
        elif found:
            break
    if not found:
        raise ValueError(f"Movement workflow {workflow_id} has no recorded decisions")


def add_movement_workflow_adjustment(decisions, workflow_id, adjustment):
    """Add a local adjustment to one Move workflow without changing earlier returns."""
    if workflow_id is None:
        raise ValueError("A local movement adjustment requires a workflow ID")
    found = False
    for index in range(len(decisions) - 1, -1, -1):
        decision = decisions[index]
        if decision.movement_workflow_id == workflow_id:
            decisions[index] = replace(
                decision,
                local_training_adjustment=(decision.local_training_adjustment + float(adjustment)),
            )
            found = True
        elif found:
            break
    if not found:
        raise ValueError(f"Movement workflow {workflow_id} has no recorded decisions")


def set_single_piece_move_utilization_target(
    decisions,
    workflow_id,
    target,
    *,
    target_role,
):
    """Hard-target a Move initiated when only one pickup was legally possible."""
    indices = [
        index
        for index, decision in enumerate(decisions)
        if decision.movement_workflow_id == workflow_id
    ]
    if len(indices) != 2:
        raise ValueError(f"Single-piece movement workflow {workflow_id} must have two decisions")
    if target_role == "initial_pickup":
        target_index = indices[0]
    else:
        raise ValueError(f"Unknown single-piece Move utilization target role: {target_role}")
    decision = decisions[target_index]
    target = float(target)
    if decision.local_training_target is not None and decision.local_training_target <= target:
        return
    decisions[target_index] = replace(
        decision,
        local_training_target=target,
        move1_utilization_penalty_role=target_role,
    )


def record_applied_move1_utilization_penalties(metrics, training_decisions):
    """Count Move1 hard targets that survive precedence into final Q targets."""
    roles = tuple(
        decision.move1_utilization_penalty_role
        for decision in training_decisions
        if decision.move1_utilization_penalty_role is not None
        and decision.local_training_target == MOVE1_UTILIZATION_LOCAL_TARGET
    )
    metrics.move1_utilization_penalties_applied = len(roles)
    metrics.move1_penalties_on_placement = roles.count("first_placement")
    metrics.move1_penalties_on_single_available_initiation = roles.count("initial_pickup")


def grant_movement_workflow_terminal_credit(decisions, workflow_id):
    """Restore terminal credit to every interaction in one normal-Move workflow."""
    if workflow_id is None:
        raise ValueError("Terminal movement credit requires a workflow ID")
    found = False
    for index in range(len(decisions) - 1, -1, -1):
        decision = decisions[index]
        if decision.movement_workflow_id == workflow_id:
            decisions[index] = replace(decision, receives_terminal_credit=True)
            found = True
        elif found:
            break
    if not found:
        raise ValueError(f"Movement workflow {workflow_id} has no recorded decisions")


def credited_movement_workflows(workflow_routes, completed_routes, claimed_route):
    """Return Move workflows that contributed to the immediately claimed route."""
    if claimed_route not in completed_routes:
        return ()
    return tuple(
        workflow_id
        for workflow_id, destination_routes in workflow_routes
        if claimed_route in destination_routes
    )


def apply_all_move_turn_target(decisions, workflow_ids, spent_actions):
    """Penalize only Move workflows when every paid action in a turn was Move."""
    workflow_ids = tuple(workflow_ids)
    if spent_actions < 2 or len(workflow_ids) != spent_actions:
        return False
    for workflow_id in workflow_ids:
        mark_movement_workflow_target(decisions, workflow_id, ALL_MOVE_TURN_LOCAL_TARGET)
    return True


def finalize_all_move_turn(
    decisions,
    movement_metrics,
    workflow_ids,
    spent_actions,
    *,
    pointless_move_audit_enabled=False,
):
    """Apply and record the all-Move target when the current turn closes."""
    applied = apply_all_move_turn_target(decisions, workflow_ids, spent_actions)
    movement_metrics.all_move_turn_penalties += int(applied)
    if applied and pointless_move_audit_enabled:
        movement_metrics.pointless_move_attribution.mark_all_move_turn(workflow_ids)
    return applied


def calculate_terminal_rewards(game, winner_indices, game_end_trigger_player):
    """Return winner-only final-score rewards and the successful-trigger bonus."""
    rewards = [0.0] * len(game.players)
    for winner_index in winner_indices:
        rewards[winner_index] = float(
            PRESTIGE_REWARD_MULTIPLIER * game.players[winner_index].final_score
        )
    if game_end_trigger_player in winner_indices:
        rewards[game_end_trigger_player] += END_GAME_WINNER_BONUS
    return tuple(rewards)


def completed_game_reason(game):
    """Describe every authoritative condition present when a game ends."""
    reasons = []
    if any(player.score >= 20 for player in game.players):
        reasons.append("20_points")
    if game.bonus_pool_exhausted_during_claim:
        reasons.append("bonus_markers_exhausted")
    if game.current_full_cities_count >= game.selected_map.max_full_cities:
        reasons.append("full_cities")
    if not reasons:
        raise TrainingRunError("Completed game has no recognized end condition")
    return "+".join(reasons)


def income_efficiency_penalty(bank_capacity, pieces_received, scale):
    """Return the proportional penalty for unused finite Bank capacity."""
    if bank_capacity == UPGRADE_MAX_VALUES["bank"]:
        return 0.0
    unused_fraction = max(bank_capacity - pieces_received, 0) / bank_capacity
    return -scale * unused_fraction


def apply_income_efficiency_penalty(
    reward_deltas,
    *,
    action,
    turn_phase,
    acting_player_index,
    bank_capacity,
    pieces_received,
    scale,
):
    """Apply normal-Income inefficiency to only the acting player's reward."""
    if turn_phase is not TurnPhase.ACTIONS or not isinstance(action, IncomeInteraction):
        return tuple(reward_deltas)
    adjusted = list(reward_deltas)
    adjusted[acting_player_index] += income_efficiency_penalty(
        bank_capacity, pieces_received, scale
    )
    return tuple(adjusted)


def movement_efficiency_penalty(pieces_moved, movement_capacity):
    """Penalize only clearly inefficient completed normal Move actions."""
    if pieces_moved == 2 and movement_capacity >= 3:
        return -100.0
    return 0.0


def single_piece_move_utilization_target(
    pieces_moved,
    *,
    immediate_q_undo,
    legal_pickups_at_start,
):
    """Choose which decision receives the one-piece Move hard target, if any."""
    if pieces_moved != 1 or immediate_q_undo:
        return None
    if legal_pickups_at_start == 1:
        return "initial_pickup"
    # Case A (another pickup remained) is now taught by a semantic-family
    # ranking loss at the pre-placement decision, not a hard TD target on the
    # one placement that happened to be selected.
    return None


def legal_normal_move_pickup_post_slots(legal_action_indices, post_contexts, player):
    """Return distinct currently legal normal-Move pickup posts owned by ``player``."""
    return frozenset(
        action.post_slot
        for action_index in legal_action_indices
        if isinstance((action := _ACTIONS_BY_INDEX[action_index]), PostInteraction)
        and post_contexts[action.post_slot][2].owner is player
    )


def consecutive_move_penalty(movement_capacity, consecutive_moves):
    """Penalize implausible repeated normal Move actions within one turn."""
    if consecutive_moves >= 3:
        return float(REPEATED_MOVE_LOCAL_TARGET)
    if movement_capacity >= 4 and consecutive_moves >= 2:
        return float(CONSECUTIVE_HIGH_CAPACITY_MOVE_PENALTY)
    return 0.0


def _is_normal_move_in_progress(action_phase, player):
    return action_phase is TurnPhase.MOVE_PIECES and bool(player.holding_pieces)


def loaded_normal_move_context(game, post_contexts):
    """Reconstruct pre-pickup tracking for a saved normal Move continuation."""
    player = game.current_player
    snapshot = game.normal_move_pre_board_snapshot
    if (
        game.turn_phase is not TurnPhase.MOVE_PIECES
        or not player.holding_pieces
        or snapshot is None
    ):
        return None
    if any(
        (
            game.waiting_for_bm_move_any_2,
            game.waiting_for_bm_move3,
            game.waiting_for_place2_from_route,
            game.waiting_for_place2_in_scotland_or_wales,
        )
    ):
        return None

    origin_posts = []
    origin_pieces = []
    observed_routes = set()
    already_claimable_routes = set()
    completed_routes_before = set()
    movable_pieces_at_start = 0
    post_slots = {}
    for post_slot, (route_index, route, post) in enumerate(post_contexts):
        post_slots[post] = post_slot
        route_snapshot = snapshot[route_index]
        snapshot_owners = tuple(owner for owner, _shape in route_snapshot)
        if all(owner is player for owner in snapshot_owners):
            completed_routes_before.add(route_index)
        movable_pieces_at_start += sum(owner is player for owner in snapshot_owners)
        post_index = route.posts.index(post)
        owner, shape = route_snapshot[post_index]
        if owner is player and not post.is_owned():
            origin_posts.append(post)
            origin_pieces.append((post, owner, shape))
            observed_routes.add(route_index)
            if all(snapshot_owner is player for snapshot_owner in snapshot_owners):
                already_claimable_routes.add(route_index)

    held = Counter((shape, owner) for shape, owner, _region in player.holding_pieces)
    origins = Counter((shape, owner) for _post, owner, shape in origin_pieces)
    if held != origins or not origin_posts:
        raise TrainingRunError("Saved normal Move state does not match its pre-Move snapshot")
    return LoadedNormalMoveContext(
        tuple(origin_posts),
        tuple(origin_pieces),
        frozenset(completed_routes_before),
        frozenset(observed_routes),
        frozenset(already_claimable_routes),
        movable_pieces_at_start,
        post_slots[origin_posts[0]],
    )


def _pointless_movement_type_from_piece_states(
    origin_pieces,
    destination_pieces,
    post_routes=None,
):
    """Classify exact or equivalent movement from explicit post-piece states."""
    if not origin_pieces or len(origin_pieces) != len(destination_pieces):
        return None
    original = {post: (owner, shape) for post, owner, shape in origin_pieces}
    if len(original) != len(origin_pieces):
        return None
    destination_posts = tuple(post for post, _owner, _shape in destination_pieces)
    destination = {post: (owner, shape) for post, owner, shape in destination_pieces}
    exact_no_change = set(destination_posts) == set(original) and all(
        destination.get(post) == (owner, shape) for post, (owner, shape) in original.items()
    )
    if exact_no_change:
        return "exact_restoration"
    if len({(owner, shape) for _, owner, shape in origin_pieces}) > 1:
        return None
    if post_routes is None:
        return None
    origin_routes = [post_routes.get(post) for post in original]
    destination_routes = [post_routes.get(post) for post in destination_posts]
    involved_routes = set(origin_routes + destination_routes)
    if None in involved_routes or any(route.required_circles for route in involved_routes):
        return None
    before = Counter((post_routes[post], owner, shape) for post, owner, shape in origin_pieces)
    after = Counter((post_routes[post], owner, shape) for post, owner, shape in destination_pieces)
    if before == after:
        return "equivalent_rearrangement"
    return None


def pointless_movement_type(origin_pieces, destination_posts, post_routes=None):
    """Classify exact or indistinguishable non-maritime movement."""
    return _pointless_movement_type_from_piece_states(
        origin_pieces,
        tuple((post, post.owner, post.owner_piece_shape) for post in destination_posts),
        post_routes,
    )


def pointless_movement_penalty(origin_pieces, destination_posts, post_routes=None):
    """Penalize exact or indistinguishable non-maritime movement."""
    if pointless_movement_type(origin_pieces, destination_posts, post_routes) is None:
        return 0.0
    return float(POINTLESS_MOVEMENT_LOCAL_TARGET)


def pointless_final_placement_lesson_eligible(
    *,
    evaluation,
    normal_move_in_progress,
    selected_empty_placement,
    held_piece_count,
    legal_pickup_post_slots,
):
    """Allow final-restoration ranking only after legal pickup choices are exhausted."""
    return bool(
        not evaluation
        and normal_move_in_progress
        and selected_empty_placement
        and held_piece_count == 1
        and not legal_pickup_post_slots
    )


def pointless_final_placement_action_groups(
    origin_pieces,
    placed_destination_posts,
    post_routes,
    post_contexts,
    semantic_action_groups,
    selected_action_index,
):
    """Classify the selected final placement and its non-pointless semantic alternatives."""
    if not origin_pieces:
        return (), (), None
    owner = origin_pieces[0][1]
    selected_group = ()
    selected_type = None
    nonpointless_groups = []
    for group in semantic_action_groups:
        member_types = []
        for action_index in group:
            action = _ACTIONS_BY_INDEX[action_index]
            if not isinstance(action, PostInteraction) or action.post_slot >= len(post_contexts):
                member_types = []
                break
            _route_index, _route, post = post_contexts[action.post_slot]
            if post.is_owned():
                member_types = []
                break
            shape = "circle" if action.shape is PieceShape.MERCHANT else "square"
            candidate_destination_posts = tuple(placed_destination_posts) + (post,)
            candidate_destination_pieces = tuple(
                (
                    destination_post,
                    owner if destination_post is post else destination_post.owner,
                    shape if destination_post is post else destination_post.owner_piece_shape,
                )
                for destination_post in candidate_destination_posts
            )
            member_types.append(
                _pointless_movement_type_from_piece_states(
                    origin_pieces,
                    candidate_destination_pieces,
                    post_routes,
                )
            )
        if not member_types:
            continue
        member_is_pointless = tuple(value is not None for value in member_types)
        if selected_action_index in group:
            selected_group = tuple(group)
            selected_member = group.index(selected_action_index)
            selected_type = member_types[selected_member]
        # Never split a semantic alias group. A positive alternative must be
        # non-pointless for every raw representation in that group.
        if not any(member_is_pointless):
            nonpointless_groups.append(tuple(group))
    return selected_group, tuple(nonpointless_groups), selected_type


def completed_route_move_reward(routes_before, routes_after):
    """Reward net claimable routes created by one completed normal Move."""
    return float(MOVE_COMPLETED_ROUTE_REWARD * (len(routes_after) - len(routes_before)))


def move_route_focus_reward(rewarded_routes, destination_counts):
    """Reward a concentrated Move once per route until that route is claimed."""
    focused_routes = {
        route_index for route_index, count in destination_counts.items() if count >= 2
    }
    newly_rewarded = focused_routes - set(rewarded_routes)
    return frozenset(set(rewarded_routes) | focused_routes), float(
        MOVE_ROUTE_FOCUS_REWARD if newly_rewarded else 0
    )


def clear_move_route_focus_after_claim(rewarded_routes, action, turn_phase):
    """Make a claimed route eligible for a later Move-focus reward."""
    rewarded = set(rewarded_routes)
    if turn_phase is TurnPhase.ACTIONS and isinstance(action, RouteInteraction):
        rewarded.discard(action.route_slot)
    return frozenset(rewarded)


def update_move_claim_combo(
    pending_routes,
    *,
    action,
    turn_phase,
    action_was_spent,
    newly_completed_routes=(),
):
    """Reward claiming a route immediately after a Move filled that route."""
    if not action_was_spent:
        return frozenset(pending_routes), 0.0
    reward = (
        MOVE_CLAIM_COMBO_REWARD
        if turn_phase is TurnPhase.ACTIONS
        and isinstance(action, RouteInteraction)
        and action.route_slot in pending_routes
        else 0.0
    )
    return frozenset(newly_completed_routes), float(reward)


def record_move_pickup_route_claimability(
    observed_routes,
    already_claimable_routes,
    route_slot,
    was_claimable,
):
    """Preserve a route's claimability before its first pickup in one normal Move."""
    if route_slot in observed_routes:
        return
    observed_routes.add(route_slot)
    if was_claimable:
        already_claimable_routes.add(route_slot)


def move_claim_eligible_routes(newly_completed_routes, already_claimable_pickup_routes):
    """Exclude pickup-origin routes that were claimable before the Move disturbed them."""
    return frozenset(newly_completed_routes) - frozenset(already_claimable_pickup_routes)


def record_move_claim_reward_outcome(
    metrics,
    reward,
    *,
    action,
    turn_phase,
    blocked_already_claimable_routes=(),
):
    """Count actual Move-to-Claim reward application or its anti-loophole block."""
    if reward:
        metrics.move_claim_reward_awarded += 1
    elif (
        turn_phase is TurnPhase.ACTIONS
        and isinstance(action, RouteInteraction)
        and action.route_slot in blocked_already_claimable_routes
    ):
        metrics.move_claim_reward_blocked_already_claimable += 1


def record_move_route_creation(metrics, pieces_moved, newly_completed_routes):
    """Count route creation by all normal Moves and its single-piece subset."""
    created_route = bool(newly_completed_routes)
    metrics.moves_creating_claimable_route += int(created_route)
    metrics.single_piece_moves_creating_claimable_route += int(pieces_moved == 1 and created_route)


def update_single_piece_move_claim_routes(
    pending_routes,
    *,
    action,
    turn_phase,
    normal_move_completed,
    pieces_moved,
    newly_completed_routes=(),
):
    """Track whether the next paid claim converts a route created by a one-piece Move."""
    converted = bool(
        turn_phase is TurnPhase.ACTIONS
        and isinstance(action, RouteInteraction)
        and action.route_slot in pending_routes
    )
    next_routes = (
        frozenset(newly_completed_routes)
        if normal_move_completed and pieces_moved == 1
        else frozenset()
    )
    return next_routes, converted


def pointless_route_claim_penalty(
    *,
    action,
    turn_phase,
    action_was_spent,
    gained_office,
    gained_upgrade,
    gained_marker,
    gained_points,
    route_had_permanent_marker,
):
    """Penalize a paid route claim that gives its player no useful outcome."""
    if (
        turn_phase is not TurnPhase.ACTIONS
        or not isinstance(action, RouteInteraction)
        or not action_was_spent
    ):
        return 0.0
    gained_outcome = any(
        (
            gained_office,
            gained_upgrade,
            gained_marker,
            gained_points,
            route_had_permanent_marker,
        )
    )
    return 0.0 if gained_outcome else float(POINTLESS_ROUTE_CLAIM_PENALTY)


def route_building_post_reward(*, route_already_has_piece, is_displacement):
    """Reward a normal placement that develops an existing route presence."""
    if not route_already_has_piece:
        return 0.0
    return float(
        ROUTE_BUILDING_DISPLACEMENT_REWARD if is_displacement else ROUTE_BUILDING_PLACEMENT_REWARD
    )


def intermediate_ability_upgrade_reward(values_before, values_after):
    """Reward non-final steps on the four non-Keys ability tracks."""
    reward = 0
    for ability, before, after in zip(INTERMEDIATE_REWARDED_ABILITIES, values_before, values_after):
        maximum = (
            len(ACTIONS_MAX_VALUES) - 1 if ability == "actions" else UPGRADE_MAX_VALUES[ability]
        )
        if before != after and after != maximum:
            reward += (
                FIRST_ACTIONS_UPGRADE_REWARD
                if ability == "actions" and before == 0 and after == 1
                else INTERMEDIATE_ABILITY_UPGRADE_REWARD
            )
    return float(reward)


def apply_opponent_route_score_penalty(
    reward_deltas,
    *,
    action,
    turn_phase,
    acting_player_index,
    projected_reward_deltas,
):
    """Charge the acting player for opponents' net projected gains from a route claim."""
    if turn_phase is not TurnPhase.ACTIONS or not isinstance(action, RouteInteraction):
        return tuple(reward_deltas)
    opponent_reward = sum(
        max(reward, 0)
        for index, reward in enumerate(projected_reward_deltas)
        if index != acting_player_index
    )
    adjusted = list(reward_deltas)
    adjusted[acting_player_index] -= opponent_reward
    return tuple(adjusted)


def apply_route_completion_reward(reward_deltas, *, action, turn_phase, acting_player_index):
    """Give a small incentive for completing a route before opponent-point costs."""
    if turn_phase is not TurnPhase.ACTIONS or not isinstance(action, RouteInteraction):
        return tuple(reward_deltas)
    adjusted = list(reward_deltas)
    adjusted[acting_player_index] += ROUTE_COMPLETION_REWARD
    return tuple(adjusted)


def policy_quality_signal(targets, return_scale):
    """Bound signed trajectory quality without copying Q ranks or behavior odds."""
    return torch.tanh(targets / return_scale)


def semantic_group_logits(logits, groups):
    """Represent equivalent action indices once using their mean learned logit."""
    return torch.stack(
        [
            logits[torch.as_tensor(group, dtype=torch.long, device=logits.device)].mean()
            for group in groups
        ]
    )


def _policy_semantic_groups(sample):
    """Return the legal semantic choices and selected choice for one sample."""
    legal_mask = sample.legal_action_mask
    if legal_mask.device.type != "cpu":
        legal_mask = legal_mask.detach().cpu()
    legal_indices = tuple(legal_mask.nonzero(as_tuple=False).flatten().tolist())
    if sample.action_index not in legal_indices:
        legal_indices += (sample.action_index,)

    legal_set = set(legal_indices)
    stored_groups = list(sample.equivalent_action_groups)
    selected_group = sample.equivalent_action_indices
    if len(selected_group) > 1 and selected_group not in stored_groups:
        stored_groups.append(selected_group)

    grouped_by_index = {}
    for group in stored_groups:
        legal_group = tuple(index for index in group if index in legal_set)
        if len(legal_group) < 2 or any(index in grouped_by_index for index in legal_group):
            continue
        for index in legal_group:
            grouped_by_index[index] = legal_group

    semantic_groups = []
    emitted_groups = set()
    for index in legal_indices:
        group = grouped_by_index.get(index, (index,))
        if group in emitted_groups:
            continue
        semantic_groups.append(group)
        emitted_groups.add(group)

    selected_position = next(
        position for position, group in enumerate(semantic_groups) if sample.action_index in group
    )
    return tuple(semantic_groups), selected_position


def policy_batch_losses(policy_logits, samples, quality_signals):
    """Return bounded return-weighted losses for a batch of semantic choices."""
    samples = tuple(samples)
    if policy_logits.ndim != 2 or policy_logits.shape[0] != len(samples):
        raise ValueError("Policy logits must contain one row per training sample")
    if quality_signals.shape != (len(samples),):
        raise ValueError("Policy quality must contain one value per training sample")

    structures = [_policy_semantic_groups(sample) for sample in samples]
    member_rows = []
    member_actions = []
    member_group_ids = []
    group_decisions = []
    group_positions = []
    group_sizes = []
    selected_positions = []
    group_id = 0
    maximum_group_count = 0
    for decision_index, (groups, selected_position) in enumerate(structures):
        maximum_group_count = max(maximum_group_count, len(groups))
        selected_positions.append(selected_position)
        for group_position, group in enumerate(groups):
            group_decisions.append(decision_index)
            group_positions.append(group_position)
            group_sizes.append(len(group))
            member_rows.extend((decision_index,) * len(group))
            member_actions.extend(group)
            member_group_ids.extend((group_id,) * len(group))
            group_id += 1

    policy_device = policy_logits.device
    member_rows = torch.tensor(member_rows, dtype=torch.long, device=policy_device)
    member_actions = torch.tensor(member_actions, dtype=torch.long, device=policy_device)
    member_group_ids = torch.tensor(member_group_ids, dtype=torch.long, device=policy_device)
    group_decisions = torch.tensor(group_decisions, dtype=torch.long, device=policy_device)
    group_positions = torch.tensor(group_positions, dtype=torch.long, device=policy_device)
    group_sizes = torch.tensor(group_sizes, dtype=policy_logits.dtype, device=policy_device)
    selected_positions = torch.tensor(selected_positions, dtype=torch.long, device=policy_device)

    member_logits = policy_logits[member_rows, member_actions]
    group_logits = torch.zeros(group_id, dtype=policy_logits.dtype, device=policy_device)
    group_logits.scatter_add_(0, member_group_ids, member_logits)
    group_logits = group_logits / group_sizes
    dense_logits = torch.full(
        (len(samples), maximum_group_count),
        -torch.inf,
        dtype=policy_logits.dtype,
        device=policy_device,
    )
    dense_logits[group_decisions, group_positions] = group_logits
    log_normalizers = torch.logsumexp(dense_logits, dim=1)
    selected_logits = dense_logits.gather(1, selected_positions.unsqueeze(1)).squeeze(1)
    selected_log_probabilities = selected_logits - log_normalizers
    unselected_logits = dense_logits.clone()
    unselected_logits.scatter_(1, selected_positions.unsqueeze(1), -torch.inf)
    log_unselected_probabilities = torch.logsumexp(unselected_logits, dim=1) - log_normalizers

    positive_quality = quality_signals.clamp_min(0)
    negative_quality = (-quality_signals).clamp_min(0)
    legal_group_counts = torch.tensor(
        [len(groups) for groups, _selected in structures],
        dtype=torch.long,
        device=policy_device,
    )
    has_choice = legal_group_counts > 1
    log_unselected_probabilities = torch.where(
        has_choice,
        log_unselected_probabilities,
        torch.zeros_like(log_unselected_probabilities),
    )
    losses = (
        -positive_quality * selected_log_probabilities
        - negative_quality * log_unselected_probabilities
    )
    return torch.where(has_choice, losses, torch.zeros_like(losses))


def policy_decision_loss(policy_logits, sample, quality_signal):
    """Return bounded return-weighted policy loss for one semantic decision."""
    return policy_batch_losses(
        policy_logits.unsqueeze(0),
        (sample,),
        quality_signal.reshape(1),
    )[0]


def _move_pickup_depth_diagnostic_sums(
    samples,
    selected_pickup_q,
    worst_pickup_q,
    best_pickup_q,
    best_placement_q,
    state_losses,
    depth_masks,
):
    """Return detached counterfactual base-Q and absolute-Q telemetry by depth."""
    targets = selected_pickup_q.new_tensor(
        [sample.reward_to_go if sample.reward_to_go is not None else 0.0 for sample in samples]
    )
    selected_pickup_q = selected_pickup_q.detach()
    worst_pickup_q = worst_pickup_q.detach()
    best_pickup_q = best_pickup_q.detach()
    best_placement_q = best_placement_q.detach()
    state_losses = state_losses.detach()
    residuals = selected_pickup_q - targets
    selected_is_worst = torch.isclose(selected_pickup_q, worst_pickup_q)
    hinge_active = state_losses > 0

    provenance = []
    for sample in samples:
        if sample.local_training_target == REPEATED_MOVE_LOCAL_TARGET:
            provenance.append(1)
        elif sample.local_training_target == ALL_MOVE_TURN_LOCAL_TARGET:
            provenance.append(2)
        elif sample.local_training_target is not None:
            provenance.append(4)
        elif sample.local_training_adjustment:
            provenance.append(3)
        else:
            provenance.append(0)
    provenance = torch.tensor(provenance, dtype=torch.long, device=selected_pickup_q.device)
    metrics = (
        worst_pickup_q,
        best_pickup_q,
        best_placement_q,
        selected_pickup_q,
        targets,
        residuals,
        targets > selected_pickup_q,
        targets < selected_pickup_q,
        selected_is_worst,
        hinge_active,
        hinge_active & selected_is_worst & (selected_pickup_q > targets),
        provenance == 0,
        provenance == 1,
        provenance == 2,
        provenance == 3,
        provenance == 4,
    )
    return tuple(
        tuple(metric[mask].to(torch.float32).sum() for metric in metrics) for mask in depth_masks
    )


def case_a_family_ranking_loss(
    q_values,
    samples,
    *,
    margin=CASE_A_FAMILY_RANK_MARGIN,
):
    """Rank every semantic pickup above every premature placement after pickup one."""
    samples = tuple(samples)
    if q_values.ndim != 2 or q_values.shape[0] != len(samples):
        raise ValueError("Case-A ranking requires one Q row per training sample")

    member_rows = []
    member_actions = []
    member_group_ids = []
    group_states = []
    group_sizes = []
    group_is_placement = []
    state_count = 0
    group_id = 0
    pair_pickup_group_ids = []
    pair_placement_group_ids = []
    pair_states = []
    eligible_samples = []
    selected_pickup_group_ids = []
    for row, sample in enumerate(samples):
        pickup_groups = sample.case_a_pickup_action_groups
        placement_groups = sample.case_a_placement_action_groups
        if not pickup_groups or not placement_groups:
            continue
        state_pickup_group_ids = []
        state_placement_group_ids = []
        selected_pickup_group_id = None
        for is_placement, groups in ((False, pickup_groups), (True, placement_groups)):
            for group in groups:
                member_rows.extend((row,) * len(group))
                member_actions.extend(group)
                member_group_ids.extend((group_id,) * len(group))
                group_states.append(state_count)
                group_sizes.append(len(group))
                group_is_placement.append(is_placement)
                (state_placement_group_ids if is_placement else state_pickup_group_ids).append(
                    group_id
                )
                if not is_placement and sample.action_index in group:
                    selected_pickup_group_id = group_id
                group_id += 1
        if selected_pickup_group_id is None:
            # Legacy/synthetic ranking probes may supply a non-pickup selected
            # action. Production scaffolded samples always select a pickup.
            selected_pickup_group_id = state_pickup_group_ids[0]
        eligible_samples.append(sample)
        selected_pickup_group_ids.append(selected_pickup_group_id)
        for placement_group_id in state_placement_group_ids:
            pair_pickup_group_ids.extend(state_pickup_group_ids)
            pair_placement_group_ids.extend((placement_group_id,) * len(state_pickup_group_ids))
            pair_states.extend((state_count,) * len(state_pickup_group_ids))
        state_count += 1

    if not state_count:
        zero_loss = q_values.sum() * 0.0
        zero_count = torch.zeros((), dtype=torch.long, device=q_values.device)
        return CaseAFamilyRankingResult(
            zero_loss,
            0,
            q_values.new_empty((0,)),
            zero_count,
            zero_count.clone(),
            zero_count.clone(),
            zero_loss,
            zero_count.clone(),
            zero_count.clone(),
            zero_loss,
            zero_loss,
            zero_loss,
            zero_count.clone(),
        )

    member_metadata = torch.tensor(
        (member_rows, member_actions, member_group_ids),
        dtype=torch.long,
        device=q_values.device,
    )
    group_metadata = torch.tensor(
        (group_states, group_sizes, group_is_placement),
        dtype=torch.long,
        device=q_values.device,
    )
    member_q_values = q_values[member_metadata[0], member_metadata[1]]
    group_q_values = q_values.new_zeros(group_id)
    group_q_values.scatter_add_(0, member_metadata[2], member_q_values)
    group_q_values = group_q_values / group_metadata[1]

    placement_mask = group_metadata[2].bool()
    pickup_mask = ~placement_mask
    worst_pickup_q = q_values.new_full((state_count,), torch.inf)
    worst_pickup_q.scatter_reduce_(
        0,
        group_metadata[0, pickup_mask],
        group_q_values[pickup_mask],
        reduce="amin",
        include_self=True,
    )
    best_placement_q = q_values.new_full((state_count,), -torch.inf)
    best_placement_q.scatter_reduce_(
        0,
        group_metadata[0, placement_mask],
        group_q_values[placement_mask],
        reduce="amax",
        include_self=True,
    )
    state_losses = functional.relu(best_placement_q - worst_pickup_q + float(margin))

    pair_metadata = torch.tensor(
        (pair_pickup_group_ids, pair_placement_group_ids, pair_states),
        dtype=torch.long,
        device=q_values.device,
    )
    violations = functional.relu(
        group_q_values[pair_metadata[1]] - group_q_values[pair_metadata[0]] + float(margin)
    )
    pair_counts = torch.bincount(pair_metadata[2], minlength=state_count)
    violating_counts = torch.zeros(
        state_count,
        dtype=torch.long,
        device=q_values.device,
    )
    violating_counts.scatter_add_(0, pair_metadata[2], (violations > 0).to(torch.long))

    detached_group_q = group_q_values.detach()
    best_pickup_q = q_values.new_full((state_count,), -torch.inf)
    best_pickup_q.scatter_reduce_(
        0,
        group_metadata[0, pickup_mask],
        detached_group_q[pickup_mask],
        reduce="amax",
        include_self=True,
    )
    detached_worst_pickup_q = worst_pickup_q.detach()
    detached_best_placement_q = best_placement_q.detach()
    boundary_gaps = detached_worst_pickup_q - detached_best_placement_q
    selected_pickup_q = group_q_values[
        torch.tensor(selected_pickup_group_ids, dtype=torch.long, device=q_values.device)
    ]
    depth_diagnostic_sums = _move_pickup_depth_diagnostic_sums(
        eligible_samples,
        selected_pickup_q,
        detached_worst_pickup_q,
        best_pickup_q,
        detached_best_placement_q,
        state_losses,
        (torch.ones(state_count, dtype=torch.bool, device=q_values.device),),
    )[0]
    return CaseAFamilyRankingResult(
        state_losses.mean(),
        state_count,
        state_losses,
        (violating_counts > 0).sum(),
        violating_counts.sum(),
        pair_counts.sum(),
        (violating_counts / pair_counts).sum(),
        (detached_worst_pickup_q > detached_best_placement_q).sum(),
        (best_pickup_q >= detached_best_placement_q).sum(),
        detached_worst_pickup_q.sum(),
        detached_best_placement_q.sum(),
        boundary_gaps.sum(),
        (boundary_gaps >= float(margin)).sum(),
        depth_diagnostic_sums,
    )


def move_continuation_family_ranking_loss(
    q_values,
    samples,
    *,
    margin=MOVE_CONTINUATION_FAMILY_RANK_MARGIN,
):
    """Rank every remaining pickup above every placement at pickup depths 2+."""
    samples = tuple(samples)
    if q_values.ndim != 2 or q_values.shape[0] != len(samples):
        raise ValueError("Move-continuation ranking requires one Q row per training sample")

    member_rows = []
    member_actions = []
    member_group_ids = []
    group_states = []
    group_sizes = []
    group_is_placement = []
    state_depths = []
    eligible_samples = []
    selected_pickup_group_ids = []
    state_count = 0
    group_id = 0
    for row, sample in enumerate(samples):
        pickup_groups = sample.move_continuation_pickup_action_groups
        placement_groups = sample.move_continuation_placement_action_groups
        if not pickup_groups or not placement_groups:
            continue
        state_pickup_group_ids = []
        selected_pickup_group_id = None
        for is_placement, groups in ((False, pickup_groups), (True, placement_groups)):
            for group in groups:
                member_rows.extend((row,) * len(group))
                member_actions.extend(group)
                member_group_ids.extend((group_id,) * len(group))
                group_states.append(state_count)
                group_sizes.append(len(group))
                group_is_placement.append(is_placement)
                if not is_placement:
                    state_pickup_group_ids.append(group_id)
                if not is_placement and sample.action_index in group:
                    selected_pickup_group_id = group_id
                group_id += 1
        if selected_pickup_group_id is None:
            selected_pickup_group_id = state_pickup_group_ids[0]
        eligible_samples.append(sample)
        selected_pickup_group_ids.append(selected_pickup_group_id)
        state_depths.append(sample.move_continuation_pickup_depth)
        state_count += 1

    if not state_count:
        zero_loss = q_values.sum() * 0.0
        zero_count = torch.zeros((), dtype=torch.long, device=q_values.device)
        depth_zeros = tuple(zero_loss.clone() for _depth in range(3))
        depth_zero_counts = tuple(zero_count.clone() for _depth in range(3))
        return MoveContinuationFamilyRankingResult(
            zero_loss,
            0,
            q_values.new_empty((0,)),
            zero_count,
            zero_count.clone(),
            zero_count.clone(),
            0,
            0,
            0,
            zero_count.clone(),
            zero_count.clone(),
            zero_loss,
            zero_loss,
            zero_loss,
            depth_zeros,
            depth_zero_counts,
            tuple(value.clone() for value in depth_zero_counts),
            tuple(value.clone() for value in depth_zero_counts),
            tuple(value.clone() for value in depth_zeros),
        )

    member_metadata = torch.tensor(
        (member_rows, member_actions, member_group_ids),
        dtype=torch.long,
        device=q_values.device,
    )
    group_metadata = torch.tensor(
        (group_states, group_sizes, group_is_placement),
        dtype=torch.long,
        device=q_values.device,
    )
    member_q_values = q_values[member_metadata[0], member_metadata[1]]
    group_q_values = q_values.new_zeros(group_id)
    group_q_values.scatter_add_(0, member_metadata[2], member_q_values)
    group_q_values = group_q_values / group_metadata[1]

    placement_mask = group_metadata[2].bool()
    pickup_mask = ~placement_mask
    worst_pickup_q = q_values.new_full((state_count,), torch.inf)
    worst_pickup_q.scatter_reduce_(
        0,
        group_metadata[0, pickup_mask],
        group_q_values[pickup_mask],
        reduce="amin",
        include_self=True,
    )
    best_pickup_q = q_values.new_full((state_count,), -torch.inf)
    best_pickup_q.scatter_reduce_(
        0,
        group_metadata[0, pickup_mask],
        group_q_values[pickup_mask],
        reduce="amax",
        include_self=True,
    )
    placement_states = group_metadata[0, placement_mask]
    best_placement_q = q_values.new_full((state_count,), -torch.inf)
    best_placement_q.scatter_reduce_(
        0,
        placement_states,
        group_q_values[placement_mask],
        reduce="amax",
        include_self=True,
    )
    state_losses = functional.relu(best_placement_q - worst_pickup_q + float(margin))

    detached_best_pickup_q = best_pickup_q.detach()
    detached_worst_pickup_q = worst_pickup_q.detach()
    detached_best_placement_q = best_placement_q.detach()
    boundary_gaps = detached_worst_pickup_q - detached_best_placement_q
    depth_tensor = torch.tensor(state_depths, dtype=torch.long, device=q_values.device)
    depth_masks = tuple(depth_tensor == depth for depth in (2, 3, 4))
    all_pickups_above = detached_worst_pickup_q > detached_best_placement_q
    margin_satisfied = boundary_gaps >= float(margin)
    q1_pickups = detached_best_pickup_q >= detached_best_placement_q
    selected_pickup_q = group_q_values[
        torch.tensor(selected_pickup_group_ids, dtype=torch.long, device=q_values.device)
    ]
    depth_diagnostic_sums = _move_pickup_depth_diagnostic_sums(
        eligible_samples,
        selected_pickup_q,
        detached_worst_pickup_q,
        detached_best_pickup_q,
        detached_best_placement_q,
        state_losses,
        depth_masks,
    )
    return MoveContinuationFamilyRankingResult(
        state_losses.mean(),
        state_count,
        state_losses,
        (state_losses > 0).sum(),
        (detached_best_pickup_q > detached_best_placement_q).sum(),
        q1_pickups.sum(),
        state_depths.count(2),
        state_depths.count(3),
        state_depths.count(4),
        all_pickups_above.sum(),
        margin_satisfied.sum(),
        detached_worst_pickup_q.sum(),
        detached_best_placement_q.sum(),
        boundary_gaps.sum(),
        tuple(state_losses[mask].sum() for mask in depth_masks),
        tuple(q1_pickups[mask].sum() for mask in depth_masks),
        tuple(all_pickups_above[mask].sum() for mask in depth_masks),
        tuple(margin_satisfied[mask].sum() for mask in depth_masks),
        tuple(boundary_gaps[mask].sum() for mask in depth_masks),
        depth_diagnostic_sums,
    )


def move_continuation_base_q_excluded(sample):
    """Use the existing continuation-ranking capture as the sole exclusion predicate."""
    eligible = bool(
        sample.move_continuation_pickup_action_groups
        and sample.move_continuation_placement_action_groups
    )
    if eligible and sample.move_continuation_pickup_depth not in (2, 3, 4):
        raise ValueError("Move-continuation base-Q exclusion requires Holding2, H3, or H4")
    return eligible


def move_pickup_ranking_depth_sample_counts(samples):
    """Count eligible H1-H4 ranking samples in one full effective batch."""
    counts = [0, 0, 0, 0]
    for sample in samples:
        if sample.case_a_pickup_action_groups and sample.case_a_placement_action_groups:
            counts[0] += 1
        if move_continuation_base_q_excluded(sample):
            depth = sample.move_continuation_pickup_depth
            if depth not in (2, 3, 4):
                raise ValueError("Move-continuation ranking depth must be 2, 3, or 4")
            counts[depth - 1] += 1
    return tuple(counts)


def combined_move_pickup_ranking_loss(
    case_a_result,
    continuation_result,
    q_values,
    *,
    effective_depth_counts=None,
):
    """Pool every eligible H1-H4 state into one sample-weighted mean."""
    local_depth_counts = (
        case_a_result.sample_count,
        continuation_result.depth_2_count,
        continuation_result.depth_3_count,
        continuation_result.depth_4_count,
    )
    if effective_depth_counts is None:
        effective_depth_counts = local_depth_counts
    effective_depth_counts = tuple(effective_depth_counts)
    if len(effective_depth_counts) != 4:
        raise ValueError("Move pickup-ranking requires one effective count per Holding depth")
    if any(count < 0 for count in effective_depth_counts):
        raise ValueError("Move pickup-ranking effective depth counts cannot be negative")
    if any(
        local > effective for local, effective in zip(local_depth_counts, effective_depth_counts)
    ):
        raise ValueError("Microbatch depth counts cannot exceed effective-batch depth counts")

    zero_loss = q_values.sum() * 0.0
    present_depths = tuple(count > 0 for count in effective_depth_counts)
    present_depth_count = sum(present_depths)
    if not present_depth_count:
        return MovePickupRankingLossResult(
            zero_loss,
            0,
            tuple(zero_loss.clone() for _depth in range(4)),
        )

    depth_loss_sums = (
        case_a_result.loss * case_a_result.sample_count,
        *continuation_result.depth_loss_sums,
    )
    effective_sample_count = sum(effective_depth_counts)
    depth_contributions = tuple(
        loss_sum / effective_sample_count if effective_count else zero_loss.clone()
        for loss_sum, effective_count in zip(depth_loss_sums, effective_depth_counts)
    )
    loss = sum(depth_contributions, zero_loss)
    return MovePickupRankingLossResult(
        loss,
        present_depth_count,
        depth_contributions,
    )


def pointless_final_placement_ranking_loss(
    q_values,
    samples,
    *,
    margin=POINTLESS_FINAL_PLACEMENT_RANK_MARGIN,
):
    """Rank the model's best non-pointless finish above its selected restoration."""
    samples = tuple(samples)
    if q_values.ndim != 2 or q_values.shape[0] != len(samples):
        raise ValueError("Pointless-final-placement ranking requires one Q row per sample")

    member_rows = []
    member_actions = []
    member_group_ids = []
    group_states = []
    group_sizes = []
    group_is_restorative = []
    eligible_samples = []
    state_count = 0
    group_id = 0
    for row, sample in enumerate(samples):
        restorative_group = sample.pointless_final_restorative_action_group
        nonpointless_groups = sample.pointless_final_nonpointless_action_groups
        if not restorative_group or not nonpointless_groups:
            continue
        eligible_samples.append(sample)
        for is_restorative, groups in (
            (True, (restorative_group,)),
            (False, nonpointless_groups),
        ):
            for group in groups:
                member_rows.extend((row,) * len(group))
                member_actions.extend(group)
                member_group_ids.extend((group_id,) * len(group))
                group_states.append(state_count)
                group_sizes.append(len(group))
                group_is_restorative.append(is_restorative)
                group_id += 1
        state_count += 1

    if not state_count:
        zero_loss = q_values.sum() * 0.0
        zero_count = torch.zeros((), dtype=torch.long, device=q_values.device)
        return PointlessFinalPlacementRankingResult(
            zero_loss,
            0,
            q_values.new_empty((0,)),
            zero_count,
            zero_count.clone(),
            zero_loss.detach(),
            0,
            0,
            0,
            0,
            0,
        )

    member_metadata = torch.tensor(
        (member_rows, member_actions, member_group_ids),
        dtype=torch.long,
        device=q_values.device,
    )
    group_metadata = torch.tensor(
        (group_states, group_sizes, group_is_restorative),
        dtype=torch.long,
        device=q_values.device,
    )
    member_q_values = q_values[member_metadata[0], member_metadata[1]]
    group_q_values = q_values.new_zeros(group_id)
    group_q_values.scatter_add_(0, member_metadata[2], member_q_values)
    group_q_values = group_q_values / group_metadata[1]

    restorative_mask = group_metadata[2].bool()
    nonpointless_mask = ~restorative_mask
    restorative_q = q_values.new_full((state_count,), -torch.inf)
    restorative_q.scatter_reduce_(
        0,
        group_metadata[0, restorative_mask],
        group_q_values[restorative_mask],
        reduce="amax",
        include_self=True,
    )
    best_nonpointless_q = q_values.new_full((state_count,), -torch.inf)
    best_nonpointless_q.scatter_reduce_(
        0,
        group_metadata[0, nonpointless_mask],
        group_q_values[nonpointless_mask],
        reduce="amax",
        include_self=True,
    )
    q_gaps = restorative_q - best_nonpointless_q
    state_losses = functional.relu(q_gaps + float(margin))
    detached_losses = state_losses.detach()
    return PointlessFinalPlacementRankingResult(
        state_losses.mean(),
        state_count,
        state_losses,
        (detached_losses > 0).sum(),
        (detached_losses == 0).sum(),
        q_gaps.detach().sum(),
        sum(sample.model_rank == 1 for sample in eligible_samples),
        sum(sample.pointless_final_placement_piece_count == 1 for sample in eligible_samples),
        sum(sample.pointless_final_placement_piece_count > 1 for sample in eligible_samples),
        sum(
            sample.pointless_final_placement_type == "exact_restoration"
            for sample in eligible_samples
        ),
        sum(
            sample.pointless_final_placement_type == "equivalent_rearrangement"
            for sample in eligible_samples
        ),
    )


def record_shadow_policy_metrics(metrics, q_group_scores, policy_logits, groups):
    """Compare shadow policy and Q over the same legal semantic choices."""
    metrics.record(q_group_scores, policy_logits, groups)


class SelfPlayTrainer:
    """Collect frozen Q-selected games and update both shared-model heads afterward."""

    def __init__(self, model=None, config=None):
        self.model = model or HansaNN()
        self.config = config or TrainingConfig()
        self.optimizer = self._build_optimizer()
        self.encoder = ObservationEncoder()
        self.rng = random.Random(self.config.seed)
        self.progress = TrainingProgress()
        self.loss_total = 0.0
        self.policy_loss_total = 0.0
        self.source_state_sha256 = None
        self._source_state_hash_cache = {}
        self.curriculum_state = None
        self.last_training_sample_coverage = ()
        self._last_effective_batch_case_a_metrics = None
        self._last_effective_batch_move_continuation_metrics = None
        self._last_effective_batch_base_q_metrics = None
        self._last_effective_batch_pointless_final_metrics = None

    @staticmethod
    def _source_state_signature(path):
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _source_state_hash(self, path):
        path = Path(path)
        signature = self._source_state_signature(path)
        cached = self._source_state_hash_cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        digest = _file_sha256(path)
        verified_signature = self._source_state_signature(path)
        if verified_signature != signature:
            raise OSError(f"Generated state changed while it was being hashed: {path}")
        self._source_state_hash_cache[path] = (verified_signature, digest)
        return digest

    def _build_optimizer(self):
        if not hasattr(self.model, "policy_head"):
            self._q_and_trunk_parameters = tuple(self.model.parameters())
            self._policy_parameters = ()
            return torch.optim.Adam(self._q_and_trunk_parameters, lr=self.config.learning_rate)
        self._policy_parameters = tuple(self.model.policy_head.parameters())
        policy_parameter_ids = {id(parameter) for parameter in self._policy_parameters}
        self._q_and_trunk_parameters = tuple(
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in policy_parameter_ids
        )
        return torch.optim.Adam(
            (
                {"params": self._q_and_trunk_parameters, "lr": self.config.learning_rate},
                {
                    "params": self._policy_parameters,
                    "lr": (self.config.learning_rate * self.config.policy_head_lr_multiplier),
                },
            )
        )

    def _clip_q_gradients(self):
        torch.nn.utils.clip_grad_norm_(
            self._q_and_trunk_parameters,
            self.config.max_gradient_norm,
        )

    def _accumulate_independent_losses(
        self,
        q_loss,
        policy_loss,
        scale=1.0,
        *,
        q_auxiliary_loss=None,
        q_scale=None,
    ):
        """Accumulate scaled Q and isolated-policy gradients without clipping or stepping."""
        q_objective = (scale if q_scale is None else q_scale) * q_loss
        if q_auxiliary_loss is not None:
            q_objective = q_objective + q_auxiliary_loss
        q_objective.backward()
        if self._policy_parameters and self.config.policy_loss_weight:
            (scale * self.config.policy_loss_weight * policy_loss).backward()

    def _clip_independent_gradients(self):
        self._clip_q_gradients()
        if self._policy_parameters and self.config.policy_loss_weight:
            torch.nn.utils.clip_grad_norm_(
                self._policy_parameters,
                self.config.max_gradient_norm,
            )

    def _backward_independent_losses(self, q_loss, policy_loss):
        """Backpropagate Q normally while confining policy gradients to its head."""
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulate_independent_losses(q_loss, policy_loss)
        self._clip_independent_gradients()

    def _model_outputs(self, observations, *, model=None):
        model = self.model if model is None else model
        if hasattr(model, "policy_head"):
            return model(observations)
        q_values = model(observations)
        return HansaNNOutput(q_values=q_values, policy_logits=q_values.detach())

    def _policy_trunk_gradient_scale(self):
        """Report the fixed shadow-policy isolation level used by this trainer."""
        return 0.0

    def _tier(self, number):
        return PolicyTier(
            number,
            self.config.tier_top_k[number - 1],
            self.config.tier_epsilons[number - 1],
        )

    def _assign_training_tiers(self, player_count, *, zero_epsilon=False):
        try:
            policy = self.config.tier_rosters.training_policies()[player_count]
        except KeyError as error:
            raise TrainingRunError(
                f"No training tier policy is configured for {player_count} players"
            ) from error
        numbers = list(policy.fixed_tiers)
        if policy.random_tier_pool:
            numbers.append(self.rng.choice(policy.random_tier_pool))
        self.rng.shuffle(numbers)
        tiers = tuple(self._tier(number) for number in numbers)
        if not zero_epsilon:
            return tiers
        return tuple(replace(tier, epsilon=0.0) for tier in tiers)

    def _assign_evaluation_tiers(self, player_count, rotation):
        try:
            numbers = list(self.config.tier_rosters.evaluation_rosters()[player_count])
        except KeyError as error:
            raise TrainingRunError(
                f"No evaluation tier roster is configured for {player_count} players"
            ) from error
        offset = rotation % player_count
        numbers = numbers[offset:] + numbers[:offset]
        return tuple(replace(self._tier(number), epsilon=0.0) for number in numbers)

    @staticmethod
    def _rank_legal_positions(legal_scores, count):
        """Rank the small set of semantic choices deterministically."""
        return tuple(
            sorted(
                range(len(legal_scores)),
                key=lambda index: (-legal_scores[index], index),
            )[:count]
        )

    @staticmethod
    def _group_mean_scores(scores, groups):
        """Transfer the model output once, then average equivalent interactions."""
        values = scores.tolist()
        return tuple(sum(values[index] for index in group) / len(group) for group in groups)

    @staticmethod
    def _model_rank(group_scores, selected_position):
        selected_score = group_scores[selected_position]
        return (
            sum(score > selected_score for score in group_scores)
            + sum(score == selected_score for score in group_scores[:selected_position])
            + 1
        )

    @staticmethod
    def _validate_action_groups(legal_indices, groups, description):
        grouped_indices = [index for group in groups for index in group]
        if len(grouped_indices) != len(legal_indices) or set(grouped_indices) != set(legal_indices):
            raise ValueError(f"{description} must contain every legal action exactly once")

    def _select_action(
        self,
        scores,
        legal_indices,
        tier,
        equivalent_groups=None,
        semantic_q_scores=None,
    ):
        legal_list = _action_index_tuple(legal_indices)
        groups = (
            tuple((index,) for index in legal_list)
            if equivalent_groups is None
            else equivalent_groups
        )
        self._validate_action_groups(legal_list, groups, "Action groups")
        group_count = len(groups)
        group_scores = (
            self._group_mean_scores(scores, groups)
            if semantic_q_scores is None
            else tuple(semantic_q_scores)
        )
        if len(group_scores) != group_count:
            raise ValueError("Semantic Q scores must match the supplied action groups")
        if group_count == 1:
            group = groups[0]
            selected = group[0] if len(group) == 1 else group[self.rng.randrange(len(group))]
            return ActionSelection(
                selected,
                False,
                1,
                1,
                group,
                group_scores,
            )
        if self.rng.random() < tier.epsilon:
            selected_position = self.rng.randrange(group_count)
            used_epsilon = True
            model_rank = self._model_rank(group_scores, selected_position)
        else:
            effective_k = min(tier.top_k or group_count, group_count)
            ranked_positions = self._rank_legal_positions(group_scores, effective_k)
            selected_rank = self.rng.choices(
                range(effective_k),
                weights=normalized_rank_weights(effective_k),
                k=1,
            )[0]
            selected_position = ranked_positions[selected_rank]
            used_epsilon = False
            model_rank = selected_rank + 1
        selected_group = groups[selected_position]
        selected = (
            selected_group[0]
            if len(selected_group) == 1
            else selected_group[self.rng.randrange(len(selected_group))]
        )
        return ActionSelection(
            selected,
            used_epsilon,
            model_rank,
            group_count,
            selected_group,
            group_scores,
        )

    def _select_workflow_action(
        self,
        scores,
        legal_indices,
        tier,
        exploration_categories=None,
        semantic_q_scores=None,
    ):
        """Select a grouped workflow choice through the tier's normal policy."""
        equivalent_groups = (
            None
            if exploration_categories is None
            else tuple(group for category in exploration_categories for group in category)
        )
        return self._select_action(
            scores,
            legal_indices,
            tier,
            equivalent_groups,
            semantic_q_scores,
        )

    def _complete_trajectory(
        self,
        decisions,
        terminal_rewards,
        final_scores,
        winner_indices,
        action_trace,
        seat_tiers,
        *,
        reason="normal",
        completed=True,
        timings=None,
        movement_metrics=None,
        shadow_policy_metrics=None,
        shadow_filter_audit=None,
        training_exploration_mode=NORMAL_EXPLORATION_MODE,
    ):
        movement_metrics = movement_metrics or MovementBehaviorMetrics()
        shadow_policy_metrics = shadow_policy_metrics or ShadowPolicyMetrics()
        policy_averages = shadow_policy_metrics.averages()
        reward_to_go_decisions = assign_reward_to_go(
            decisions,
            terminal_rewards,
            self.config.gamma,
        )
        training_decisions = apply_local_training_targets(reward_to_go_decisions)
        record_applied_move1_utilization_penalties(movement_metrics, training_decisions)
        shadow_filter_records = (
            shadow_filter_audit.flagged_outcomes(
                reward_to_go_decisions,
                training_decisions,
                terminal_rewards,
                final_scores,
                winner_indices,
            )
            if shadow_filter_audit is not None
            else ()
        )
        scaffold_margins = movement_metrics.move1_scaffold_best_pickup_minus_best_placement
        scaffold_margin_mean = (
            sum(scaffold_margins) / len(scaffold_margins) if scaffold_margins else None
        )
        trajectory = CompletedTrajectory(
            training_decisions,
            tuple(terminal_rewards),
            tuple(final_scores),
            tuple(winner_indices),
            tuple(action_trace),
            tuple(tier.number for tier in seat_tiers),
            reason,
            *(timings or (0.0,) * 10),
            movement_metrics.move_action_count,
            movement_metrics.spent_action_count,
            movement_metrics.move_ratio,
            movement_metrics.pointless_move_workflows,
            movement_metrics.repeated_move_penalties,
            movement_metrics.all_move_turn_penalties,
            movement_metrics.moves_creating_claimable_route,
            movement_metrics.move_claim_conversions,
            movement_metrics.move_claim_conversion_rate,
            training_exploration_mode,
            *policy_averages,
            shadow_filter_records,
            len(shadow_filter_records),
            sum(record.used_epsilon for record in shadow_filter_records),
            movement_metrics.pointless_normal_move_workflows,
            movement_metrics.pointless_move_any2_workflows,
            movement_metrics.immediate_one_piece_q_undos,
            movement_metrics.immediate_q_undo_epsilon_pickups,
            movement_metrics.immediate_q_undo_ranked_pickups,
            movement_metrics.immediate_q_undo_q1_pickups,
            movement_metrics.immediate_q1_restores,
            movement_metrics.immediate_q1_pickup_q1_restores,
            movement_metrics.immediate_valuable_q1_pickup_q1_restores,
            movement_metrics.full_multi_piece_q_undos,
            movement_metrics.full_multi_piece_q_undo_any_exploration,
            movement_metrics.full_multi_piece_q_undo_entirely_ranked,
            movement_metrics.normal_move_nominal_capacity_total,
            movement_metrics.normal_move_movable_pieces_available_total,
            movement_metrics.normal_move_effective_capacity_total,
            movement_metrics.normal_move_pieces_moved_total,
            movement_metrics.normal_move_unused_capacity_total,
            movement_metrics.single_piece_moves,
            movement_metrics.single_piece_moves_with_multiple_available,
            movement_metrics.single_piece_moves_creating_claimable_route,
            movement_metrics.single_piece_move_claim_conversions,
            movement_metrics.full_effective_capacity_moves,
            movement_metrics.under_effective_capacity_moves,
            movement_metrics.move1_utilization_penalties_applied,
            movement_metrics.move1_penalties_on_placement,
            movement_metrics.move1_penalties_on_single_available_initiation,
            movement_metrics.move_claim_reward_awarded,
            movement_metrics.move_claim_reward_blocked_already_claimable,
            movement_metrics.consecutive_move1_pairs,
            movement_metrics.consecutive_move1_pairs_with_multiple_available,
            movement_metrics.consecutive_move1_pairs_move2_capacity,
            movement_metrics.consecutive_move1_pairs_move3_capacity,
            movement_metrics.consecutive_move1_pairs_move4_capacity,
            movement_metrics.consecutive_move1_pairs_move5_capacity,
            movement_metrics.avoidable_extra_move_actions,
            movement_metrics.move1_scaffold_mask_states,
            movement_metrics.move1_scaffold_masked_placement_semantic_actions,
            movement_metrics.move1_scaffold_legal_pickup_semantic_actions,
            movement_metrics.move1_scaffold_unmasked_q1_pickups,
            movement_metrics.move1_scaffold_unmasked_q1_placements,
            movement_metrics.move1_scaffold_unmasked_q1_pickup_fraction,
            movement_metrics.move1_scaffold_unmasked_q1_placement_fraction,
            movement_metrics.move1_scaffold_all_pickups_above_all_placements_fraction,
            movement_metrics.move1_scaffold_margin_satisfied_pair_fraction,
            movement_metrics.move1_scaffold_family_ranking_loss,
            scaffold_margin_mean,
            _percentile(scaffold_margins, 0.5) if scaffold_margins else None,
            _percentile(scaffold_margins, 0.1) if scaffold_margins else None,
            _percentile(scaffold_margins, 0.9) if scaffold_margins else None,
            movement_metrics.move1_scaffold_unmasked_top_k_pickup_fraction,
            movement_metrics.move1_scaffold_unmasked_top_k_has_pickup_fraction,
            *(getattr(movement_metrics, field) for field in NORMAL_MOVE_CAPACITY_TELEMETRY_FIELDS),
            pointless_move_attribution=(
                movement_metrics.pointless_move_attribution.as_dict()
                if self.config.pointless_move_attribution_audit_enabled
                else {}
            ),
        )
        if completed:
            self.progress.completed_games += 1
        self.progress.decisions += len(decisions)
        self._record_tier_metrics(trajectory)
        return trajectory

    def collect_game(
        self,
        starting_state,
        *,
        quiet=True,
        failure_callback=None,
        evaluation=False,
        evaluation_tier_rotation=0,
        capture_action_limit=False,
        zero_epsilon=False,
        shadow_filter_audit=None,
        evaluation_models_by_seat=None,
    ) -> CompletedTrajectory:
        """Play one exact starting state without changing model weights."""
        if evaluation_models_by_seat is not None and not evaluation:
            raise ValueError("Per-seat model overrides are restricted to evaluation games")
        if shadow_filter_audit is None:
            shadow_filter_audit = self.config.shadow_filter_audit_enabled
        play_started = perf_counter()
        detailed_profiling = self.config.detailed_profiling
        inference_seconds = 0.0
        scoring_seconds = 0.0
        execution_seconds = 0.0
        validation_seconds = 0.0
        observation_seconds = 0.0
        legality_seconds = 0.0
        selection_seconds = 0.0
        context_seconds = 0.0
        reward_seconds = 0.0

        def timings():
            return (
                perf_counter() - play_started,
                inference_seconds,
                scoring_seconds,
                execution_seconds,
                validation_seconds,
                observation_seconds,
                legality_seconds,
                selection_seconds,
                context_seconds,
                reward_seconds,
            )

        game = load_game(starting_state)
        game.set_interactive_errors(False)
        if evaluation_models_by_seat is not None:
            evaluation_models_by_seat = tuple(evaluation_models_by_seat)
            if len(evaluation_models_by_seat) != len(game.players):
                raise ValueError("Evaluation requires exactly one model for each player seat")
            for evaluation_model in evaluation_models_by_seat:
                evaluation_model.eval()
        post_contexts = _post_contexts_by_slot(game)
        post_routes = {post: route for _route_index, route, post in post_contexts}
        post_route_indices = {post: route_index for route_index, _route, post in post_contexts}
        seat_tiers = (
            self._assign_evaluation_tiers(len(game.players), evaluation_tier_rotation)
            if evaluation
            else self._assign_training_tiers(len(game.players), zero_epsilon=zero_epsilon)
        )
        training_exploration_mode = (
            ZERO_EPSILON_EXPLORATION_MODE
            if zero_epsilon and not evaluation
            else NORMAL_EXPLORATION_MODE
        )
        decisions = []
        action_trace = []
        game_end_trigger_player = None
        pending_disruption = None
        tracked_turn = game.turn_number
        tracked_turn_player = game.current_player
        movement_metrics = MovementBehaviorMetrics()
        pointless_move_audit_enabled = self.config.pointless_move_attribution_audit_enabled
        shadow_policy_metrics = ShadowPolicyMetrics()
        collect_shadow_filter = shadow_filter_audit and not evaluation
        shadow_filter_metrics = ShadowFilterAudit() if collect_shadow_filter else None
        for player in game.players:
            player.pending_move_claim_route_slots = frozenset()
            player.rewarded_move_focus_route_slots = frozenset()
        pending_terminal_move_workflows = []
        pending_terminal_completed_routes = set()
        turn_move_workflow_ids = []
        move_destination_counts = {}
        move_blocked_next_player = False
        move_completed_routes_before = set()
        move_tracking_active = False
        move_pieces_picked_up = 0
        move_origin_posts = []
        move_origin_pieces = []
        move_destination_posts = []
        permanent_move_tracking_active = False
        next_movement_workflow_id = 1
        normal_move_workflow_id = None
        permanent_move_workflow_id = None
        normal_move_selections = []
        normal_move_valuable_origin = False
        normal_move_nominal_capacity = 0
        normal_move_movable_pieces_available = 0
        normal_move_legal_pickups_at_start = 0
        normal_move_effective_capacity = 0
        normal_move_initial_pickup_post_slot = None
        normal_move_additional_pickup_post_slots = frozenset()
        previous_paid_move1 = None
        previous_paid_move1_workflow_id = None
        move_observed_pickup_routes = set()
        already_claimable_pickup_routes = set()
        pending_single_piece_move_claim_routes = frozenset()
        pending_blocked_move_claim_routes = frozenset()
        loaded_move = loaded_normal_move_context(game, post_contexts)
        if loaded_move is not None:
            normal_move_workflow_id = next_movement_workflow_id
            next_movement_workflow_id += 1
            move_tracking_active = True
            move_pieces_picked_up = len(game.current_player.holding_pieces)
            move_origin_posts = list(loaded_move.origin_posts)
            move_origin_pieces = list(loaded_move.origin_pieces)
            move_completed_routes_before = set(loaded_move.completed_route_slots_before)
            move_observed_pickup_routes = set(loaded_move.observed_pickup_route_slots)
            already_claimable_pickup_routes = set(loaded_move.already_claimable_pickup_route_slots)
            normal_move_valuable_origin = bool(already_claimable_pickup_routes)
            normal_move_nominal_capacity = game.current_player.book
            normal_move_movable_pieces_available = loaded_move.movable_pieces_at_start
            normal_move_legal_pickups_at_start = loaded_move.movable_pieces_at_start
            normal_move_effective_capacity = min(
                normal_move_nominal_capacity,
                normal_move_movable_pieces_available,
            )
            normal_move_initial_pickup_post_slot = loaded_move.initial_pickup_post_slot
        output = redirect_stdout(io.StringIO()) if quiet else nullcontext()
        if detailed_profiling:
            scoring_started = perf_counter()
        projected_before = game.projected_scores()
        if detailed_profiling:
            scoring_seconds += perf_counter() - scoring_started

        self.model.eval()
        with output, torch.inference_mode():
            for action_number in range(1, self.config.max_actions + 1):
                if game.game_end:
                    break
                if game.turn_number != tracked_turn:
                    finalize_all_move_turn(
                        decisions,
                        movement_metrics,
                        turn_move_workflow_ids,
                        tracked_turn_player.paid_actions_spent_this_turn,
                        pointless_move_audit_enabled=pointless_move_audit_enabled,
                    )
                    tracked_turn = game.turn_number
                    tracked_turn_player = game.current_player
                    for player in game.players:
                        player.pending_move_claim_route_slots = frozenset()
                    pending_terminal_move_workflows = []
                    pending_terminal_completed_routes = set()
                    turn_move_workflow_ids = []
                    move_destination_counts = {}
                    move_blocked_next_player = False
                    move_completed_routes_before = set()
                    move_tracking_active = False
                    move_pieces_picked_up = 0
                    move_origin_posts = []
                    move_origin_pieces = []
                    move_destination_posts = []
                    permanent_move_tracking_active = False
                    normal_move_workflow_id = None
                    permanent_move_workflow_id = None
                    normal_move_selections = []
                    normal_move_valuable_origin = False
                    normal_move_nominal_capacity = 0
                    normal_move_movable_pieces_available = 0
                    normal_move_legal_pickups_at_start = 0
                    normal_move_effective_capacity = 0
                    normal_move_initial_pickup_post_slot = None
                    normal_move_additional_pickup_post_slots = frozenset()
                    previous_paid_move1 = None
                    previous_paid_move1_workflow_id = None
                    move_observed_pickup_routes = set()
                    already_claimable_pickup_routes = set()
                    pending_single_piece_move_claim_routes = frozenset()
                    pending_blocked_move_claim_routes = frozenset()
                action_attempted = False
                try:
                    if detailed_profiling:
                        observation_started = perf_counter()
                    observation = self.encoder.build(game)
                    if detailed_profiling:
                        observation_seconds += perf_counter() - observation_started
                        legality_started = perf_counter()
                    mask = training_action_mask(
                        game,
                        disable_move_action=self.config.disable_move_action,
                        move_general_stock_threshold=self.config.move_general_stock_threshold,
                        base_mask=observation.legal_action_mask,
                        post_contexts=post_contexts,
                    )
                    legal_indices = mask.nonzero(as_tuple=False).flatten()
                    if detailed_profiling:
                        legality_seconds += perf_counter() - legality_started
                    if legal_indices.numel() == 0:
                        self.progress.game_completion_failures += 1
                        if (
                            game.turn_phase == TurnPhase.REPLACE_BONUS_MARKERS
                            and game.replace_bonus_marker > 0
                        ):
                            self.progress.replacement_route_deadlocks += 1
                            error = TrainingRunError(
                                "No route can receive the pending replacement bonus marker"
                            )
                            if failure_callback is not None:
                                failure_callback(game, tuple(action_trace), seat_tiers, error)
                            terminal_rewards = [0.0] * len(game.players)
                            terminal_rewards[observation.observer_index] = (
                                NO_REPLACEMENT_ROUTE_PENALTY
                            )
                            return self._complete_trajectory(
                                decisions,
                                terminal_rewards,
                                projected_before,
                                (),
                                action_trace,
                                seat_tiers,
                                reason="no_replacement_route",
                                completed=False,
                                timings=timings(),
                                movement_metrics=movement_metrics,
                                shadow_policy_metrics=shadow_policy_metrics,
                                shadow_filter_audit=shadow_filter_metrics,
                                training_exploration_mode=training_exploration_mode,
                            )
                        error = IncompleteGameError(
                            "The game has no legal interaction at "
                            f"turn {game.turn_number}, phase {game.turn_phase.value}"
                        )
                        raise error
                    legal_action_indices = _action_index_tuple(legal_indices)
                    if detailed_profiling:
                        inference_started = perf_counter()
                    model_output = self._model_outputs(
                        observation.features.float().unsqueeze(0).to(device),
                        model=(
                            None
                            if evaluation_models_by_seat is None
                            else evaluation_models_by_seat[observation.observer_index]
                        ),
                    )
                    scores = model_output.q_values[0]
                    policy_logits = model_output.policy_logits[0]
                    if detailed_profiling:
                        inference_seconds += perf_counter() - inference_started
                        selection_started = perf_counter()
                    tier = seat_tiers[observation.observer_index]
                    training_legal_action_mask = mask
                    training_semantic_action_groups = None
                    case_a_pickup_action_groups = ()
                    case_a_placement_action_groups = ()
                    move_continuation_pickup_action_groups = ()
                    move_continuation_placement_action_groups = ()
                    move_continuation_pickup_depth = 0
                    normal_move_legal_pickups_at_decision = frozenset()
                    pointless_final_restorative_action_group = ()
                    pointless_final_nonpointless_action_groups = ()
                    pointless_final_selected_type = None
                    scaffold_semantic_q_scores = None
                    if game.turn_phase is TurnPhase.ACTIONS:
                        equivalent_groups = action_phase_selection_groups(
                            game,
                            legal_action_indices,
                            post_contexts,
                        )
                        semantic_action_groups = equivalent_groups or tuple(
                            (index,) for index in legal_action_indices
                        )
                        training_semantic_action_groups = semantic_action_groups
                        selection = self._select_action(
                            scores,
                            legal_action_indices,
                            tier,
                            equivalent_groups,
                        )
                    else:
                        if game.turn_phase is TurnPhase.MOVE_PIECES:
                            # Capture this from the unmodified legal mask so final-placement
                            # eligibility never depends on the temporary pickup scaffold.
                            normal_move_legal_pickups_at_decision = (
                                legal_normal_move_pickup_post_slots(
                                    legal_action_indices,
                                    post_contexts,
                                    game.current_player,
                                )
                            )
                            exploration_categories = move_workflow_exploration_categories(
                                game,
                                legal_action_indices,
                                post_contexts=post_contexts,
                            )
                            (
                                case_a_pickup_action_groups,
                                case_a_placement_action_groups,
                            ) = case_a_move_action_families(
                                normal_move_workflow_id,
                                len(game.current_player.holding_pieces),
                                bool(move_destination_posts),
                                normal_move_effective_capacity,
                                exploration_categories,
                            )
                            (
                                move_continuation_pickup_action_groups,
                                move_continuation_placement_action_groups,
                            ) = move_continuation_action_families(
                                normal_move_workflow_id,
                                len(game.current_player.holding_pieces),
                                bool(move_destination_posts),
                                normal_move_effective_capacity,
                                exploration_categories,
                            )
                            if move_continuation_pickup_action_groups:
                                move_continuation_pickup_depth = len(
                                    game.current_player.holding_pieces
                                )
                            (
                                scaffold_pickup_action_groups,
                                scaffold_placement_action_groups,
                            ) = training_move_scaffold_action_families(
                                normal_move_workflow_id,
                                len(game.current_player.holding_pieces),
                                bool(move_destination_posts),
                                normal_move_effective_capacity,
                                exploration_categories,
                                evaluation=evaluation,
                            )
                            if scaffold_pickup_action_groups and scaffold_placement_action_groups:
                                pre_scaffold_groups = (
                                    scaffold_pickup_action_groups + scaffold_placement_action_groups
                                )
                                training_semantic_action_groups = pre_scaffold_groups
                                pre_scaffold_q_scores = self._group_mean_scores(
                                    scores,
                                    pre_scaffold_groups,
                                )
                                if len(game.current_player.holding_pieces) == 1:
                                    record_move1_scaffold_readiness(
                                        movement_metrics,
                                        pre_scaffold_q_scores,
                                        len(scaffold_pickup_action_groups),
                                        tier.top_k,
                                    )
                                mask = move1_scaffold_action_mask(
                                    mask,
                                    scaffold_pickup_action_groups,
                                    scaffold_placement_action_groups,
                                )
                                legal_indices = mask.nonzero(as_tuple=False).flatten()
                                legal_action_indices = _action_index_tuple(legal_indices)
                                exploration_categories = (scaffold_pickup_action_groups,)
                                scaffold_semantic_q_scores = pre_scaffold_q_scores[
                                    : len(scaffold_pickup_action_groups)
                                ]
                        elif game.turn_phase is TurnPhase.BONUS_MARKER_CHOICE and (
                            game.waiting_for_bm_move3 or game.waiting_for_bm_move_any_2
                        ):
                            exploration_categories = move_workflow_exploration_categories(
                                game,
                                legal_action_indices,
                                opponent_pickups=game.waiting_for_bm_move3,
                                any_pickups=game.waiting_for_bm_move_any_2,
                                post_contexts=post_contexts,
                            )
                        else:
                            exploration_categories = None
                        semantic_action_groups = (
                            tuple(
                                group for category in exploration_categories for group in category
                            )
                            if exploration_categories is not None
                            else tuple((index,) for index in legal_action_indices)
                        )
                        if training_semantic_action_groups is None:
                            training_semantic_action_groups = semantic_action_groups
                        selection = self._select_workflow_action(
                            scores,
                            legal_action_indices,
                            tier,
                            exploration_categories,
                            scaffold_semantic_q_scores,
                        )
                    if game.turn_phase is TurnPhase.ACTIONS:
                        record_shadow_policy_metrics(
                            shadow_policy_metrics,
                            selection.semantic_q_scores,
                            policy_logits,
                            semantic_action_groups,
                        )
                    if collect_shadow_filter:
                        shadow_filter_metrics.record(
                            len(decisions),
                            selection,
                            policy_logits,
                            semantic_action_groups,
                        )
                    action_index = selection.action_index
                    action = _ACTIONS_BY_INDEX[action_index]
                    if detailed_profiling:
                        selection_seconds += perf_counter() - selection_started
                        context_started = perf_counter()
                    action_phase = game.turn_phase
                    acting_player = game.players[observation.observer_index]
                    context = (
                        post_contexts[action.post_slot]
                        if isinstance(action, PostInteraction)
                        else None
                    )
                    if action_phase is TurnPhase.ACTIONS and context is not None:
                        route_index, _route, selected_post = context
                        next_player_index = (observation.observer_index + 1) % len(game.players)
                        next_player = game.players[next_player_index]
                        if selected_post.owner is next_player:
                            valuable_routes = valuable_completed_route_slots(game, next_player)
                            if route_index in valuable_routes:
                                pending_disruption = (
                                    observation.observer_index,
                                    next_player_index,
                                    len(valuable_routes),
                                )
                    bank_capacity = acting_player.bank
                    normal_move_in_progress = _is_normal_move_in_progress(
                        action_phase, acting_player
                    )
                    permanent_move_in_progress = bool(
                        action_phase is TurnPhase.BONUS_MARKER_CHOICE
                        and game.waiting_for_bm_move_any_2
                    )
                    starts_normal_move = bool(
                        action_phase is TurnPhase.ACTIONS
                        and not acting_player.holding_pieces
                        and isinstance(action, PostInteraction)
                        and context is not None
                        and context[2].owner is acting_player
                    )
                    if starts_normal_move:
                        normal_move_workflow_id = next_movement_workflow_id
                        next_movement_workflow_id += 1
                        normal_move_selections = []
                        origin_route_index, origin_route, _origin_post = context
                        move_observed_pickup_routes = set()
                        already_claimable_pickup_routes = set()
                        record_move_pickup_route_claimability(
                            move_observed_pickup_routes,
                            already_claimable_pickup_routes,
                            origin_route_index,
                            origin_route.is_controlled_by(acting_player),
                        )
                        normal_move_valuable_origin = bool(
                            origin_route.is_controlled_by(acting_player)
                            or origin_route_index in acting_player.rewarded_move_focus_route_slots
                        )
                        normal_move_nominal_capacity = acting_player.book
                        normal_move_movable_pieces_available = sum(
                            post.owner is acting_player
                            for _route_index, _route, post in post_contexts
                        )
                        normal_move_legal_pickups_at_start = len(
                            legal_normal_move_pickup_post_slots(
                                legal_action_indices,
                                post_contexts,
                                acting_player,
                            )
                        )
                        normal_move_effective_capacity = min(
                            normal_move_nominal_capacity,
                            normal_move_legal_pickups_at_start,
                        )
                        normal_move_initial_pickup_post_slot = action.post_slot
                    if permanent_move_in_progress and permanent_move_workflow_id is None:
                        permanent_move_workflow_id = next_movement_workflow_id
                        next_movement_workflow_id += 1
                    movement_workflow_id = (
                        normal_move_workflow_id
                        if starts_normal_move or normal_move_in_progress
                        else permanent_move_workflow_id
                        if permanent_move_in_progress
                        else None
                    )
                    movement_role = None
                    if starts_normal_move:
                        movement_role = "initial_pickup"
                    elif (
                        normal_move_in_progress
                        and isinstance(action, PostInteraction)
                        and context is not None
                    ):
                        selected_movement_post = context[2]
                        if selected_movement_post.owner is acting_player:
                            movement_role = "additional_pickup"
                        elif not selected_movement_post.is_owned():
                            movement_role = (
                                "final_placement"
                                if len(acting_player.holding_pieces) == 1
                                else "intermediate_placement"
                            )
                    if movement_role is not None:
                        if movement_role in {
                            "intermediate_placement",
                            "final_placement",
                        } and not any(
                            prior.role in {"intermediate_placement", "final_placement"}
                            for prior in normal_move_selections
                        ):
                            normal_move_additional_pickup_post_slots = (
                                legal_normal_move_pickup_post_slots(
                                    legal_action_indices, post_contexts, acting_player
                                )
                            )
                        normal_move_selections.append(
                            MoveSelectionTelemetry(
                                movement_role,
                                selection.used_epsilon,
                                selection.model_rank,
                            )
                        )
                    movement_capacity = acting_player.book
                    pieces_moved = move_pieces_picked_up
                    actions_remaining_before = acting_player.actions_remaining
                    move_placement_route = None
                    move_placement_post = None
                    movement_destination_routes = frozenset()
                    move_blocks_next_player = False
                    route_building_reward = 0.0
                    route_building_post = None
                    if (
                        action_phase is TurnPhase.ACTIONS
                        and not acting_player.holding_pieces
                        and isinstance(action, PostInteraction)
                        and context is not None
                    ):
                        _route_index, route, selected_post = context
                        if selected_post.owner is not acting_player:
                            route_building_reward = route_building_post_reward(
                                route_already_has_piece=any(
                                    post.owner is acting_player for post in route.posts
                                ),
                                is_displacement=selected_post.is_owned(),
                            )
                            route_building_post = selected_post
                    if (
                        normal_move_in_progress
                        and isinstance(action, PostInteraction)
                        and context is not None
                    ):
                        route_index, route, selected_post = context
                        if not selected_post.is_owned():
                            move_placement_route = route_index
                            move_placement_post = selected_post
                            next_player = game.players[
                                (observation.observer_index + 1) % len(game.players)
                            ]
                            move_blocks_next_player = bool(route.posts) and all(
                                post is selected_post or post.owner is next_player
                                for post in route.posts
                            )
                        elif selected_post.owner is acting_player:
                            record_move_pickup_route_claimability(
                                move_observed_pickup_routes,
                                already_claimable_pickup_routes,
                                route_index,
                                route.is_controlled_by(acting_player),
                            )
                            move_origin_posts.append(selected_post)
                            move_origin_pieces.append(
                                (
                                    selected_post,
                                    selected_post.owner,
                                    selected_post.owner_piece_shape,
                                )
                            )
                    elif (
                        permanent_move_in_progress
                        and isinstance(action, PostInteraction)
                        and context is not None
                    ):
                        _route_index, _route, selected_post = context
                        if selected_post.is_owned():
                            if not permanent_move_tracking_active:
                                move_origin_posts = []
                                move_origin_pieces = []
                                move_destination_posts = []
                                permanent_move_tracking_active = True
                            move_origin_posts.append(selected_post)
                            move_origin_pieces.append(
                                (
                                    selected_post,
                                    selected_post.owner,
                                    selected_post.owner_piece_shape,
                                )
                            )
                        elif acting_player.holding_pieces:
                            move_placement_post = selected_post
                    elif (
                        action_phase is TurnPhase.ACTIONS
                        and not acting_player.holding_pieces
                        and isinstance(action, PostInteraction)
                        and context is not None
                        and context[2].owner is acting_player
                    ):
                        move_destination_counts = {}
                        move_completed_routes_before = {
                            route_index
                            for route_index, route in enumerate(game.selected_map.routes)
                            if route.is_controlled_by(acting_player)
                        }
                        move_tracking_active = True
                        move_pieces_picked_up = 0
                        move_origin_posts = [selected_post]
                        move_origin_pieces = [
                            (
                                selected_post,
                                selected_post.owner,
                                selected_post.owner_piece_shape,
                            )
                        ]
                        move_destination_posts = []
                    if pointless_final_placement_lesson_eligible(
                        evaluation=evaluation,
                        normal_move_in_progress=normal_move_in_progress,
                        selected_empty_placement=move_placement_post is not None,
                        held_piece_count=len(acting_player.holding_pieces),
                        legal_pickup_post_slots=normal_move_legal_pickups_at_decision,
                    ):
                        (
                            pointless_final_restorative_action_group,
                            pointless_final_nonpointless_action_groups,
                            pointless_final_selected_type,
                        ) = pointless_final_placement_action_groups(
                            move_origin_pieces,
                            move_destination_posts,
                            post_routes,
                            post_contexts,
                            semantic_action_groups,
                            action_index,
                        )
                    general_stock_before = (
                        acting_player.general_stock_squares + acting_player.general_stock_circles
                    )
                    score_before = acting_player.score
                    office_count_before = sum(
                        office.controller is acting_player
                        for city in game.selected_map.cities
                        for office in city.offices
                    )
                    bonus_marker_count_before = len(acting_player.bonus_markers) + len(
                        acting_player.used_bonus_markers
                    )
                    route_had_permanent_marker = bool(
                        action_phase is TurnPhase.ACTIONS
                        and isinstance(action, RouteInteraction)
                        and game.selected_map.routes[action.route_slot].permanent_bonus_marker
                    )
                    abilities_before = tuple(
                        acting_player.actions_index
                        if ability == "actions"
                        else getattr(acting_player, ability)
                        for ability in INTERMEDIATE_REWARDED_ABILITIES
                    )
                    if detailed_profiling:
                        context_seconds += perf_counter() - context_started
                    end_was_pending = game.game_end or game.game_end_pending_immediate_resolution
                    turn_before = game.turn_number
                    action_trace.append(action_index)
                    action_attempted = True
                    if detailed_profiling:
                        execution_started = perf_counter()
                    game._apply_prevalidated_ai_action(action_index, mask)
                    if move_tracking_active:
                        move_pieces_picked_up = max(
                            move_pieces_picked_up, len(acting_player.holding_pieces)
                        )
                    if move_placement_post is not None and move_placement_post.is_owned():
                        move_destination_posts.append(move_placement_post)
                    if detailed_profiling:
                        execution_seconds += perf_counter() - execution_started
                    if should_fully_validate(
                        action_number,
                        self.config.full_validation_interval,
                        turn_before,
                        action_phase,
                        game,
                    ):
                        if detailed_profiling:
                            validation_started = perf_counter()
                        validate_game(game)
                        if detailed_profiling:
                            validation_seconds += perf_counter() - validation_started
                except Exception as error:
                    if action_attempted:
                        self.progress.invalid_action_attempts += 1
                    if failure_callback is not None:
                        failure_callback(game, tuple(action_trace), seat_tiers, error)
                    raise
                if detailed_profiling:
                    scoring_started = perf_counter()
                projected_after = game.projected_scores()
                if detailed_profiling:
                    scoring_seconds += perf_counter() - scoring_started
                    reward_started = perf_counter()
                score_reward_deltas = tuple(
                    float(PRESTIGE_REWARD_MULTIPLIER * (after - before))
                    for before, after in zip(projected_before, projected_after)
                )
                projected_before = projected_after
                general_stock_after = (
                    acting_player.general_stock_squares + acting_player.general_stock_circles
                )
                player_reward_deltas = apply_income_efficiency_penalty(
                    score_reward_deltas,
                    action=action,
                    turn_phase=action_phase,
                    acting_player_index=observation.observer_index,
                    bank_capacity=bank_capacity,
                    pieces_received=general_stock_before - general_stock_after,
                    scale=self.config.income_penalty_scale,
                )
                abilities_after = tuple(
                    acting_player.actions_index
                    if ability == "actions"
                    else getattr(acting_player, ability)
                    for ability in INTERMEDIATE_REWARDED_ABILITIES
                )
                intermediate_upgrade_reward = intermediate_ability_upgrade_reward(
                    abilities_before, abilities_after
                )
                if intermediate_upgrade_reward:
                    adjusted = list(player_reward_deltas)
                    adjusted[observation.observer_index] += intermediate_upgrade_reward
                    player_reward_deltas = tuple(adjusted)
                office_count_after = sum(
                    office.controller is acting_player
                    for city in game.selected_map.cities
                    for office in city.offices
                )
                bonus_marker_count_after = len(acting_player.bonus_markers) + len(
                    acting_player.used_bonus_markers
                )
                route_claim_penalty = pointless_route_claim_penalty(
                    action=action,
                    turn_phase=action_phase,
                    action_was_spent=acting_player.actions_remaining < actions_remaining_before,
                    gained_office=office_count_after > office_count_before,
                    gained_upgrade=abilities_after != abilities_before,
                    gained_marker=bonus_marker_count_after > bonus_marker_count_before,
                    gained_points=(
                        acting_player.score > score_before
                        or score_reward_deltas[observation.observer_index] > 0
                    ),
                    route_had_permanent_marker=route_had_permanent_marker,
                )
                if route_claim_penalty:
                    adjusted = list(player_reward_deltas)
                    adjusted[observation.observer_index] += route_claim_penalty
                    player_reward_deltas = tuple(adjusted)
                if route_building_reward and route_building_post.owner is acting_player:
                    adjusted = list(player_reward_deltas)
                    adjusted[observation.observer_index] += route_building_reward
                    player_reward_deltas = tuple(adjusted)
                normal_move_completed = normal_move_in_progress and not acting_player.holding_pieces
                permanent_move_completed = bool(
                    permanent_move_in_progress
                    and not game.waiting_for_bm_move_any_2
                    and not acting_player.holding_pieces
                )
                action_was_spent = acting_player.actions_remaining < actions_remaining_before
                repeated_move_penalty = 0.0
                no_change_penalty = 0.0
                no_change_type = None
                movement_local_target = None
                movement_local_adjustment = 0.0
                immediate_q_undo = False
                single_piece_utilization_role = None
                if normal_move_completed and action_was_spent:
                    next_consecutive_move = acting_player.consecutive_paid_move_actions
                    repeated_move_penalty = consecutive_move_penalty(
                        movement_capacity, next_consecutive_move
                    )
                    no_change_type = pointless_movement_type(
                        move_origin_pieces,
                        move_destination_posts,
                        post_routes,
                    )
                    no_change_penalty = (
                        float(POINTLESS_MOVEMENT_LOCAL_TARGET) if no_change_type else 0.0
                    )
                    immediate_q_undo = is_immediate_one_piece_q_undo(
                        no_change_penalty,
                        normal_move_selections,
                    )
                    single_piece_utilization_role = single_piece_move_utilization_target(
                        pieces_moved,
                        immediate_q_undo=immediate_q_undo,
                        legal_pickups_at_start=normal_move_legal_pickups_at_start,
                    )
                    if next_consecutive_move >= 3:
                        movement_local_target = repeated_move_penalty
                    else:
                        movement_local_adjustment = repeated_move_penalty + (
                            0.0
                            if pieces_moved == 1
                            else movement_efficiency_penalty(pieces_moved, movement_capacity)
                        )
                elif permanent_move_completed:
                    no_change_type = pointless_movement_type(
                        move_origin_pieces,
                        move_destination_posts,
                        post_routes,
                    )
                    no_change_penalty = (
                        float(POINTLESS_MOVEMENT_LOCAL_TARGET) if no_change_type else 0.0
                    )
                    movement_local_target = no_change_penalty or None
                if normal_move_completed:
                    record_q_undo_workflow(
                        movement_metrics,
                        no_change_penalty,
                        normal_move_selections,
                        valuable_origin=normal_move_valuable_origin,
                    )
                record_pointless_movement_workflow(
                    movement_metrics,
                    no_change_penalty,
                    normal_move_completed=normal_move_completed,
                    permanent_move_any2_completed=permanent_move_completed,
                )
                movement_metrics.repeated_move_penalties += int(bool(repeated_move_penalty))
                if move_placement_route is not None:
                    move_destination_counts[move_placement_route] = (
                        move_destination_counts.get(move_placement_route, 0) + 1
                    )
                    if move_blocks_next_player:
                        move_blocked_next_player = True
                if action_was_spent:
                    movement_metrics.spent_action_count += 1
                    newly_completed_routes = frozenset()
                    move_claim_routes = frozenset()
                    blocked_move_claim_routes = frozenset()
                    if normal_move_completed:
                        movement_metrics.move_action_count += 1
                        record_move_capacity_utilization(
                            movement_metrics,
                            normal_move_nominal_capacity,
                            normal_move_movable_pieces_available,
                            pieces_moved,
                        )
                        consecutive_move1 = bool(
                            pieces_moved == 1 and previous_paid_move1 is not None
                        )
                        avoidable_extra_move = bool(
                            consecutive_move1
                            and normal_move_initial_pickup_post_slot
                            in previous_paid_move1.additional_pickup_post_slots
                        )
                        if pointless_move_audit_enabled:
                            initiation_selection = (
                                normal_move_selections[0]
                                if normal_move_selections
                                and normal_move_selections[0].role == "initial_pickup"
                                else None
                            )
                            movement_metrics.pointless_move_attribution.record_completed_move(
                                workflow_id=movement_workflow_id,
                                initiation_selection=initiation_selection,
                                pointless_type=no_change_type,
                                pieces_moved=pieces_moved,
                                effective_capacity=normal_move_effective_capacity,
                                immediate_q_undo=immediate_q_undo,
                                repeated_penalty=bool(repeated_move_penalty),
                                avoidable_extra_move=avoidable_extra_move,
                            )
                            if consecutive_move1:
                                movement_metrics.pointless_move_attribution.mark_consecutive_move1(
                                    (previous_paid_move1_workflow_id, movement_workflow_id)
                                )
                        previous_paid_move1 = update_consecutive_move1_telemetry(
                            movement_metrics,
                            previous_paid_move1,
                            normal_move_completed=True,
                            pieces_moved=pieces_moved,
                            nominal_capacity=normal_move_nominal_capacity,
                            initial_pickup_post_slot=normal_move_initial_pickup_post_slot,
                            additional_pickup_post_slots=(normal_move_additional_pickup_post_slots),
                        )
                        previous_paid_move1_workflow_id = (
                            movement_workflow_id if previous_paid_move1 is not None else None
                        )
                        turn_move_workflow_ids.append(movement_workflow_id)
                        movement_destination_routes = frozenset(
                            post_route_indices[post] for post in move_destination_posts
                        )
                        completed_routes_after = (
                            {
                                route_index
                                for route_index, route in enumerate(game.selected_map.routes)
                                if route.is_controlled_by(acting_player)
                            }
                            if move_tracking_active
                            else set()
                        )
                        newly_completed_routes = frozenset(
                            completed_routes_after - move_completed_routes_before
                        )
                        move_claim_routes = move_claim_eligible_routes(
                            newly_completed_routes,
                            already_claimable_pickup_routes,
                        )
                        blocked_move_claim_routes = frozenset(
                            completed_routes_after & already_claimable_pickup_routes
                        )
                        record_move_route_creation(
                            movement_metrics,
                            pieces_moved,
                            newly_completed_routes,
                        )
                        adjusted = list(player_reward_deltas)
                        if movement_local_target is None and not no_change_type:
                            if move_blocked_next_player:
                                adjusted[observation.observer_index] += MOVE_BLOCK_REWARD
                            (
                                acting_player.rewarded_move_focus_route_slots,
                                route_focus_reward,
                            ) = move_route_focus_reward(
                                acting_player.rewarded_move_focus_route_slots,
                                move_destination_counts,
                            )
                            adjusted[observation.observer_index] += route_focus_reward
                            if move_tracking_active:
                                adjusted[observation.observer_index] += completed_route_move_reward(
                                    move_completed_routes_before,
                                    completed_routes_after,
                                )
                        player_reward_deltas = tuple(adjusted)
                        move_destination_counts = {}
                        move_blocked_next_player = False
                        move_completed_routes_before = set()
                        move_tracking_active = False
                        move_pieces_picked_up = 0
                        move_origin_posts = []
                        move_origin_pieces = []
                        move_destination_posts = []
                        normal_move_selections = []
                        normal_move_valuable_origin = False
                        normal_move_nominal_capacity = 0
                        normal_move_movable_pieces_available = 0
                        normal_move_legal_pickups_at_start = 0
                        normal_move_effective_capacity = 0
                        normal_move_initial_pickup_post_slot = None
                        normal_move_additional_pickup_post_slots = frozenset()
                        move_observed_pickup_routes = set()
                        already_claimable_pickup_routes = set()
                    else:
                        previous_paid_move1 = None
                        previous_paid_move1_workflow_id = None
                    acting_player.rewarded_move_focus_route_slots = (
                        clear_move_route_focus_after_claim(
                            acting_player.rewarded_move_focus_route_slots,
                            action,
                            action_phase,
                        )
                    )
                    pending_routes, combo_reward = update_move_claim_combo(
                        acting_player.pending_move_claim_route_slots,
                        action=action,
                        turn_phase=action_phase,
                        action_was_spent=True,
                        newly_completed_routes=move_claim_routes,
                    )
                    acting_player.pending_move_claim_route_slots = pending_routes
                    (
                        pending_single_piece_move_claim_routes,
                        single_piece_claim_conversion,
                    ) = update_single_piece_move_claim_routes(
                        pending_single_piece_move_claim_routes,
                        action=action,
                        turn_phase=action_phase,
                        normal_move_completed=normal_move_completed,
                        pieces_moved=pieces_moved,
                        newly_completed_routes=move_claim_routes,
                    )
                    record_move_claim_reward_outcome(
                        movement_metrics,
                        combo_reward,
                        action=action,
                        turn_phase=action_phase,
                        blocked_already_claimable_routes=pending_blocked_move_claim_routes,
                    )
                    pending_blocked_move_claim_routes = (
                        blocked_move_claim_routes if normal_move_completed else frozenset()
                    )
                    if combo_reward:
                        movement_metrics.move_claim_conversions += 1
                        movement_metrics.single_piece_move_claim_conversions += int(
                            single_piece_claim_conversion
                        )
                        adjusted = list(player_reward_deltas)
                        adjusted[observation.observer_index] += combo_reward
                        player_reward_deltas = tuple(adjusted)
                    if normal_move_completed:
                        pending_terminal_move_workflows.append(
                            (movement_workflow_id, movement_destination_routes)
                        )
                        pending_terminal_completed_routes.update(newly_completed_routes)
                    else:
                        claimed_route = (
                            action.route_slot
                            if action_phase is TurnPhase.ACTIONS
                            and isinstance(action, RouteInteraction)
                            else None
                        )
                        for workflow_id in credited_movement_workflows(
                            pending_terminal_move_workflows,
                            pending_terminal_completed_routes,
                            claimed_route,
                        ):
                            grant_movement_workflow_terminal_credit(decisions, workflow_id)
                        pending_terminal_move_workflows.clear()
                        pending_terminal_completed_routes.clear()
                if permanent_move_completed:
                    move_origin_posts = []
                    move_origin_pieces = []
                    move_destination_posts = []
                    permanent_move_tracking_active = False
                player_reward_deltas = apply_route_completion_reward(
                    player_reward_deltas,
                    action=action,
                    turn_phase=action_phase,
                    acting_player_index=observation.observer_index,
                )
                player_reward_deltas = apply_opponent_route_score_penalty(
                    player_reward_deltas,
                    action=action,
                    turn_phase=action_phase,
                    acting_player_index=observation.observer_index,
                    projected_reward_deltas=score_reward_deltas,
                )
                if pending_disruption is not None and game.turn_phase is TurnPhase.ACTIONS:
                    disrupting_player, threatened_player, threats_before = pending_disruption
                    threats_after = len(
                        valuable_completed_route_slots(game, game.players[threatened_player])
                    )
                    disrupted_routes = max(threats_before - threats_after, 0)
                    if disrupted_routes:
                        adjusted = list(player_reward_deltas)
                        adjusted[disrupting_player] += 25.0 * disrupted_routes
                        player_reward_deltas = tuple(adjusted)
                    pending_disruption = None
                decisions.append(
                    TrainingDecision(
                        observation.features.clone(),
                        training_legal_action_mask.to(torch.uint8),
                        action_index,
                        observation.observer_index,
                        player_reward_deltas,
                        player_reward_deltas[observation.observer_index],
                        tier.number,
                        tier.epsilon,
                        tier.top_k,
                        selection.used_epsilon,
                        selection.model_rank,
                        selection.legal_action_count,
                        turn_before,
                        movement_workflow_id,
                        equivalent_action_indices=selection.equivalent_action_indices,
                        equivalent_action_groups=tuple(
                            group for group in training_semantic_action_groups if len(group) > 1
                        ),
                        receives_terminal_credit=not (
                            starts_normal_move or normal_move_in_progress
                        ),
                        case_a_pickup_action_groups=case_a_pickup_action_groups,
                        case_a_placement_action_groups=case_a_placement_action_groups,
                        move_continuation_pickup_action_groups=(
                            move_continuation_pickup_action_groups
                        ),
                        move_continuation_placement_action_groups=(
                            move_continuation_placement_action_groups
                        ),
                        move_continuation_pickup_depth=move_continuation_pickup_depth,
                        pointless_final_restorative_action_group=(
                            pointless_final_restorative_action_group
                            if no_change_type
                            and pointless_final_selected_type == no_change_type
                            and pointless_final_nonpointless_action_groups
                            else ()
                        ),
                        pointless_final_nonpointless_action_groups=(
                            pointless_final_nonpointless_action_groups
                            if no_change_type
                            and pointless_final_selected_type == no_change_type
                            and pointless_final_nonpointless_action_groups
                            else ()
                        ),
                        pointless_final_placement_type=(
                            no_change_type
                            if no_change_type
                            and pointless_final_selected_type == no_change_type
                            and pointless_final_nonpointless_action_groups
                            else None
                        ),
                        pointless_final_placement_piece_count=(
                            pieces_moved
                            if no_change_type
                            and pointless_final_selected_type == no_change_type
                            and pointless_final_nonpointless_action_groups
                            else 0
                        ),
                    )
                )
                if movement_local_target is not None:
                    mark_movement_workflow_target(
                        decisions,
                        movement_workflow_id,
                        movement_local_target,
                    )
                elif movement_local_adjustment:
                    add_movement_workflow_adjustment(
                        decisions,
                        movement_workflow_id,
                        movement_local_adjustment,
                    )
                if single_piece_utilization_role is not None:
                    set_single_piece_move_utilization_target(
                        decisions,
                        movement_workflow_id,
                        MOVE1_UTILIZATION_LOCAL_TARGET,
                        target_role=single_piece_utilization_role,
                    )
                if normal_move_completed:
                    normal_move_workflow_id = None
                if permanent_move_completed:
                    permanent_move_workflow_id = None
                if detailed_profiling:
                    reward_seconds += perf_counter() - reward_started
                end_is_pending = game.game_end or game.game_end_pending_immediate_resolution
                if game_end_trigger_player is None and end_is_pending and not end_was_pending:
                    game_end_trigger_player = observation.observer_index
            else:
                finalize_all_move_turn(
                    decisions,
                    movement_metrics,
                    turn_move_workflow_ids,
                    tracked_turn_player.paid_actions_spent_this_turn,
                    pointless_move_audit_enabled=pointless_move_audit_enabled,
                )
                self.progress.game_completion_failures += 1
                error = ActionLimitExceeded(
                    f"Game did not finish within {self.config.max_actions} interactions"
                )
                if failure_callback is not None:
                    failure_callback(game, tuple(action_trace), seat_tiers, error)
                if evaluation and not capture_action_limit:
                    raise error
                # A timeout is not a game loss. Keep every authoritative reward
                # and penalty already earned, but add no invented terminal value.
                return self._complete_trajectory(
                    decisions,
                    (0.0,) * len(game.players),
                    projected_before,
                    (),
                    action_trace,
                    seat_tiers,
                    reason="action_limit",
                    completed=False,
                    timings=timings(),
                    movement_metrics=movement_metrics,
                    shadow_policy_metrics=shadow_policy_metrics,
                    shadow_filter_audit=shadow_filter_metrics,
                    training_exploration_mode=training_exploration_mode,
                )

        finalize_all_move_turn(
            decisions,
            movement_metrics,
            turn_move_workflow_ids,
            tracked_turn_player.paid_actions_spent_this_turn,
            pointless_move_audit_enabled=pointless_move_audit_enabled,
        )
        if detailed_profiling:
            validation_started = perf_counter()
        validate_game(game)
        if detailed_profiling:
            validation_seconds += perf_counter() - validation_started
        winners = tuple(player.order - 1 for player in game.end_the_game())
        terminal_rewards = calculate_terminal_rewards(game, winners, game_end_trigger_player)
        return self._complete_trajectory(
            decisions,
            terminal_rewards,
            tuple(player.final_score for player in game.players),
            winners,
            action_trace,
            seat_tiers,
            reason=completed_game_reason(game),
            timings=timings(),
            movement_metrics=movement_metrics,
            shadow_policy_metrics=shadow_policy_metrics,
            shadow_filter_audit=shadow_filter_metrics,
            training_exploration_mode=training_exploration_mode,
        )

    @staticmethod
    def _increment(values, tier, amount=1):
        values[tier] = values.get(tier, 0) + amount

    def _record_tier_metrics(self, trajectory):
        for tier in trajectory.seat_tiers:
            self._increment(self.progress.tier_games, tier)
        for winner_index in trajectory.winner_indices:
            self._increment(self.progress.tier_wins, trajectory.seat_tiers[winner_index])
        for decision in trajectory.decisions:
            tier = decision.policy_tier
            self._increment(self.progress.tier_selected_rank_total, tier, decision.model_rank)
            selections = (
                self.progress.tier_epsilon_selections
                if decision.used_epsilon
                else self.progress.tier_top_k_selections
            )
            self._increment(selections, tier)
            self._increment(
                self.progress.tier_immediate_reward_total, tier, decision.immediate_reward
            )
            self._increment(self.progress.tier_reward_to_go_total, tier, decision.reward_to_go)

    def tier_metrics(self):
        metrics = {}
        decision_counts = {
            tier: self.progress.tier_epsilon_selections.get(tier, 0)
            + self.progress.tier_top_k_selections.get(tier, 0)
            for tier in range(1, len(self.config.tier_top_k) + 1)
        }
        for tier, games in self.progress.tier_games.items():
            decisions = decision_counts[tier]
            divisor = decisions or 1
            metrics[tier] = {
                "games": games,
                "wins": self.progress.tier_wins.get(tier, 0),
                "win_rate": self.progress.tier_wins.get(tier, 0) / games,
                "average_selected_rank": self.progress.tier_selected_rank_total.get(tier, 0)
                / divisor,
                "epsilon_selections": self.progress.tier_epsilon_selections.get(tier, 0),
                "top_k_selections": self.progress.tier_top_k_selections.get(tier, 0),
                "average_immediate_reward": self.progress.tier_immediate_reward_total.get(tier, 0)
                / divisor,
                "average_reward_to_go": self.progress.tier_reward_to_go_total.get(tier, 0)
                / divisor,
            }
        return metrics

    def _training_batches(
        self,
        decisions,
        *,
        max_training_decisions=None,
        effective_batch_count=None,
    ):
        decisions = list(decisions)
        if not decisions:
            return ()
        if max_training_decisions is None:
            max_training_decisions = self.config.normal_max_training_decisions
        sample_size = min(len(decisions), max_training_decisions)
        if effective_batch_count is None:
            batch_size = self.config.decision_batch_size
            batch_count = math.ceil(sample_size / batch_size)
        else:
            batch_count = min(effective_batch_count, sample_size)
            batch_size = math.ceil(sample_size / batch_count)
        if batch_count == 1:
            self.rng.shuffle(decisions)
            return (decisions[:sample_size],)

        grouped = {}
        for index, decision in enumerate(decisions):
            key = (
                ("movement", decision.movement_workflow_id)
                if decision.movement_workflow_id is not None
                else ("decision", index)
            )
            grouped.setdefault(key, []).append(decision)

        final_key = (
            ("movement", decisions[-1].movement_workflow_id)
            if decisions[-1].movement_workflow_id is not None
            else ("decision", len(decisions) - 1)
        )
        selected_keys = {final_key}
        ordered_selected_keys = [final_key]
        selected_count = len(grouped[final_key])
        priority_groups = sorted(
            (
                (key, group)
                for key, group in grouped.items()
                if key != final_key
                and any(_training_priority_value(decision) for decision in group)
            ),
            key=lambda item: max(_training_priority_value(decision) for decision in item[1]),
            reverse=True,
        )
        priority_budget = sample_size // 2
        for key, group in priority_groups:
            if selected_count + len(group) > priority_budget:
                continue
            selected_keys.add(key)
            ordered_selected_keys.append(key)
            selected_count += len(group)

        remaining_keys = [key for key in grouped if key not in selected_keys]
        self.rng.shuffle(remaining_keys)
        for key in remaining_keys:
            group_size = len(grouped[key])
            if selected_count + group_size <= sample_size:
                selected_keys.add(key)
                ordered_selected_keys.append(key)
                selected_count += group_size

        batches = [[] for _ in range(batch_count)]
        for key in ordered_selected_keys:
            group = grouped[key]
            available = [batch for batch in batches if len(batch) + len(group) <= batch_size]
            if not available:
                continue
            target = min(available, key=len)
            target.extend(group)
        for batch in batches:
            self.rng.shuffle(batch)
        return tuple(batch for batch in batches if batch)

    def _early_training_batches(self, decisions):
        """Sample an early trajectory evenly across eight chronological sections."""
        decisions = list(decisions)
        if not decisions:
            return ()

        section_count = 8
        batch_size = self.config.decision_batch_size
        sample_size = min(
            len(decisions),
            self.config.early_max_training_decisions,
        )
        section_budgets = [sample_size // section_count] * section_count
        for section in range(sample_size % section_count):
            section_budgets[section] += 1

        grouped = {}
        group_indices = {}
        for index, decision in enumerate(decisions):
            key = (
                ("movement", decision.movement_workflow_id)
                if decision.movement_workflow_id is not None
                else ("decision", index)
            )
            grouped.setdefault(key, []).append(decision)
            group_indices.setdefault(key, []).append(index)

        sections = [[] for _ in range(section_count)]
        for key, indices in group_indices.items():
            midpoint = (indices[0] + indices[-1]) // 2
            section = min(section_count - 1, midpoint * section_count // len(decisions))
            sections[section].append(key)

        final_key = (
            ("movement", decisions[-1].movement_workflow_id)
            if decisions[-1].movement_workflow_id is not None
            else ("decision", len(decisions) - 1)
        )
        selected_by_section = [[] for _ in range(section_count)]
        selected_counts = [0] * section_count
        selected_keys = set()

        priority_values = {
            key: max((_training_priority_value(decision) for decision in group), default=0)
            for key, group in grouped.items()
        }

        def priority_value(key):
            return priority_values[key]

        def add_to_section(key, section_index):
            selected_keys.add(key)
            selected_by_section[section_index].append(key)
            selected_counts[section_index] += len(grouped[key])

        for section, keys in enumerate(sections):
            budget = section_budgets[section]
            if (
                final_key in keys
                and len(grouped[final_key]) <= budget
                and len(grouped[final_key]) <= batch_size
            ):
                add_to_section(final_key, section)

            priority_keys = sorted(
                (key for key in keys if key not in selected_keys and priority_value(key)),
                key=priority_value,
                reverse=True,
            )
            priority_budget = budget // 2
            for key in priority_keys:
                if (
                    len(grouped[key]) <= batch_size
                    and selected_counts[section] + len(grouped[key]) <= priority_budget
                ):
                    add_to_section(key, section)

            remaining_keys = [key for key in keys if key not in selected_keys]
            self.rng.shuffle(remaining_keys)
            for key in remaining_keys:
                if (
                    len(grouped[key]) <= batch_size
                    and selected_counts[section] + len(grouped[key]) <= budget
                ):
                    add_to_section(key, section)

        remaining_capacity = sample_size - sum(selected_counts)
        overflow_by_section = []
        for keys in sections:
            priority_keys = sorted(
                (key for key in keys if key not in selected_keys and priority_value(key)),
                key=priority_value,
                reverse=True,
            )
            random_keys = [
                key for key in keys if key not in selected_keys and not priority_value(key)
            ]
            self.rng.shuffle(random_keys)
            overflow_by_section.append(priority_keys + random_keys)

        while remaining_capacity:
            added = False
            for candidates in overflow_by_section:
                fitting_position = next(
                    (
                        position
                        for position, key in enumerate(candidates)
                        if len(grouped[key]) <= remaining_capacity
                        and len(grouped[key]) <= batch_size
                    ),
                    None,
                )
                if fitting_position is None:
                    continue
                key = candidates.pop(fitting_position)
                section = min(
                    section_count - 1,
                    group_indices[key][0] * section_count // len(decisions),
                )
                add_to_section(key, section)
                remaining_capacity -= len(grouped[key])
                added = True
                if not remaining_capacity:
                    break
            if not added:
                break

        batches = []
        for keys in selected_by_section:
            for key in keys:
                group = grouped[key]
                if not batches or len(batches[-1]) + len(group) > batch_size:
                    batches.append([])
                batches[-1].extend(group)
        for batch in batches:
            self.rng.shuffle(batch)
        return tuple(batch for batch in batches if batch)

    @staticmethod
    def _sampled_octiles(decisions, batches):
        decisions = tuple(decisions)
        if not decisions:
            return ()
        positions = {id(decision): index for index, decision in enumerate(decisions)}
        counts = [0] * 8
        for decision in (decision for batch in batches for decision in batch):
            section = min(7, positions[id(decision)] * 8 // len(decisions))
            counts[section] += 1
        return tuple(counts)

    def _trajectory_training_decision_cap(self, curriculum_maturity):
        if curriculum_maturity == "early":
            return self.config.early_max_training_decisions
        if curriculum_maturity == "fresh":
            return self.config.fresh_max_training_decisions
        return self.config.normal_max_training_decisions

    def _decision_batch_loss_components(self, batch, *, base_q_effective_sample_count=None):
        observations = torch.stack([sample.observation for sample in batch]).float().to(device)
        targets = torch.tensor(
            [sample.reward_to_go for sample in batch], dtype=torch.float32, device=device
        )
        model_outputs = self._model_outputs(observations)
        action_groups = tuple(
            sample.equivalent_action_indices or (sample.action_index,) for sample in batch
        )
        maximum_size = max(map(len, action_groups))
        group_sizes = torch.as_tensor(
            [len(group) for group in action_groups],
            dtype=torch.long,
            device=device,
        )
        padded_indices = torch.as_tensor(
            [tuple(group) + (group[0],) * (maximum_size - len(group)) for group in action_groups],
            dtype=torch.long,
            device=device,
        )
        member_mask = torch.arange(maximum_size, device=device).unsqueeze(0) < group_sizes[:, None]
        selected_q_values = model_outputs.q_values.gather(1, padded_indices)
        member_losses = functional.smooth_l1_loss(
            selected_q_values,
            targets[:, None].expand_as(selected_q_values),
            reduction="none",
        )
        decision_losses = (member_losses * member_mask).sum(dim=1) / group_sizes
        base_q_included = torch.tensor(
            [not move_continuation_base_q_excluded(sample) for sample in batch],
            dtype=torch.bool,
            device=device,
        )
        local_base_q_sample_count = int(base_q_included.sum().item())
        if base_q_effective_sample_count is None:
            base_q_effective_sample_count = local_base_q_sample_count
        if base_q_effective_sample_count < local_base_q_sample_count:
            raise ValueError(
                "Microbatch base-Q samples cannot exceed the effective-batch denominator"
            )
        base_q_loss = (
            decision_losses[base_q_included].sum() / base_q_effective_sample_count
            if base_q_effective_sample_count
            else model_outputs.q_values.sum() * 0.0
        )
        case_a_result = case_a_family_ranking_loss(model_outputs.q_values, batch)
        continuation_result = move_continuation_family_ranking_loss(
            model_outputs.q_values,
            batch,
        )
        pointless_final_result = pointless_final_placement_ranking_loss(
            model_outputs.q_values,
            batch,
        )
        quality_signals = policy_quality_signal(
            targets,
            self.config.policy_return_scale,
        )
        policy_loss = policy_batch_losses(
            model_outputs.policy_logits,
            batch,
            quality_signals,
        ).mean()
        return (
            base_q_loss,
            policy_loss,
            case_a_result,
            continuation_result,
            pointless_final_result,
        )

    def _decision_batch_losses(self, batch):
        base_q_loss, policy_loss, case_a_result, continuation_result, pointless_final_result = (
            self._decision_batch_loss_components(batch)
        )
        move_pickup_ranking_result = combined_move_pickup_ranking_loss(
            case_a_result,
            continuation_result,
            base_q_loss,
        )
        q_loss = (
            base_q_loss
            + CASE_A_FAMILY_RANK_WEIGHT * move_pickup_ranking_result.loss
            + POINTLESS_FINAL_PLACEMENT_RANK_WEIGHT * pointless_final_result.loss
        )
        total_loss = q_loss + self.config.policy_loss_weight * policy_loss
        return q_loss, policy_loss, total_loss

    def _optimize_effective_batch(self, batch, *, microbatch_size=None):
        """Apply one optimizer update from a correctly weighted effective batch."""
        if not batch:
            raise TrainingRunError("Cannot optimize an empty decision batch")
        if microbatch_size is None:
            microbatch_size = self.config.decision_batch_size
        if microbatch_size < 1:
            raise ValueError("Microbatch size must be positive")

        self.optimizer.zero_grad(set_to_none=True)
        effective_size = len(batch)
        move_pickup_depth_counts = move_pickup_ranking_depth_sample_counts(batch)
        case_a_sample_count = move_pickup_depth_counts[0]
        continuation_sample_count = sum(move_pickup_depth_counts[1:])
        base_q_excluded_depth_counts = move_pickup_depth_counts[1:]
        base_q_excluded_sample_count = sum(base_q_excluded_depth_counts)
        base_q_included_sample_count = effective_size - base_q_excluded_sample_count
        pointless_final_sample_count = sum(
            bool(
                sample.pointless_final_restorative_action_group
                and sample.pointless_final_nonpointless_action_groups
            )
            for sample in batch
        )
        detached_losses = torch.zeros(3, dtype=torch.float32, device=device)
        detached_case_a = torch.zeros(26, dtype=torch.float32, device=device)
        detached_continuation = torch.zeros(9, dtype=torch.float32, device=device)
        detached_continuation_depths = torch.zeros((3, 21), dtype=torch.float32, device=device)
        detached_move_pickup = torch.zeros(5, dtype=torch.float32, device=device)
        detached_pointless_final = torch.zeros(4, dtype=torch.float32, device=device)
        pointless_final_counts = [0, 0, 0, 0, 0]
        continuation_depth_counts = [0, 0, 0]
        for start in range(0, effective_size, microbatch_size):
            microbatch = batch[start : start + microbatch_size]
            scale = len(microbatch) / effective_size
            (
                base_q_loss,
                policy_loss,
                case_a_result,
                continuation_result,
                pointless_final_result,
            ) = self._decision_batch_loss_components(
                microbatch,
                base_q_effective_sample_count=base_q_included_sample_count,
            )
            move_pickup_ranking_result = combined_move_pickup_ranking_loss(
                case_a_result,
                continuation_result,
                base_q_loss,
                effective_depth_counts=move_pickup_depth_counts,
            )
            move_pickup_ranking_contribution = (
                CASE_A_FAMILY_RANK_WEIGHT * move_pickup_ranking_result.loss
                if move_pickup_ranking_result.present_depth_count
                else None
            )
            pointless_final_contribution = (
                POINTLESS_FINAL_PLACEMENT_RANK_WEIGHT
                * pointless_final_result.loss
                * pointless_final_result.sample_count
                / pointless_final_sample_count
                if pointless_final_sample_count
                else None
            )
            auxiliary_contribution = move_pickup_ranking_contribution
            if pointless_final_contribution is not None:
                auxiliary_contribution = (
                    pointless_final_contribution
                    if auxiliary_contribution is None
                    else auxiliary_contribution + pointless_final_contribution
                )
            self._accumulate_independent_losses(
                base_q_loss,
                policy_loss,
                scale,
                q_auxiliary_loss=auxiliary_contribution,
                q_scale=1.0,
            )
            q_contribution = base_q_loss.detach()
            if move_pickup_ranking_contribution is not None:
                q_contribution = q_contribution + move_pickup_ranking_contribution.detach()
            if pointless_final_contribution is not None:
                q_contribution = q_contribution + pointless_final_contribution.detach()
            policy_contribution = scale * policy_loss.detach()
            detached_losses += torch.stack(
                (
                    q_contribution,
                    policy_contribution,
                    q_contribution + self.config.policy_loss_weight * policy_contribution,
                )
            )
            if case_a_result.sample_count:
                detached_case_a += torch.stack(
                    (
                        case_a_result.loss.detach() * case_a_result.sample_count,
                        case_a_result.violating_sample_count.to(torch.float32),
                        case_a_result.violating_pair_count.to(torch.float32),
                        case_a_result.violating_pair_fraction_sum.to(torch.float32),
                        case_a_result.all_pickups_above_all_placements_count.to(torch.float32),
                        case_a_result.q1_pickup_count.to(torch.float32),
                        case_a_result.worst_pickup_q_sum.to(torch.float32),
                        case_a_result.best_placement_q_sum.to(torch.float32),
                        case_a_result.worst_pickup_minus_best_placement_gap_sum.to(torch.float32),
                        case_a_result.margin_satisfied_count.to(torch.float32),
                        *(value.to(torch.float32) for value in case_a_result.depth_diagnostic_sums),
                    )
                )
            if continuation_result.sample_count:
                detached_continuation += torch.stack(
                    (
                        continuation_result.loss.detach() * continuation_result.sample_count,
                        continuation_result.violating_sample_count.to(torch.float32),
                        continuation_result.best_pickup_above_all_placements_count.to(
                            torch.float32
                        ),
                        continuation_result.q1_pickup_count.to(torch.float32),
                        continuation_result.all_pickups_above_all_placements_count.to(
                            torch.float32
                        ),
                        continuation_result.margin_satisfied_count.to(torch.float32),
                        continuation_result.worst_pickup_q_sum.to(torch.float32),
                        continuation_result.best_placement_q_sum.to(torch.float32),
                        continuation_result.worst_pickup_minus_best_placement_gap_sum.to(
                            torch.float32
                        ),
                    )
                )
                detached_continuation_depths += torch.stack(
                    tuple(
                        torch.stack(
                            (
                                continuation_result.depth_loss_sums[index].detach(),
                                continuation_result.depth_q1_pickup_counts[index].to(torch.float32),
                                continuation_result.depth_all_pickups_above_all_placements_counts[
                                    index
                                ].to(torch.float32),
                                continuation_result.depth_margin_satisfied_counts[index].to(
                                    torch.float32
                                ),
                                continuation_result.depth_gap_sums[index].to(torch.float32),
                                *(
                                    value.to(torch.float32)
                                    for value in continuation_result.depth_diagnostic_sums[index]
                                ),
                            )
                        )
                        for index in range(3)
                    )
                )
                continuation_depth_counts[0] += continuation_result.depth_2_count
                continuation_depth_counts[1] += continuation_result.depth_3_count
                continuation_depth_counts[2] += continuation_result.depth_4_count
            if move_pickup_ranking_result.present_depth_count:
                detached_move_pickup += torch.stack(
                    (
                        move_pickup_ranking_result.loss.detach(),
                        *(
                            contribution.detach()
                            for contribution in move_pickup_ranking_result.depth_contributions
                        ),
                    )
                )
            if pointless_final_result.sample_count:
                detached_pointless_final += torch.stack(
                    (
                        pointless_final_result.loss.detach() * pointless_final_result.sample_count,
                        pointless_final_result.violating_sample_count.to(torch.float32),
                        pointless_final_result.margin_satisfied_count.to(torch.float32),
                        pointless_final_result.q_gap_sum.to(torch.float32),
                    )
                )
                pointless_final_counts[0] += pointless_final_result.restorative_q1_count
                pointless_final_counts[1] += pointless_final_result.immediate_q_undo_count
                pointless_final_counts[2] += pointless_final_result.multi_piece_count
                pointless_final_counts[3] += pointless_final_result.exact_restoration_count
                pointless_final_counts[4] += pointless_final_result.equivalent_rearrangement_count
        self._clip_independent_gradients()
        self.optimizer.step()
        case_a_values = detached_case_a.cpu().tolist()
        self._last_effective_batch_case_a_metrics = (
            case_a_sample_count,
            case_a_values[0],
            int(case_a_values[1]),
            int(case_a_values[2]),
            case_a_values[3],
            int(case_a_values[4]),
            int(case_a_values[5]),
            case_a_values[6],
            case_a_values[7],
            case_a_values[8],
            int(case_a_values[9]),
            tuple(case_a_values[10:]),
        )
        continuation_values = detached_continuation.cpu().tolist()
        continuation_depth_values = detached_continuation_depths.cpu().tolist()
        self._last_effective_batch_move_continuation_metrics = (
            continuation_sample_count,
            continuation_values[0],
            int(continuation_values[1]),
            int(continuation_values[2]),
            int(continuation_values[3]),
            int(continuation_values[4]),
            int(continuation_values[5]),
            continuation_values[6],
            continuation_values[7],
            continuation_values[8],
            *continuation_depth_counts,
            tuple(tuple(values) for values in continuation_depth_values),
        )
        move_pickup_values = detached_move_pickup.cpu().tolist()
        self._last_effective_batch_move_pickup_metrics = (
            sum(count > 0 for count in move_pickup_depth_counts),
            move_pickup_values[0],
            tuple(move_pickup_values[1:]),
        )
        self._last_effective_batch_base_q_metrics = (
            base_q_included_sample_count,
            base_q_excluded_sample_count,
            tuple(base_q_excluded_depth_counts),
        )
        pointless_final_values = detached_pointless_final.cpu().tolist()
        self._last_effective_batch_pointless_final_metrics = (
            pointless_final_sample_count,
            pointless_final_values[0],
            int(pointless_final_values[1]),
            int(pointless_final_values[2]),
            pointless_final_values[3],
            *pointless_final_counts,
        )
        return tuple(detached_losses.cpu().tolist())

    def update_model(self, trajectories, *, curriculum_maturities=None) -> float:
        """Update from representative batches within each trajectory's configured cap."""
        trajectories = tuple(trajectories)
        if not trajectories or any(not trajectory.decisions for trajectory in trajectories):
            raise TrainingRunError("Cannot train from an empty trajectory batch")
        if curriculum_maturities is None:
            curriculum_maturities = (None,) * len(trajectories)
        else:
            curriculum_maturities = tuple(curriculum_maturities)
            if len(curriculum_maturities) != len(trajectories):
                raise TrainingRunError("Each trajectory must have one curriculum maturity")
        self.model.train()
        q_losses = []
        policy_losses = []
        total_losses = []
        case_a_sample_count = 0
        case_a_loss_sum = 0.0
        case_a_violating_sample_count = 0
        case_a_violating_pair_count = 0
        case_a_violating_pair_fraction_sum = 0.0
        case_a_all_pickups_above_all_placements_count = 0
        case_a_q1_pickup_count = 0
        case_a_worst_pickup_q_sum = 0.0
        case_a_best_placement_q_sum = 0.0
        case_a_worst_pickup_minus_best_placement_gap_sum = 0.0
        case_a_margin_satisfied_count = 0
        case_a_depth_diagnostic_sums = [0.0] * len(MOVE_PICKUP_RANKING_DEPTH_DIAGNOSTIC_METRICS)
        continuation_sample_count = 0
        continuation_loss_sum = 0.0
        continuation_violating_sample_count = 0
        continuation_best_pickup_above_all_placements_count = 0
        continuation_q1_pickup_count = 0
        continuation_all_pickups_above_all_placements_count = 0
        continuation_margin_satisfied_count = 0
        continuation_worst_pickup_q_sum = 0.0
        continuation_best_placement_q_sum = 0.0
        continuation_worst_pickup_minus_best_placement_gap_sum = 0.0
        continuation_depth_2_count = 0
        continuation_depth_3_count = 0
        continuation_depth_4_count = 0
        continuation_depth_metric_sums = [
            [0.0] * (5 + len(MOVE_PICKUP_RANKING_DEPTH_DIAGNOSTIC_METRICS)) for _depth in range(3)
        ]
        base_q_included_sample_count = 0
        base_q_excluded_sample_count = 0
        base_q_excluded_depth_counts = [0, 0, 0]
        move_pickup_eligible_effective_batch_count = 0
        move_pickup_present_depth_count_total = 0
        move_pickup_weighted_loss_total = 0.0
        move_pickup_depth_contribution_totals = [0.0] * 4
        pointless_final_sample_count = 0
        pointless_final_loss_sum = 0.0
        pointless_final_violating_sample_count = 0
        pointless_final_margin_satisfied_count = 0
        pointless_final_q_gap_sum = 0.0
        pointless_final_restorative_q1_count = 0
        pointless_final_immediate_q_undo_count = 0
        pointless_final_multi_piece_count = 0
        pointless_final_exact_restoration_count = 0
        pointless_final_equivalent_rearrangement_count = 0
        coverage = []
        for trajectory, curriculum_maturity in zip(trajectories, curriculum_maturities):
            if curriculum_maturity == "early":
                batches = self._early_training_batches(trajectory.decisions)
                sampled_octiles = self._sampled_octiles(trajectory.decisions, batches)
            else:
                batches = self._training_batches(
                    trajectory.decisions,
                    max_training_decisions=self._trajectory_training_decision_cap(
                        curriculum_maturity
                    ),
                    effective_batch_count=(
                        FRESH_OPTIMIZER_UPDATES_PER_TRAJECTORY
                        if curriculum_maturity == "fresh"
                        else None
                    ),
                )
                sampled_octiles = ()
            coverage.append(
                TrainingSampleCoverage(
                    total_decisions=len(trajectory.decisions),
                    sampled_decisions=sum(map(len, batches)),
                    sampled_octiles=sampled_octiles,
                )
            )
            for batch in batches:
                q_value, policy_value, total_value = self._optimize_effective_batch(batch)
                q_losses.append(q_value)
                policy_losses.append(policy_value)
                total_losses.append(total_value)
                (
                    batch_case_a_samples,
                    batch_case_a_loss_sum,
                    batch_case_a_violating_samples,
                    batch_case_a_violating_pairs,
                    batch_case_a_violating_pair_fraction_sum,
                    batch_case_a_all_pickups_above_all_placements,
                    batch_case_a_q1_pickups,
                    batch_case_a_worst_pickup_q_sum,
                    batch_case_a_best_placement_q_sum,
                    batch_case_a_worst_pickup_minus_best_placement_gap_sum,
                    batch_case_a_margin_satisfied,
                    batch_case_a_depth_diagnostic_sums,
                ) = self._last_effective_batch_case_a_metrics
                case_a_sample_count += batch_case_a_samples
                case_a_loss_sum += batch_case_a_loss_sum
                case_a_violating_sample_count += batch_case_a_violating_samples
                case_a_violating_pair_count += batch_case_a_violating_pairs
                case_a_violating_pair_fraction_sum += batch_case_a_violating_pair_fraction_sum
                case_a_all_pickups_above_all_placements_count += (
                    batch_case_a_all_pickups_above_all_placements
                )
                case_a_q1_pickup_count += batch_case_a_q1_pickups
                case_a_worst_pickup_q_sum += batch_case_a_worst_pickup_q_sum
                case_a_best_placement_q_sum += batch_case_a_best_placement_q_sum
                case_a_worst_pickup_minus_best_placement_gap_sum += (
                    batch_case_a_worst_pickup_minus_best_placement_gap_sum
                )
                case_a_margin_satisfied_count += batch_case_a_margin_satisfied
                for metric_index, metric_value in enumerate(batch_case_a_depth_diagnostic_sums):
                    case_a_depth_diagnostic_sums[metric_index] += metric_value
                (
                    batch_continuation_samples,
                    batch_continuation_loss_sum,
                    batch_continuation_violating_samples,
                    batch_continuation_best_pickup_above_all_placements,
                    batch_continuation_q1_pickups,
                    batch_continuation_all_pickups_above_all_placements,
                    batch_continuation_margin_satisfied,
                    batch_continuation_worst_pickup_q_sum,
                    batch_continuation_best_placement_q_sum,
                    batch_continuation_gap_sum,
                    batch_continuation_depth_2,
                    batch_continuation_depth_3,
                    batch_continuation_depth_4,
                    batch_continuation_depth_metrics,
                ) = self._last_effective_batch_move_continuation_metrics
                continuation_sample_count += batch_continuation_samples
                continuation_loss_sum += batch_continuation_loss_sum
                continuation_violating_sample_count += batch_continuation_violating_samples
                continuation_best_pickup_above_all_placements_count += (
                    batch_continuation_best_pickup_above_all_placements
                )
                continuation_q1_pickup_count += batch_continuation_q1_pickups
                continuation_all_pickups_above_all_placements_count += (
                    batch_continuation_all_pickups_above_all_placements
                )
                continuation_margin_satisfied_count += batch_continuation_margin_satisfied
                continuation_worst_pickup_q_sum += batch_continuation_worst_pickup_q_sum
                continuation_best_placement_q_sum += batch_continuation_best_placement_q_sum
                continuation_worst_pickup_minus_best_placement_gap_sum += batch_continuation_gap_sum
                continuation_depth_2_count += batch_continuation_depth_2
                continuation_depth_3_count += batch_continuation_depth_3
                continuation_depth_4_count += batch_continuation_depth_4
                for depth_index, depth_values in enumerate(batch_continuation_depth_metrics):
                    for metric_index, metric_value in enumerate(depth_values):
                        continuation_depth_metric_sums[depth_index][metric_index] += metric_value
                (
                    batch_move_pickup_present_depth_count,
                    batch_move_pickup_weighted_loss,
                    batch_move_pickup_depth_contributions,
                ) = self._last_effective_batch_move_pickup_metrics
                if batch_move_pickup_present_depth_count:
                    move_pickup_eligible_effective_batch_count += 1
                    move_pickup_present_depth_count_total += batch_move_pickup_present_depth_count
                    move_pickup_weighted_loss_total += batch_move_pickup_weighted_loss
                    for depth_index, contribution in enumerate(
                        batch_move_pickup_depth_contributions
                    ):
                        move_pickup_depth_contribution_totals[depth_index] += contribution
                (
                    batch_base_q_included_samples,
                    batch_base_q_excluded_samples,
                    batch_base_q_excluded_depth_counts,
                ) = self._last_effective_batch_base_q_metrics
                base_q_included_sample_count += batch_base_q_included_samples
                base_q_excluded_sample_count += batch_base_q_excluded_samples
                for depth_index, depth_count in enumerate(batch_base_q_excluded_depth_counts):
                    base_q_excluded_depth_counts[depth_index] += depth_count
                (
                    batch_pointless_final_samples,
                    batch_pointless_final_loss_sum,
                    batch_pointless_final_violating_samples,
                    batch_pointless_final_margin_satisfied,
                    batch_pointless_final_q_gap_sum,
                    batch_pointless_final_restorative_q1,
                    batch_pointless_final_immediate_q_undo,
                    batch_pointless_final_multi_piece,
                    batch_pointless_final_exact_restoration,
                    batch_pointless_final_equivalent_rearrangement,
                ) = self._last_effective_batch_pointless_final_metrics
                pointless_final_sample_count += batch_pointless_final_samples
                pointless_final_loss_sum += batch_pointless_final_loss_sum
                pointless_final_violating_sample_count += batch_pointless_final_violating_samples
                pointless_final_margin_satisfied_count += batch_pointless_final_margin_satisfied
                pointless_final_q_gap_sum += batch_pointless_final_q_gap_sum
                pointless_final_restorative_q1_count += batch_pointless_final_restorative_q1
                pointless_final_immediate_q_undo_count += batch_pointless_final_immediate_q_undo
                pointless_final_multi_piece_count += batch_pointless_final_multi_piece
                pointless_final_exact_restoration_count += batch_pointless_final_exact_restoration
                pointless_final_equivalent_rearrangement_count += (
                    batch_pointless_final_equivalent_rearrangement
                )
                self.progress.training_updates += 1
                if self._policy_parameters and self.config.policy_loss_weight:
                    self.progress.policy_training_updates += 1
        self.last_training_sample_coverage = tuple(coverage)
        self.model.eval()

        value = sum(q_losses) / len(q_losses)
        policy_value = sum(policy_losses) / len(policy_losses)
        total_value = sum(total_losses) / len(total_losses)
        self.progress.last_loss = value
        self.progress.last_q_loss = value
        self.progress.last_policy_loss = policy_value
        self.progress.last_total_loss = total_value
        self.progress.last_base_q_included_samples = base_q_included_sample_count
        self.progress.last_base_q_excluded_samples = base_q_excluded_sample_count
        self.progress.last_move_continuation_base_q_excluded_samples = base_q_excluded_sample_count
        self.progress.last_move_continuation_base_q_excluded_h2 = base_q_excluded_depth_counts[0]
        self.progress.last_move_continuation_base_q_excluded_h3 = base_q_excluded_depth_counts[1]
        self.progress.last_move_continuation_base_q_excluded_h4 = base_q_excluded_depth_counts[2]
        self.progress.last_case_a_family_ranking_samples = case_a_sample_count
        self.progress.last_case_a_family_ranking_loss = (
            case_a_loss_sum / case_a_sample_count if case_a_sample_count else None
        )
        self.progress.last_case_a_family_ranking_violating_samples = case_a_violating_sample_count
        self.progress.last_case_a_family_ranking_violation_fraction = (
            case_a_violating_sample_count / case_a_sample_count if case_a_sample_count else None
        )
        # Retained in the CSV schema for historical rows from the previous
        # best-pickup experiment; the all-pickups objective reports pair metrics.
        self.progress.last_case_a_family_ranking_mean_violating_placements = None
        self.progress.last_case_a_family_ranking_mean_violating_pairs = (
            case_a_violating_pair_count / case_a_sample_count if case_a_sample_count else None
        )
        self.progress.last_case_a_family_ranking_mean_violating_pair_fraction = (
            case_a_violating_pair_fraction_sum / case_a_sample_count
            if case_a_sample_count
            else None
        )
        self.progress.last_case_a_family_ranking_all_pickups_above_all_placements_fraction = (
            case_a_all_pickups_above_all_placements_count / case_a_sample_count
            if case_a_sample_count
            else None
        )
        self.progress.last_case_a_family_ranking_q1_pickup_fraction = (
            case_a_q1_pickup_count / case_a_sample_count if case_a_sample_count else None
        )
        self.progress.last_case_a_family_ranking_worst_pickup_q_mean = (
            case_a_worst_pickup_q_sum / case_a_sample_count if case_a_sample_count else None
        )
        self.progress.last_case_a_family_ranking_best_placement_q_mean = (
            case_a_best_placement_q_sum / case_a_sample_count if case_a_sample_count else None
        )
        self.progress.last_case_a_family_ranking_worst_pickup_minus_best_placement_gap_mean = (
            case_a_worst_pickup_minus_best_placement_gap_sum / case_a_sample_count
            if case_a_sample_count
            else None
        )
        self.progress.last_case_a_family_ranking_margin_satisfied_fraction = (
            case_a_margin_satisfied_count / case_a_sample_count if case_a_sample_count else None
        )
        self.progress.last_move_continuation_family_ranking_samples = continuation_sample_count
        self.progress.last_move_continuation_family_ranking_after_2_pickups = (
            continuation_depth_2_count
        )
        self.progress.last_move_continuation_family_ranking_after_3_pickups = (
            continuation_depth_3_count
        )
        self.progress.last_move_continuation_family_ranking_after_4_pickups = (
            continuation_depth_4_count
        )
        self.progress.last_move_continuation_family_ranking_loss = (
            continuation_loss_sum / continuation_sample_count if continuation_sample_count else None
        )
        self.progress.last_move_continuation_family_ranking_violating_samples = (
            continuation_violating_sample_count
        )
        self.progress.last_move_continuation_family_ranking_violation_fraction = (
            continuation_violating_sample_count / continuation_sample_count
            if continuation_sample_count
            else None
        )
        self.progress.last_move_continuation_best_pickup_above_all_placements_fraction = (
            continuation_best_pickup_above_all_placements_count / continuation_sample_count
            if continuation_sample_count
            else None
        )
        self.progress.last_move_continuation_q1_pickup_fraction = (
            continuation_q1_pickup_count / continuation_sample_count
            if continuation_sample_count
            else None
        )
        move_pickup_sample_count = case_a_sample_count + continuation_sample_count
        move_pickup_q1_count = case_a_q1_pickup_count + continuation_q1_pickup_count
        move_pickup_all_above_count = (
            case_a_all_pickups_above_all_placements_count
            + continuation_all_pickups_above_all_placements_count
        )
        move_pickup_margin_satisfied_count = (
            case_a_margin_satisfied_count + continuation_margin_satisfied_count
        )
        move_pickup_gap_sum = (
            case_a_worst_pickup_minus_best_placement_gap_sum
            + continuation_worst_pickup_minus_best_placement_gap_sum
        )
        self.progress.last_move_pickup_ranking_samples = move_pickup_sample_count
        self.progress.last_move_pickup_ranking_loss = (
            move_pickup_weighted_loss_total / move_pickup_eligible_effective_batch_count
            if move_pickup_eligible_effective_batch_count
            else None
        )
        self.progress.last_move_pickup_ranking_eligible_effective_batches = (
            move_pickup_eligible_effective_batch_count
        )
        self.progress.last_move_pickup_ranking_mean_present_depth_count = (
            move_pickup_present_depth_count_total / move_pickup_eligible_effective_batch_count
            if move_pickup_eligible_effective_batch_count
            else None
        )
        self.progress.last_move_pickup_ranking_q1_pickup_fraction = (
            move_pickup_q1_count / move_pickup_sample_count if move_pickup_sample_count else None
        )
        self.progress.last_move_pickup_ranking_all_pickups_above_all_placements_fraction = (
            move_pickup_all_above_count / move_pickup_sample_count
            if move_pickup_sample_count
            else None
        )
        self.progress.last_move_pickup_ranking_margin_satisfied_fraction = (
            move_pickup_margin_satisfied_count / move_pickup_sample_count
            if move_pickup_sample_count
            else None
        )
        self.progress.last_move_pickup_ranking_worst_pickup_minus_best_placement_gap_mean = (
            move_pickup_gap_sum / move_pickup_sample_count if move_pickup_sample_count else None
        )
        depth_metrics = (
            (
                case_a_sample_count,
                case_a_loss_sum,
                case_a_q1_pickup_count,
                case_a_all_pickups_above_all_placements_count,
                case_a_margin_satisfied_count,
                case_a_worst_pickup_minus_best_placement_gap_sum,
                *case_a_depth_diagnostic_sums,
            ),
            *(
                (count, *values)
                for count, values in zip(
                    (
                        continuation_depth_2_count,
                        continuation_depth_3_count,
                        continuation_depth_4_count,
                    ),
                    continuation_depth_metric_sums,
                )
            ),
        )
        for depth, depth_values in enumerate(depth_metrics, start=1):
            (
                count,
                loss_sum,
                q1_count,
                all_above_count,
                margin_count,
                gap_sum,
                *diagnostic_sums,
            ) = depth_values
            prefix = f"last_move_pickup_ranking_holding_{depth}"
            setattr(self.progress, f"{prefix}_samples", count)
            setattr(self.progress, f"{prefix}_loss", loss_sum / count if count else None)
            setattr(
                self.progress,
                f"{prefix}_q1_pickup_fraction",
                q1_count / count if count else None,
            )
            setattr(
                self.progress,
                f"{prefix}_all_pickups_above_all_placements_fraction",
                all_above_count / count if count else None,
            )
            setattr(
                self.progress,
                f"{prefix}_margin_satisfied_fraction",
                margin_count / count if count else None,
            )
            setattr(
                self.progress,
                f"{prefix}_worst_pickup_minus_best_placement_gap_mean",
                gap_sum / count if count else None,
            )
            setattr(
                self.progress,
                f"{prefix}_weighted_contribution",
                (
                    move_pickup_depth_contribution_totals[depth - 1]
                    / move_pickup_eligible_effective_batch_count
                    if move_pickup_eligible_effective_batch_count
                    else None
                ),
            )
            for metric_name, metric_sum in zip(
                MOVE_PICKUP_RANKING_DEPTH_DIAGNOSTIC_METRICS,
                diagnostic_sums,
            ):
                metric_value = (
                    int(metric_sum)
                    if metric_name.endswith("_samples")
                    else metric_sum / count
                    if count
                    else None
                )
                setattr(self.progress, f"{prefix}_{metric_name}", metric_value)
        self.progress.last_pointless_final_placement_ranking_samples = pointless_final_sample_count
        self.progress.last_pointless_final_placement_ranking_loss = (
            pointless_final_loss_sum / pointless_final_sample_count
            if pointless_final_sample_count
            else None
        )
        self.progress.last_pointless_final_placement_restorative_q1_fraction = (
            pointless_final_restorative_q1_count / pointless_final_sample_count
            if pointless_final_sample_count
            else None
        )
        self.progress.last_pointless_final_placement_margin_satisfied_fraction = (
            pointless_final_margin_satisfied_count / pointless_final_sample_count
            if pointless_final_sample_count
            else None
        )
        self.progress.last_pointless_final_placement_q_gap_mean = (
            pointless_final_q_gap_sum / pointless_final_sample_count
            if pointless_final_sample_count
            else None
        )
        self.progress.last_pointless_final_placement_immediate_q_undo_samples = (
            pointless_final_immediate_q_undo_count
        )
        self.progress.last_pointless_final_placement_multi_piece_samples = (
            pointless_final_multi_piece_count
        )
        self.progress.last_pointless_final_placement_exact_restoration_samples = (
            pointless_final_exact_restoration_count
        )
        self.progress.last_pointless_final_placement_equivalent_rearrangement_samples = (
            pointless_final_equivalent_rearrangement_count
        )
        self.loss_total += sum(q_losses)
        if self._policy_parameters and self.config.policy_loss_weight:
            self.policy_loss_total += sum(policy_losses)
        self.progress.mean_loss = self.loss_total / self.progress.training_updates
        self.progress.mean_policy_loss = (
            self.policy_loss_total / self.progress.policy_training_updates
            if self.progress.policy_training_updates
            else None
        )
        return value

    def trajectory_loss(
        self,
        trajectory,
        *,
        chunk_size=TRAJECTORY_LOSS_CHUNK_SIZE,
    ) -> float | None:
        """Measure one learning game's loss without updating the model."""
        samples = list(trajectory.decisions)
        if not samples:
            return None
        if chunk_size < 1:
            raise ValueError("Trajectory-loss chunk size must be positive")
        self.model.eval()
        loss_chunks = []
        with torch.no_grad():
            for start in range(0, len(samples), chunk_size):
                chunk = samples[start : start + chunk_size]
                observations = (
                    torch.stack([sample.observation for sample in chunk]).float().to(device)
                )
                targets = torch.tensor(
                    [sample.reward_to_go for sample in chunk],
                    dtype=torch.float32,
                    device=device,
                )
                model_outputs = (
                    self.model.forward_q(observations)
                    if hasattr(self.model, "forward_q")
                    else self._model_outputs(observations).q_values
                )
                loss_chunks.append(
                    torch.stack(
                        [
                            functional.smooth_l1_loss(
                                model_outputs[
                                    row,
                                    torch.as_tensor(
                                        sample.equivalent_action_indices or (sample.action_index,),
                                        dtype=torch.long,
                                        device=device,
                                    ),
                                ],
                                targets[row].expand(
                                    len(sample.equivalent_action_indices or (sample.action_index,))
                                ),
                            )
                            for row, sample in enumerate(chunk)
                        ]
                    )
                )
            return torch.cat(loss_chunks).mean().item()

    def train(self, starting_states, episodes, *, batch_size=8, quiet=True):
        if episodes < 1:
            raise ValueError("episodes must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        states = tuple(Path(path) for path in starting_states)
        if not states:
            raise ValueError("At least one starting state is required")
        actual_hashes = {_file_sha256(path) for path in states}
        if self.source_state_sha256 is not None and actual_hashes != set(
            self.source_state_sha256.values()
        ):
            raise ValueError("Starting states do not match the resumed checkpoint")

        trajectories = []
        starting_game_count = self.progress.completed_games
        for episode in range(episodes):
            state = states[(starting_game_count + episode) % len(states)]
            trajectory = self.collect_game(state, quiet=quiet)
            trajectories.append(trajectory)
            self.update_model((trajectory,))
        return tuple(trajectories)

    def save_checkpoint(
        self,
        path,
        starting_states,
        *,
        curriculum_state=_CURRICULUM_STATE_UNSET,
    ):
        if curriculum_state is _CURRICULUM_STATE_UNSET:
            curriculum_state = self.curriculum_state
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state_paths = tuple(Path(source) for source in starting_states)
        sources = {str(source): self._source_state_hash(source) for source in state_paths}
        active_paths = set(state_paths)
        self._source_state_hash_cache = {
            source: cached
            for source, cached in self._source_state_hash_cache.items()
            if source in active_paths
        }
        self.progress.checkpoint_saves += 1
        checkpoint = {
            "training_checkpoint_format": TRAINING_CHECKPOINT_FORMAT,
            "training_checkpoint_version": TRAINING_CHECKPOINT_VERSION,
            "model_checkpoint_format": MODEL_CHECKPOINT_FORMAT,
            "model_checkpoint_version": MODEL_CHECKPOINT_VERSION,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_progress": asdict(self.progress),
            "training_config": asdict(self.config),
            "source_state_sha256": sources,
            "policy_rng_state": self.rng.getstate(),
            "loss_total": self.loss_total,
            "policy_loss_total": self.policy_loss_total,
            "policy_trunk_gradient_scale": self._policy_trunk_gradient_scale(),
            "curriculum_state": curriculum_state,
            **action_schema_metadata(),
            **observation_schema_metadata(),
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
            torch.save(checkpoint, temporary)
            temporary.replace(target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self.source_state_sha256 = sources
        self.curriculum_state = curriculum_state
        return target

    @classmethod
    def from_checkpoint(cls, path):
        checkpoint = torch.load(path, map_location=device)
        if checkpoint.get("training_checkpoint_format") != TRAINING_CHECKPOINT_FORMAT:
            raise ValueError("Not a Hansa shared-model training checkpoint")
        checkpoint_version = checkpoint.get("training_checkpoint_version")
        if checkpoint_version not in (
            LEGACY_Q_ONLY_CHECKPOINT_VERSION,
            LEGACY_DUAL_HEAD_CHECKPOINT_VERSION,
            TRAINING_CHECKPOINT_VERSION,
        ):
            raise ValueError("Incompatible training checkpoint version")
        expected_model_version = (
            MODEL_CHECKPOINT_VERSION
            if checkpoint_version == TRAINING_CHECKPOINT_VERSION
            else LEGACY_MODEL_CHECKPOINT_VERSION
        )
        if checkpoint_version != LEGACY_Q_ONLY_CHECKPOINT_VERSION and (
            checkpoint.get("model_checkpoint_format") != MODEL_CHECKPOINT_FORMAT
            or checkpoint.get("model_checkpoint_version") != expected_model_version
        ):
            raise ValueError("Training checkpoint has an incompatible model schema")
        validate_action_schema_metadata(checkpoint, "Training checkpoint")
        migrated_observation_schema = validate_model_observation_schema_metadata(
            checkpoint, "Training checkpoint"
        )

        config_values = dict(checkpoint["training_config"])
        for obsolete_key in (
            "policy_trunk_gradient_scale_initial",
            "policy_trunk_gradient_scale_final",
            "policy_trunk_gradient_ramp_updates",
        ):
            config_values.pop(obsolete_key, None)
        if config_values.get("learning_rate") == LEGACY_LEARNING_RATE:
            config_values["learning_rate"] = DEFAULT_LEARNING_RATE
        if tuple(config_values.get("tier_top_k", ())) in (
            LEGACY_TIER_TOP_K,
            PREVIOUS_TIER_TOP_K,
        ):
            config_values["tier_top_k"] = DEFAULT_TIER_TOP_K
        if tuple(config_values.get("tier_epsilons", ())) == LEGACY_TIER_EPSILONS:
            config_values["tier_epsilons"] = DEFAULT_TIER_EPSILONS
        if config_values.get("early_max_training_decisions") == LEGACY_EARLY_MAX_TRAINING_DECISIONS:
            config_values["early_max_training_decisions"] = 4_096
        serialized_rosters = config_values.pop("tier_rosters", None)
        if serialized_rosters is None:
            evaluation_three = tuple(config_values.pop("three_player_tiers", (1, 3, 5)))
            evaluation_four = tuple(config_values.pop("four_player_tiers", (1, 2, 4, 5)))
            evaluation_five = tuple(config_values.pop("five_player_tiers", (1, 2, 3, 4, 5)))
            config_values["tier_rosters"] = TierRosterConfig(
                evaluation_three_player=evaluation_three,
                evaluation_four_player=evaluation_four,
                evaluation_five_player=evaluation_five,
                training_five_player=TrainingRosterPolicy(evaluation_five),
            )
        else:
            for legacy_key in (
                "three_player_tiers",
                "four_player_tiers",
                "five_player_tiers",
            ):
                config_values.pop(legacy_key, None)
            config_values["tier_rosters"] = TierRosterConfig.from_serialized(serialized_rosters)
        config = TrainingConfig(**config_values)
        trainer = cls(config=config)
        trainer.model._load_checkpoint_state(checkpoint, "Training checkpoint")
        optimizer_state = checkpoint["optimizer_state_dict"]
        if migrated_observation_schema:
            optimizer_state = trainer._migrate_observation_optimizer_state(optimizer_state)
        if checkpoint_version == LEGACY_Q_ONLY_CHECKPOINT_VERSION:
            trainer._load_q_only_optimizer_state(optimizer_state)
        elif trainer.model.migrated_shared_layer:
            trainer.optimizer.load_state_dict(
                trainer._migrate_shared_layer_optimizer_state(optimizer_state)
            )
        else:
            trainer.optimizer.load_state_dict(optimizer_state)
        trainer.optimizer.param_groups[0]["lr"] = config.learning_rate
        if len(trainer.optimizer.param_groups) > 1:
            trainer.optimizer.param_groups[1]["lr"] = (
                config.learning_rate * config.policy_head_lr_multiplier
            )
        progress_values = dict(checkpoint["training_progress"])
        progress_values.pop("last_move_pickup_ranking_sample_weighted_loss", None)
        progress_values.pop("last_move_pickup_ranking_raw_weighted_loss", None)
        if checkpoint_version == LEGACY_Q_ONLY_CHECKPOINT_VERSION:
            progress_values["policy_training_updates"] = 0
            progress_values["last_policy_loss"] = None
            progress_values["mean_policy_loss"] = None
        else:
            progress_values.setdefault("policy_training_updates", 0)
        trainer.progress = TrainingProgress(**progress_values)
        trainer.progress.checkpoint_loads += 1
        trainer.rng.setstate(checkpoint["policy_rng_state"])
        trainer.loss_total = checkpoint["loss_total"]
        trainer.policy_loss_total = (
            0.0
            if checkpoint_version == LEGACY_Q_ONLY_CHECKPOINT_VERSION
            else checkpoint.get("policy_loss_total", 0.0)
        )
        trainer.source_state_sha256 = checkpoint["source_state_sha256"]
        trainer.curriculum_state = checkpoint.get("curriculum_state")
        trainer.model.migrated_observation_schema = migrated_observation_schema
        trainer.model.eval()
        return trainer

    def _load_q_only_optimizer_state(self, legacy_state):
        """Preserve mature trunk/Q Adam state while initializing the new head."""
        self.optimizer.load_state_dict(
            self._migrate_shared_layer_optimizer_state(legacy_state, q_only=True)
        )

    def _migrate_shared_layer_optimizer_state(self, legacy_state, *, q_only=False):
        """Add neutral Adam state for the identity-initialized shared layer."""
        current = self.optimizer.state_dict()
        old_groups = legacy_state.get("param_groups", ())
        expected_group_count = 1 if q_only else 2
        if len(old_groups) != expected_group_count:
            raise ValueError(
                f"Legacy optimizer must contain {expected_group_count} parameter group(s)"
            )

        old_q_ids = tuple(old_groups[0]["params"])
        new_q_ids = tuple(current["param_groups"][0]["params"])
        shared_parameters = tuple(self.model.shared_layer3.parameters())
        shared_indices = tuple(
            index
            for index, parameter in enumerate(self._q_and_trunk_parameters)
            if any(parameter is shared for shared in shared_parameters)
        )
        legacy_q_count = len(new_q_ids) - len(shared_indices)
        if len(old_q_ids) not in (legacy_q_count, len(new_q_ids)):
            raise ValueError("Legacy optimizer Q/trunk parameter layout is incompatible")

        migrated_state = {}
        for old_id, new_id in zip(old_q_ids, new_q_ids):
            if old_id in legacy_state["state"]:
                migrated_state[new_id] = legacy_state["state"][old_id]

        if len(old_q_ids) == legacy_q_count:
            exemplar = next(iter(legacy_state["state"].values()), {})
            for index in shared_indices:
                migrated_state[new_q_ids[index]] = self._neutral_adam_state(
                    self._q_and_trunk_parameters[index], exemplar
                )

        migrated_groups = current["param_groups"]
        for key, value in old_groups[0].items():
            if key != "params":
                migrated_groups[0][key] = value
        if not q_only:
            old_policy_ids = tuple(old_groups[1]["params"])
            new_policy_ids = tuple(migrated_groups[1]["params"])
            if len(old_policy_ids) != len(new_policy_ids):
                raise ValueError("Legacy optimizer policy parameter layout is incompatible")
            for old_id, new_id in zip(old_policy_ids, new_policy_ids):
                if old_id in legacy_state["state"]:
                    migrated_state[new_id] = legacy_state["state"][old_id]
            for key, value in old_groups[1].items():
                if key != "params":
                    migrated_groups[1][key] = value
        return {"state": migrated_state, "param_groups": migrated_groups}

    @staticmethod
    def _neutral_adam_state(parameter, exemplar):
        step = exemplar.get("step")
        neutral = {
            "step": torch.zeros_like(step) if isinstance(step, torch.Tensor) else 0.0,
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
        if "max_exp_avg_sq" in exemplar:
            neutral["max_exp_avg_sq"] = torch.zeros_like(parameter)
        return neutral

    def _migrate_observation_optimizer_state(self, legacy_state):
        """Zero-expand Adam tensors associated with the legacy input layer."""
        groups = legacy_state.get("param_groups", ())
        if not groups or not groups[0].get("params"):
            return legacy_state
        layer1_index = next(
            index
            for index, parameter in enumerate(self._q_and_trunk_parameters)
            if parameter is self.model.layer1.weight
        )
        layer1_parameter_id = groups[0]["params"][layer1_index]
        layer1_state = legacy_state.get("state", {}).get(layer1_parameter_id)
        if not layer1_state:
            return legacy_state

        migrated_state = dict(legacy_state)
        migrated_entries = dict(legacy_state["state"])
        migrated_layer1 = dict(layer1_state)
        changed = False
        for key, value in layer1_state.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[0] != self.model.layer1.out_features
                or value.shape[1]
                not in (
                    LEGACY_OBSERVATION_SIZE,
                    LEGACY_OBSERVATION_SIZE_V3,
                    LEGACY_OBSERVATION_SIZE_V4,
                )
            ):
                continue
            expanded = value.new_zeros(self.model.layer1.weight.shape)
            expanded[:, : value.shape[1]].copy_(value)
            migrated_layer1[key] = expanded
            changed = True
        if not changed:
            return legacy_state
        migrated_entries[layer1_parameter_id] = migrated_layer1
        migrated_state["state"] = migrated_entries
        return migrated_state
