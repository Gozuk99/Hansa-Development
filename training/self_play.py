"""Shared-model training from exact Hansa starting positions."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import hashlib
import io
import math
from pathlib import Path
import random
import tempfile
from time import perf_counter

import torch
import torch.nn.functional as functional

from ai.ai_model import (
    MODEL_CHECKPOINT_FORMAT,
    MODEL_CHECKPOINT_VERSION,
    LEGACY_MODEL_CHECKPOINT_VERSION,
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
from game.structured_actions import IncomeInteraction, PostInteraction, RouteInteraction
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
_CURRICULUM_STATE_UNSET = object()
DEFAULT_TIER_TOP_K = (2, 5, 10, 15, 20)
DEFAULT_TIER_EPSILONS = (0.05, 0.10, 0.20, 0.35, 0.35)
FRESH_OPTIMIZER_UPDATES_PER_TRAJECTORY = 4
NORMAL_EXPLORATION_MODE = "normal"
ZERO_EPSILON_EXPLORATION_MODE = "zero_epsilon"
SHADOW_FILTER_POLICY_TOP_K = 10
SHADOW_FILTER_Q_TOP_K = 20
LEGACY_TIER_TOP_K = (2, 5, 10, 15, None)
LEGACY_TIER_EPSILONS = (0.05, 0.10, 0.20, 0.35, 1.00)
_ACTIONS_BY_INDEX = tuple(
    None if DEFAULT_ACTION_CODEC.is_reserved(index) else DEFAULT_ACTION_CODEC.decode(index)
    for index in range(ACTION_SPACE_SIZE)
)


@lru_cache(maxsize=None)
def inverse_sqrt_rank_weights(count):
    """Return unnormalized 1/sqrt(rank) weights for one-indexed ranks."""
    if count < 1:
        raise ValueError("Rank weight count must be positive")
    return tuple(1.0 / math.sqrt(rank) for rank in range(1, count + 1))


@lru_cache(maxsize=None)
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
    """Raised when a game remains unfinished at its configured action limit."""


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


@dataclass
class MovementBehaviorMetrics:
    """Per-game counters recorded at existing movement reward/penalty boundaries."""

    move_action_count: int = 0
    spent_action_count: int = 0
    pointless_move_workflows: int = 0
    repeated_move_penalties: int = 0
    all_move_turn_penalties: int = 0
    moves_creating_claimable_route: int = 0
    move_claim_conversions: int = 0

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


def finalize_all_move_turn(decisions, movement_metrics, workflow_ids, spent_actions):
    """Apply and record the all-Move target when the current turn closes."""
    applied = apply_all_move_turn_target(decisions, workflow_ids, spent_actions)
    movement_metrics.all_move_turn_penalties += int(applied)
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
    if pieces_moved == 1:
        return -200.0
    if pieces_moved == 2 and movement_capacity >= 3:
        return -100.0
    return 0.0


def consecutive_move_penalty(movement_capacity, consecutive_moves):
    """Penalize implausible repeated normal Move actions within one turn."""
    if consecutive_moves >= 3:
        return float(REPEATED_MOVE_LOCAL_TARGET)
    if movement_capacity >= 4 and consecutive_moves >= 2:
        return float(CONSECUTIVE_HIGH_CAPACITY_MOVE_PENALTY)
    return 0.0


def _is_normal_move_in_progress(action_phase, player):
    return action_phase is TurnPhase.MOVE_PIECES and bool(player.holding_pieces)


def pointless_movement_penalty(origin_pieces, destination_posts, post_routes=None):
    """Penalize exact or indistinguishable non-maritime movement."""
    if not origin_pieces or len(origin_pieces) != len(destination_posts):
        return 0.0
    original = {post: (owner, shape) for post, owner, shape in origin_pieces}
    if len(original) != len(origin_pieces):
        return 0.0
    exact_no_change = set(destination_posts) == set(original) and all(
        post.owner is owner and post.owner_piece_shape == shape
        for post, (owner, shape) in original.items()
    )
    if exact_no_change:
        return float(POINTLESS_MOVEMENT_LOCAL_TARGET)
    if len({(owner, shape) for _, owner, shape in origin_pieces}) > 1:
        return 0.0
    if post_routes is None:
        return 0.0
    origin_routes = [post_routes.get(post) for post in original]
    destination_routes = [post_routes.get(post) for post in destination_posts]
    involved_routes = set(origin_routes + destination_routes)
    if None in involved_routes or any(route.required_circles for route in involved_routes):
        return 0.0
    before = Counter((post_routes[post], owner, shape) for post, owner, shape in origin_pieces)
    after = Counter(
        (post_routes[post], post.owner, post.owner_piece_shape) for post in destination_posts
    )
    if before == after:
        return float(POINTLESS_MOVEMENT_LOCAL_TARGET)
    return 0.0


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
        self.curriculum_state = None
        self.last_training_sample_coverage = ()

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

    def _accumulate_independent_losses(self, q_loss, policy_loss, scale=1.0):
        """Accumulate scaled Q and isolated-policy gradients without clipping or stepping."""
        (scale * q_loss).backward()
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

    def _select_action(self, scores, legal_indices, tier, equivalent_groups=None):
        legal_list = _action_index_tuple(legal_indices)
        groups = (
            tuple((index,) for index in legal_list)
            if equivalent_groups is None
            else equivalent_groups
        )
        self._validate_action_groups(legal_list, groups, "Action groups")
        group_count = len(groups)
        if group_count == 1:
            group = groups[0]
            selected = group[0] if len(group) == 1 else group[self.rng.randrange(len(group))]
            return ActionSelection(
                selected,
                False,
                1,
                1,
                group,
                self._group_mean_scores(scores, groups),
            )
        group_scores = self._group_mean_scores(scores, groups)
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

    def _select_workflow_action(self, scores, legal_indices, exploration_categories=None):
        """Keep multi-click workflows coherent while retaining bounded exploration."""
        legal_list = _action_index_tuple(legal_indices)
        has_exploration_categories = exploration_categories is not None
        categories = (
            (tuple((index,) for index in legal_list),)
            if exploration_categories is None
            else exploration_categories
        )
        candidate_groups = tuple(group for category in categories for group in category)
        self._validate_action_groups(
            legal_list,
            candidate_groups,
            "Workflow exploration categories",
        )
        candidate_count = len(candidate_groups)
        if candidate_count == 1:
            group = candidate_groups[0]
            selected = group[0] if len(group) == 1 else group[self.rng.randrange(len(group))]
            return ActionSelection(
                selected,
                False,
                1,
                1,
                group,
                self._group_mean_scores(scores, candidate_groups),
            )
        candidate_scores = self._group_mean_scores(scores, candidate_groups)

        ranked_positions = self._rank_legal_positions(candidate_scores, min(3, candidate_count))
        roll = self.rng.random()
        used_epsilon = False
        if candidate_count == 2:
            selected_rank = 0 if roll < 0.60 else 1
            selected_group_position = ranked_positions[selected_rank]
            model_rank = selected_rank + 1
        elif roll < 0.40:
            selected_group_position = ranked_positions[0]
            model_rank = 1
        elif roll < 0.60:
            selected_group_position = ranked_positions[1]
            model_rank = 2
        elif roll < 0.75:
            selected_group_position = ranked_positions[2]
            model_rank = 3
        else:
            if has_exploration_categories:
                category_position = self.rng.randrange(len(categories))
                category = categories[category_position]
                group_position = self.rng.randrange(len(category))
                selected_group_position = (
                    sum(len(previous) for previous in categories[:category_position])
                    + group_position
                )
            else:
                selected_group_position = self.rng.randrange(candidate_count)
            model_rank = self._model_rank(candidate_scores, selected_group_position)
            used_epsilon = True

        selected_group = candidate_groups[selected_group_position]
        selected_index = (
            selected_group[0]
            if len(selected_group) == 1
            else selected_group[self.rng.randrange(len(selected_group))]
        )
        return ActionSelection(
            selected_index,
            used_epsilon,
            model_rank,
            candidate_count,
            selected_group,
            candidate_scores,
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
                    if game.turn_phase is TurnPhase.ACTIONS:
                        equivalent_groups = action_phase_selection_groups(
                            game,
                            legal_action_indices,
                            post_contexts,
                        )
                        semantic_action_groups = equivalent_groups or tuple(
                            (index,) for index in legal_action_indices
                        )
                        selection = self._select_action(
                            scores,
                            legal_action_indices,
                            tier,
                            equivalent_groups,
                        )
                    else:
                        if game.turn_phase is TurnPhase.MOVE_PIECES:
                            exploration_categories = move_workflow_exploration_categories(
                                game,
                                legal_action_indices,
                                post_contexts=post_contexts,
                            )
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
                        selection = self._select_workflow_action(
                            scores,
                            legal_action_indices,
                            exploration_categories,
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
                movement_local_target = None
                movement_local_adjustment = 0.0
                if normal_move_completed and action_was_spent:
                    next_consecutive_move = acting_player.consecutive_paid_move_actions
                    repeated_move_penalty = consecutive_move_penalty(
                        movement_capacity, next_consecutive_move
                    )
                    no_change_penalty = pointless_movement_penalty(
                        move_origin_pieces,
                        move_destination_posts,
                        post_routes,
                    )
                    if next_consecutive_move >= 3 or no_change_penalty:
                        movement_local_target = repeated_move_penalty + no_change_penalty
                    else:
                        movement_local_adjustment = (
                            repeated_move_penalty
                            + movement_efficiency_penalty(pieces_moved, movement_capacity)
                        )
                elif permanent_move_completed:
                    no_change_penalty = pointless_movement_penalty(
                        move_origin_pieces,
                        move_destination_posts,
                        post_routes,
                    )
                    movement_local_target = no_change_penalty or None
                movement_metrics.pointless_move_workflows += int(bool(no_change_penalty))
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
                    if normal_move_completed:
                        movement_metrics.move_action_count += 1
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
                        movement_metrics.moves_creating_claimable_route += int(
                            bool(newly_completed_routes)
                        )
                        adjusted = list(player_reward_deltas)
                        if movement_local_target is None:
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
                        newly_completed_routes=newly_completed_routes,
                    )
                    acting_player.pending_move_claim_route_slots = pending_routes
                    if combo_reward:
                        movement_metrics.move_claim_conversions += 1
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
                        mask.to(torch.uint8),
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
                            group for group in semantic_action_groups if len(group) > 1
                        ),
                        receives_terminal_credit=not (
                            starts_normal_move or normal_move_in_progress
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
                )
                self.progress.game_completion_failures += 1
                error = ActionLimitExceeded(
                    f"Game did not finish within {self.config.max_actions} actions"
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

    def _decision_batch_losses(self, batch):
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
        q_loss = decision_losses.mean()
        quality_signals = policy_quality_signal(
            targets,
            self.config.policy_return_scale,
        )
        policy_loss = policy_batch_losses(
            model_outputs.policy_logits,
            batch,
            quality_signals,
        ).mean()
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
        detached_losses = torch.zeros(3, dtype=torch.float32, device=device)
        for start in range(0, effective_size, microbatch_size):
            microbatch = batch[start : start + microbatch_size]
            scale = len(microbatch) / effective_size
            q_loss, policy_loss, total_loss = self._decision_batch_losses(microbatch)
            self._accumulate_independent_losses(q_loss, policy_loss, scale)
            detached_losses += scale * torch.stack(
                (q_loss.detach(), policy_loss.detach(), total_loss.detach())
            )
        self._clip_independent_gradients()
        self.optimizer.step()
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
        sources = {str(Path(source)): _file_sha256(Path(source)) for source in starting_states}
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
        if tuple(config_values.get("tier_top_k", ())) == LEGACY_TIER_TOP_K:
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
