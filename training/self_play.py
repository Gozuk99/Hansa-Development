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

from ai.ai_model import HansaNN, device
from ai.observation_encoder import ObservationEncoder
from ai.observation_schema import (
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
TRAINING_CHECKPOINT_VERSION = 5
DEFAULT_LEARNING_RATE = 0.0001
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
    early_max_training_decisions: int = 4_096
    full_validation_interval: int = 50
    income_penalty_scale: float = 100.0
    tier_top_k: tuple[int | None, ...] = DEFAULT_TIER_TOP_K
    tier_epsilons: tuple[float, ...] = DEFAULT_TIER_EPSILONS
    three_player_tiers: tuple[int, ...] = (1, 3, 5)
    four_player_tiers: tuple[int, ...] = (1, 2, 4, 5)
    five_player_tiers: tuple[int, ...] = (1, 2, 3, 4, 5)

    def __post_init__(self):
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
        if self.early_max_training_decisions < 1:
            raise ValueError("early maximum training decisions must be positive")
        if self.full_validation_interval < 1:
            raise ValueError("full validation interval must be positive")
        if self.income_penalty_scale < 0:
            raise ValueError("income penalty scale cannot be negative")
        if len(self.tier_top_k) != len(self.tier_epsilons):
            raise ValueError("tier top-k and epsilon settings must have equal lengths")
        if any(top_k is not None and top_k < 1 for top_k in self.tier_top_k):
            raise ValueError("tier top-k values must be positive")
        if any(not 0 <= epsilon <= 1 for epsilon in self.tier_epsilons):
            raise ValueError("tier epsilon values must be between 0 and 1")
        for player_count, tiers in self.tier_subsets().items():
            if len(tiers) != player_count or len(set(tiers)) != player_count:
                raise ValueError(f"{player_count}-player tiers must be unique")
            if any(tier < 1 or tier > len(self.tier_top_k) for tier in tiers):
                raise ValueError(f"{player_count}-player tier is undefined")

    def tier_subsets(self):
        return {
            3: self.three_player_tiers,
            4: self.four_player_tiers,
            5: self.five_player_tiers,
        }


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
    receives_terminal_credit: bool = True


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
    decisions: int = 0
    invalid_action_attempts: int = 0
    game_completion_failures: int = 0
    replacement_route_deadlocks: int = 0
    checkpoint_saves: int = 0
    checkpoint_loads: int = 0
    last_loss: float | None = None
    mean_loss: float | None = None
    tier_games: dict[int, int] = field(default_factory=dict)
    tier_wins: dict[int, int] = field(default_factory=dict)
    tier_selected_rank_total: dict[int, int] = field(default_factory=dict)
    tier_epsilon_selections: dict[int, int] = field(default_factory=dict)
    tier_top_k_selections: dict[int, int] = field(default_factory=dict)
    tier_immediate_reward_total: dict[int, float] = field(default_factory=dict)
    tier_reward_to_go_total: dict[int, float] = field(default_factory=dict)


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
            is_pickup = (
                post.owner is not None and post.owner is not game.current_player
                if opponent_pickups
                else post.owner is game.current_player
            )
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
    return tuple(
        replace(decision, reward_to_go=decision.local_training_target)
        if decision.local_training_target is not None
        else replace(
            decision,
            reward_to_go=decision.reward_to_go + decision.local_training_adjustment,
        )
        if decision.local_training_adjustment
        else decision
        for decision in completed
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
    """Penalize exact or non-maritime route-equivalent movement."""
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


class SelfPlayTrainer:
    """Collect frozen-policy games and update one shared action-value model afterward."""

    def __init__(self, model=None, config=None):
        self.model = model or HansaNN()
        self.config = config or TrainingConfig()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.encoder = ObservationEncoder()
        self.rng = random.Random(self.config.seed)
        self.progress = TrainingProgress()
        self.loss_total = 0.0
        self.source_state_sha256 = None
        self.curriculum_state = None
        self.last_training_sample_coverage = ()

    def _tier(self, number):
        return PolicyTier(
            number,
            self.config.tier_top_k[number - 1],
            self.config.tier_epsilons[number - 1],
        )

    def _assign_tiers(self, player_count):
        try:
            numbers = list(self.config.tier_subsets()[player_count])
        except KeyError as error:
            raise TrainingRunError(
                f"No tier subset is configured for {player_count} players"
            ) from error
        self.rng.shuffle(numbers)
        return tuple(self._tier(number) for number in numbers)

    def _assign_evaluation_tiers(self, player_count, rotation):
        try:
            numbers = list(self.config.tier_subsets()[player_count])
        except KeyError as error:
            raise TrainingRunError(
                f"No tier subset is configured for {player_count} players"
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
        """Transfer the small output once, then average equivalent interactions."""
        values = scores.tolist()
        means = []
        for group in groups:
            if len(group) == 1:
                means.append(values[group[0]])
                continue
            means.append(sum(values[index] for index in group) / len(group))
        return tuple(means)

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
            return ActionSelection(selected, False, 1, 1, group)
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
            return ActionSelection(selected, False, 1, 1, group)
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
    ):
        movement_metrics = movement_metrics or MovementBehaviorMetrics()
        trajectory = CompletedTrajectory(
            assign_training_targets(decisions, terminal_rewards, self.config.gamma),
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
    ) -> CompletedTrajectory:
        """Play one exact starting state without changing model weights."""
        play_started = perf_counter()
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
        post_contexts = _post_contexts_by_slot(game)
        post_routes = {post: route for _route_index, route, post in post_contexts}
        post_route_indices = {post: route_index for route_index, _route, post in post_contexts}
        seat_tiers = (
            self._assign_evaluation_tiers(len(game.players), evaluation_tier_rotation)
            if evaluation
            else self._assign_tiers(len(game.players))
        )
        decisions = []
        action_trace = []
        game_end_trigger_player = None
        pending_disruption = None
        tracked_turn = game.turn_number
        movement_metrics = MovementBehaviorMetrics()
        pending_move_claim_routes = [frozenset() for _player in game.players]
        rewarded_move_focus_routes = [frozenset() for _player in game.players]
        pending_terminal_move_workflows = []
        pending_terminal_completed_routes = set()
        consecutive_move_actions = 0
        turn_spent_actions = 0
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
        scoring_started = perf_counter()
        projected_before = game.projected_scores()
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
                        turn_spent_actions,
                    )
                    tracked_turn = game.turn_number
                    pending_move_claim_routes = [frozenset() for _player in game.players]
                    pending_terminal_move_workflows = []
                    pending_terminal_completed_routes = set()
                    consecutive_move_actions = 0
                    turn_spent_actions = 0
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
                    observation_started = perf_counter()
                    observation = self.encoder.build(game)
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
                            )
                        error = IncompleteGameError(
                            "The game has no legal interaction at "
                            f"turn {game.turn_number}, phase {game.turn_phase.value}"
                        )
                        raise error
                    legal_action_indices = _action_index_tuple(legal_indices)
                    inference_started = perf_counter()
                    scores = self.model(observation.features.float().unsqueeze(0).to(device))[0]
                    inference_seconds += perf_counter() - inference_started
                    selection_started = perf_counter()
                    tier = seat_tiers[observation.observer_index]
                    if game.turn_phase is TurnPhase.ACTIONS:
                        selection = self._select_action(
                            scores,
                            legal_action_indices,
                            tier,
                            action_phase_selection_groups(
                                game,
                                legal_action_indices,
                                post_contexts,
                            ),
                        )
                    else:
                        if game.turn_phase is TurnPhase.MOVE_PIECES:
                            exploration_categories = move_workflow_exploration_categories(
                                game,
                                legal_action_indices,
                                post_contexts=post_contexts,
                            )
                        elif (
                            game.turn_phase is TurnPhase.BONUS_MARKER_CHOICE
                            and game.waiting_for_bm_move3
                        ):
                            exploration_categories = move_workflow_exploration_categories(
                                game,
                                legal_action_indices,
                                opponent_pickups=True,
                                post_contexts=post_contexts,
                            )
                        else:
                            exploration_categories = None
                        selection = self._select_workflow_action(
                            scores,
                            legal_action_indices,
                            exploration_categories,
                        )
                    action_index = selection.action_index
                    action = _ACTIONS_BY_INDEX[action_index]
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
                    context_seconds += perf_counter() - context_started
                    end_was_pending = game.game_end or game.game_end_pending_immediate_resolution
                    turn_before = game.turn_number
                    action_trace.append(action_index)
                    action_attempted = True
                    execution_started = perf_counter()
                    game.apply_ai_action(action_index)
                    if move_tracking_active:
                        move_pieces_picked_up = max(
                            move_pieces_picked_up, len(acting_player.holding_pieces)
                        )
                    if move_placement_post is not None and move_placement_post.is_owned():
                        move_destination_posts.append(move_placement_post)
                    execution_seconds += perf_counter() - execution_started
                    if should_fully_validate(
                        action_number,
                        self.config.full_validation_interval,
                        turn_before,
                        action_phase,
                        game,
                    ):
                        validation_started = perf_counter()
                        validate_game(game)
                        validation_seconds += perf_counter() - validation_started
                except Exception as error:
                    if action_attempted:
                        self.progress.invalid_action_attempts += 1
                    if failure_callback is not None:
                        failure_callback(game, tuple(action_trace), seat_tiers, error)
                    raise
                scoring_started = perf_counter()
                projected_after = game.projected_scores()
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
                    next_consecutive_move = consecutive_move_actions + 1
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
                    turn_spent_actions += 1
                    newly_completed_routes = frozenset()
                    if normal_move_completed:
                        movement_metrics.move_action_count += 1
                        turn_move_workflow_ids.append(movement_workflow_id)
                        consecutive_move_actions += 1
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
                                rewarded_move_focus_routes[observation.observer_index],
                                route_focus_reward,
                            ) = move_route_focus_reward(
                                rewarded_move_focus_routes[observation.observer_index],
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
                    else:
                        consecutive_move_actions = 0
                    if action_phase is TurnPhase.ACTIONS and isinstance(action, RouteInteraction):
                        rewarded = set(rewarded_move_focus_routes[observation.observer_index])
                        rewarded.discard(action.route_slot)
                        rewarded_move_focus_routes[observation.observer_index] = frozenset(rewarded)
                    pending_routes, combo_reward = update_move_claim_combo(
                        pending_move_claim_routes[observation.observer_index],
                        action=action,
                        turn_phase=action_phase,
                        action_was_spent=True,
                        newly_completed_routes=newly_completed_routes,
                    )
                    pending_move_claim_routes[observation.observer_index] = pending_routes
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
                reward_seconds += perf_counter() - reward_started
                end_is_pending = game.game_end or game.game_end_pending_immediate_resolution
                if game_end_trigger_player is None and end_is_pending and not end_was_pending:
                    game_end_trigger_player = observation.observer_index
            else:
                finalize_all_move_turn(
                    decisions,
                    movement_metrics,
                    turn_move_workflow_ids,
                    turn_spent_actions,
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
                )

        finalize_all_move_turn(
            decisions,
            movement_metrics,
            turn_move_workflow_ids,
            turn_spent_actions,
        )
        validation_started = perf_counter()
        validate_game(game)
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

    def _training_batches(self, decisions, *, max_training_decisions=None):
        decisions = list(decisions)
        batch_size = self.config.decision_batch_size
        if max_training_decisions is None:
            max_training_decisions = self.config.normal_max_training_decisions
        if len(decisions) <= min(batch_size, max_training_decisions):
            self.rng.shuffle(decisions)
            return (decisions,)

        sample_size = min(len(decisions), max_training_decisions)
        batch_count = math.ceil(sample_size / batch_size)
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
        return (
            self.config.early_max_training_decisions
            if curriculum_maturity in {"early", "early_mixed"}
            else self.config.normal_max_training_decisions
        )

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
        losses = []
        coverage = []
        for trajectory, curriculum_maturity in zip(trajectories, curriculum_maturities):
            if curriculum_maturity in {"early", "early_mixed"}:
                batches = self._early_training_batches(trajectory.decisions)
                sampled_octiles = self._sampled_octiles(trajectory.decisions, batches)
            else:
                batches = self._training_batches(
                    trajectory.decisions,
                    max_training_decisions=self._trajectory_training_decision_cap(
                        curriculum_maturity
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
                observations = (
                    torch.stack([sample.observation for sample in batch]).float().to(device)
                )
                targets = torch.tensor(
                    [sample.reward_to_go for sample in batch], dtype=torch.float32, device=device
                )
                model_outputs = self.model(observations)
                decision_losses = torch.stack(
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
                        for row, sample in enumerate(batch)
                    ]
                )
                loss = decision_losses.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_gradient_norm
                )
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
                self.progress.training_updates += 1
        self.last_training_sample_coverage = tuple(coverage)
        self.model.eval()

        value = sum(losses) / len(losses)
        self.progress.last_loss = value
        self.loss_total += sum(losses)
        self.progress.mean_loss = self.loss_total / self.progress.training_updates
        return value

    def trajectory_loss(self, trajectory) -> float | None:
        """Measure one learning game's loss without updating the model."""
        samples = list(trajectory.decisions)
        if not samples:
            return None
        observations = torch.stack([sample.observation for sample in samples]).float().to(device)
        targets = torch.tensor(
            [sample.reward_to_go for sample in samples], dtype=torch.float32, device=device
        )
        self.model.eval()
        with torch.no_grad():
            model_outputs = self.model(observations)
            decision_losses = torch.stack(
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
                    for row, sample in enumerate(samples)
                ]
            )
            return decision_losses.mean().item()

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
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_progress": asdict(self.progress),
            "training_config": asdict(self.config),
            "source_state_sha256": sources,
            "policy_rng_state": self.rng.getstate(),
            "loss_total": self.loss_total,
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
        if checkpoint.get("training_checkpoint_version") != TRAINING_CHECKPOINT_VERSION:
            raise ValueError("Incompatible training checkpoint version")
        validate_action_schema_metadata(checkpoint, "Training checkpoint")
        migrated_observation_schema = validate_model_observation_schema_metadata(
            checkpoint, "Training checkpoint"
        )

        config_values = dict(checkpoint["training_config"])
        if config_values.get("learning_rate") == LEGACY_LEARNING_RATE:
            config_values["learning_rate"] = DEFAULT_LEARNING_RATE
        if tuple(config_values.get("tier_top_k", ())) == LEGACY_TIER_TOP_K:
            config_values["tier_top_k"] = DEFAULT_TIER_TOP_K
        if tuple(config_values.get("tier_epsilons", ())) == LEGACY_TIER_EPSILONS:
            config_values["tier_epsilons"] = DEFAULT_TIER_EPSILONS
        if config_values.get("early_max_training_decisions") == LEGACY_EARLY_MAX_TRAINING_DECISIONS:
            config_values["early_max_training_decisions"] = 4_096
        config = TrainingConfig(**config_values)
        trainer = cls(config=config)
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for parameter_group in trainer.optimizer.param_groups:
            parameter_group["lr"] = config.learning_rate
        trainer.progress = TrainingProgress(**checkpoint["training_progress"])
        trainer.progress.checkpoint_loads += 1
        trainer.rng.setstate(checkpoint["policy_rng_state"])
        trainer.loss_total = checkpoint["loss_total"]
        trainer.source_state_sha256 = checkpoint["source_state_sha256"]
        trainer.curriculum_state = checkpoint.get("curriculum_state")
        trainer.model.migrated_observation_schema = migrated_observation_schema
        trainer.model.eval()
        return trainer
