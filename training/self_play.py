"""Shared-model training from exact Hansa starting positions."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, field, replace
import hashlib
import io
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
    validate_observation_schema_metadata,
)
from game.action_codec import DEFAULT_ACTION_CODEC
from game.action_schema import action_schema_metadata, validate_action_schema_metadata
from game.invariants import validate_game
from game.persistence import load_game
from game.structured_actions import IncomeInteraction, PostInteraction, RouteInteraction
from game.turn_state import TurnPhase
from map_data.constants import ACTIONS_MAX_VALUES, DARK_GREEN, UPGRADE_MAX_VALUES


TRAINING_CHECKPOINT_FORMAT = "hansa-shared-q-training"
TRAINING_CHECKPOINT_VERSION = 5
DEFAULT_LEARNING_RATE = 0.00001
LEGACY_LEARNING_RATE = 0.0001
PRESTIGE_REWARD_MULTIPLIER = 100
END_GAME_WINNER_BONUS = 150
NO_REPLACEMENT_ROUTE_PENALTY = -500
MOVE_ROUTE_FOCUS_REWARD = 25
MOVE_BLOCK_REWARD = 25
ROUTE_COMPLETION_REWARD = 50
MOVE_COMPLETED_ROUTE_REWARD = 70
ROUTE_BUILDING_PLACEMENT_REWARD = 5
ROUTE_BUILDING_DISPLACEMENT_REWARD = 3
INTERMEDIATE_ABILITY_UPGRADE_REWARD = 250
FIRST_ACTIONS_UPGRADE_REWARD = 400
INTERMEDIATE_REWARDED_ABILITIES = ("privilege", "book", "actions", "bank")
MASSIVE_MOVE_PENALTY = -500
CONSECUTIVE_HIGH_CAPACITY_MOVE_PENALTY = -100
_CURRICULUM_STATE_UNSET = object()
DEFAULT_TIER_TOP_K = (2, 5, 10, 15, 20)
DEFAULT_TIER_EPSILONS = (0.05, 0.10, 0.20, 0.35, 0.35)
LEGACY_TIER_TOP_K = (2, 5, 10, 15, None)
LEGACY_TIER_EPSILONS = (0.05, 0.10, 0.20, 0.35, 1.00)


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
    reward_to_go: float | None = None


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


def _post_at(game, slot):
    posts = (post for route in game.selected_map.routes for post in route.posts)
    return next((post for index, post in enumerate(posts) if index == slot), None)


def _post_context_at(game, slot):
    posts = (
        (route_index, route, post)
        for route_index, route in enumerate(game.selected_map.routes)
        for post in route.posts
    )
    return next((item for index, item in enumerate(posts) if index == slot), None)


def _would_complete_east_west(game, player, route):
    if player in game.players_who_completed_east_west:
        return False
    occupied = {city for city in game.selected_map.cities if city.has_office_controlled_by(player)}
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
    for index in mask.nonzero(as_tuple=False).flatten().tolist():
        action = DEFAULT_ACTION_CODEC.decode(index)
        if isinstance(action, PostInteraction):
            post = _post_at(game, action.post_slot)
            if post is not None and post.owner is acting_player:
                mask[index] = False
    return mask if mask.any() else original_mask


def assign_reward_to_go(decisions, terminal_rewards, gamma):
    """Discount reward streams once per player turn, not once per interaction."""
    if not decisions:
        return ()
    running = [float(reward) for reward in terminal_rewards]
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
                running[player_index] *= gamma ** (latest_turns[player_index] - turns_started)
            running[player_index] += reward
            latest_turns[player_index] = turns_started
        player_index = decision.acting_player_index
        completed[index] = replace(decision, reward_to_go=running[player_index])
    return tuple(completed)


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
    if movement_capacity == 2 and consecutive_moves == 3:
        return float(MASSIVE_MOVE_PENALTY)
    if movement_capacity >= 4 and consecutive_moves >= 2:
        return float(CONSECUTIVE_HIGH_CAPACITY_MOVE_PENALTY)
    return 0.0


def completed_route_move_reward(routes_before, routes_after):
    """Reward net claimable routes created by one completed normal Move."""
    return float(MOVE_COMPLETED_ROUTE_REWARD * (len(routes_after) - len(routes_before)))


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


def apply_movement_efficiency_penalty(
    reward_deltas,
    *,
    acting_player_index,
    movement_capacity,
    pieces_moved,
    normal_move_completed,
):
    """Apply a normal-Move penalty only when its final piece has been placed."""
    if not normal_move_completed:
        return tuple(reward_deltas)
    adjusted = list(reward_deltas)
    adjusted[acting_player_index] += movement_efficiency_penalty(pieces_moved, movement_capacity)
    return tuple(adjusted)


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

    def _select_action(self, scores, legal_indices, tier):
        legal_scores = scores[legal_indices]
        legal_count = len(legal_indices)
        if legal_count == 1:
            return ActionSelection(legal_indices[0], False, 1, 1)
        if self.rng.random() < tier.epsilon:
            selected = self.rng.choice(legal_indices)
            used_epsilon = True
            selected_position = legal_indices.index(selected)
            selected_score = legal_scores[selected_position]
            model_rank = (
                int((legal_scores > selected_score).sum().item())
                + int((legal_scores[:selected_position] == selected_score).sum().item())
                + 1
            )
        else:
            effective_k = min(tier.top_k or legal_count, legal_count)
            ranked_positions = torch.topk(
                legal_scores, effective_k, largest=True, sorted=True
            ).indices.tolist()
            ranked = [legal_indices[position] for position in ranked_positions]
            selected = self.rng.choice(ranked)
            used_epsilon = False
            model_rank = ranked.index(selected) + 1
        return ActionSelection(
            selected,
            used_epsilon,
            model_rank,
            legal_count,
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
    ):
        trajectory = CompletedTrajectory(
            assign_reward_to_go(decisions, terminal_rewards, self.config.gamma),
            tuple(terminal_rewards),
            tuple(final_scores),
            tuple(winner_indices),
            tuple(action_trace),
            tuple(tier.number for tier in seat_tiers),
            reason,
            *(timings or (0.0,) * 10),
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
        game.interactive_errors = False
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
        turn_action_budget = game.current_player.actions_remaining
        turn_actions_spent = 0
        turn_move_actions = 0
        consecutive_move_actions = 0
        massive_move_penalty_applied = False
        move_destination_counts = {}
        move_completed_routes_before = set()
        move_tracking_active = False
        output = redirect_stdout(io.StringIO()) if quiet else nullcontext()

        self.model.eval()
        with output, torch.inference_mode():
            for action_number in range(1, self.config.max_actions + 1):
                if game.game_end:
                    break
                if game.turn_number != tracked_turn:
                    tracked_turn = game.turn_number
                    turn_action_budget = game.current_player.actions_remaining
                    turn_actions_spent = 0
                    turn_move_actions = 0
                    consecutive_move_actions = 0
                    massive_move_penalty_applied = False
                    move_destination_counts = {}
                    move_completed_routes_before = set()
                    move_tracking_active = False
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
                    )
                    legal_indices = mask.nonzero(as_tuple=False).flatten().tolist()
                    legality_seconds += perf_counter() - legality_started
                    if not legal_indices:
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
                            scoring_started = perf_counter()
                            projected_scores = game.projected_scores()
                            scoring_seconds += perf_counter() - scoring_started
                            return self._complete_trajectory(
                                decisions,
                                terminal_rewards,
                                projected_scores,
                                (),
                                action_trace,
                                seat_tiers,
                                reason="no_replacement_route",
                                completed=False,
                                timings=timings(),
                            )
                        raise IncompleteGameError(
                            "The game has no legal interaction at "
                            f"turn {game.turn_number}, phase {game.turn_phase.value}"
                        )
                    inference_started = perf_counter()
                    scores = self.model(observation.features.float().unsqueeze(0).to(device))[0]
                    inference_seconds += perf_counter() - inference_started
                    selection_started = perf_counter()
                    tier = seat_tiers[observation.observer_index]
                    selection = self._select_action(scores, legal_indices, tier)
                    action_index = selection.action_index
                    action = DEFAULT_ACTION_CODEC.decode(action_index)
                    selection_seconds += perf_counter() - selection_started
                    context_started = perf_counter()
                    action_phase = game.turn_phase
                    acting_player = game.players[observation.observer_index]
                    context = None
                    if action_phase is TurnPhase.ACTIONS and isinstance(action, PostInteraction):
                        context = _post_context_at(game, action.post_slot)
                        if context is not None:
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
                    normal_move_in_progress = bool(acting_player.holding_pieces) and (
                        action_phase is TurnPhase.ACTIONS
                    )
                    movement_capacity = acting_player.book
                    pieces_moved = movement_capacity - acting_player.pieces_to_pickup
                    actions_remaining_before = acting_player.actions_remaining
                    move_placement_route = None
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
                            next_player = game.players[
                                (observation.observer_index + 1) % len(game.players)
                            ]
                            move_blocks_next_player = bool(route.posts) and all(
                                post is selected_post or post.owner is next_player
                                for post in route.posts
                            )
                    elif (
                        not acting_player.holding_pieces
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
                    general_stock_before = (
                        acting_player.general_stock_squares + acting_player.general_stock_circles
                    )
                    abilities_before = tuple(
                        acting_player.actions_index
                        if ability == "actions"
                        else getattr(acting_player, ability)
                        for ability in INTERMEDIATE_REWARDED_ABILITIES
                    )
                    context_seconds += perf_counter() - context_started
                    scoring_started = perf_counter()
                    projected_before = game.projected_scores()
                    scoring_seconds += perf_counter() - scoring_started
                    end_was_pending = game.game_end or game.game_end_pending_immediate_resolution
                    turn_before = game.turn_number
                    action_trace.append(action_index)
                    action_attempted = True
                    execution_started = perf_counter()
                    game.apply_ai_action(action_index)
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
                if route_building_reward and route_building_post.owner is acting_player:
                    adjusted = list(player_reward_deltas)
                    adjusted[observation.observer_index] += route_building_reward
                    player_reward_deltas = tuple(adjusted)
                player_reward_deltas = apply_movement_efficiency_penalty(
                    player_reward_deltas,
                    acting_player_index=observation.observer_index,
                    movement_capacity=movement_capacity,
                    pieces_moved=pieces_moved,
                    normal_move_completed=(
                        normal_move_in_progress and not acting_player.holding_pieces
                    ),
                )
                normal_move_completed = normal_move_in_progress and not acting_player.holding_pieces
                if move_placement_route is not None:
                    move_destination_counts[move_placement_route] = (
                        move_destination_counts.get(move_placement_route, 0) + 1
                    )
                    if move_blocks_next_player:
                        adjusted = list(player_reward_deltas)
                        adjusted[observation.observer_index] += MOVE_BLOCK_REWARD
                        player_reward_deltas = tuple(adjusted)
                action_was_spent = acting_player.actions_remaining < actions_remaining_before
                if action_was_spent:
                    turn_actions_spent += 1
                    if normal_move_completed:
                        turn_move_actions += 1
                        consecutive_move_actions += 1
                        adjusted = list(player_reward_deltas)
                        adjusted[observation.observer_index] += consecutive_move_penalty(
                            movement_capacity, consecutive_move_actions
                        )
                        if any(count >= 2 for count in move_destination_counts.values()):
                            adjusted[observation.observer_index] += MOVE_ROUTE_FOCUS_REWARD
                        if move_tracking_active:
                            completed_routes_after = {
                                route_index
                                for route_index, route in enumerate(game.selected_map.routes)
                                if route.is_controlled_by(acting_player)
                            }
                            adjusted[observation.observer_index] += completed_route_move_reward(
                                move_completed_routes_before, completed_routes_after
                            )
                        if movement_capacity == 2 and consecutive_move_actions == 3:
                            massive_move_penalty_applied = True
                        if (
                            turn_action_budget >= 5
                            and turn_actions_spent >= turn_action_budget
                            and turn_move_actions == turn_actions_spent
                            and not massive_move_penalty_applied
                        ):
                            adjusted[observation.observer_index] += MASSIVE_MOVE_PENALTY
                            massive_move_penalty_applied = True
                        player_reward_deltas = tuple(adjusted)
                        move_destination_counts = {}
                        move_completed_routes_before = set()
                        move_tracking_active = False
                    else:
                        consecutive_move_actions = 0
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
                    )
                )
                reward_seconds += perf_counter() - reward_started
                end_is_pending = game.game_end or game.game_end_pending_immediate_resolution
                if game_end_trigger_player is None and end_is_pending and not end_was_pending:
                    game_end_trigger_player = observation.observer_index
            else:
                self.progress.game_completion_failures += 1
                error = ActionLimitExceeded(
                    f"Game did not finish within {self.config.max_actions} actions"
                )
                if failure_callback is not None:
                    failure_callback(game, tuple(action_trace), seat_tiers, error)
                raise error

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

    def update_model(self, trajectories) -> float:
        """Give each completed game one equally weighted model update."""
        trajectories = tuple(trajectories)
        if not trajectories or any(not trajectory.decisions for trajectory in trajectories):
            raise TrainingRunError("Cannot train from an empty trajectory batch")
        self.model.train()
        losses = []
        for trajectory in trajectories:
            decisions = list(trajectory.decisions)
            if len(decisions) > self.config.decision_batch_size:
                rewarded_indices = sorted(
                    (
                        index
                        for index, decision in enumerate(decisions[:-1])
                        if decision.immediate_reward
                    ),
                    key=lambda index: abs(decisions[index].immediate_reward),
                    reverse=True,
                )[: self.config.decision_batch_size // 2]
                selected = set(rewarded_indices)
                remaining_indices = [
                    index for index in range(len(decisions) - 1) if index not in selected
                ]
                slots = self.config.decision_batch_size - len(rewarded_indices) - 1
                selected.update(self.rng.sample(remaining_indices, slots))
                batch = [decisions[index] for index in selected] + [decisions[-1]]
            else:
                batch = decisions
            self.rng.shuffle(batch)
            observations = torch.stack([sample.observation for sample in batch]).float().to(device)
            actions = torch.tensor([sample.action_index for sample in batch], device=device)
            targets = torch.tensor(
                [sample.reward_to_go for sample in batch], dtype=torch.float32, device=device
            )
            predictions = self.model(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = functional.smooth_l1_loss(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            self.progress.training_updates += 1
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
        actions = torch.tensor([sample.action_index for sample in samples], device=device)
        targets = torch.tensor(
            [sample.reward_to_go for sample in samples], dtype=torch.float32, device=device
        )
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
            return functional.smooth_l1_loss(predictions, targets).item()

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
        validate_observation_schema_metadata(checkpoint, "Training checkpoint")

        config_values = dict(checkpoint["training_config"])
        if config_values.get("learning_rate") == LEGACY_LEARNING_RATE:
            config_values["learning_rate"] = DEFAULT_LEARNING_RATE
        if tuple(config_values.get("tier_top_k", ())) == LEGACY_TIER_TOP_K:
            config_values["tier_top_k"] = DEFAULT_TIER_TOP_K
        if tuple(config_values.get("tier_epsilons", ())) == LEGACY_TIER_EPSILONS:
            config_values["tier_epsilons"] = DEFAULT_TIER_EPSILONS
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
        trainer.model.eval()
        return trainer
