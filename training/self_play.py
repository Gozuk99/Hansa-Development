"""Shared-model training from exact Hansa starting positions."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, field, replace
import hashlib
import io
from pathlib import Path
import random
import tempfile

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
from game.structured_actions import PostInteraction
from game.turn_state import TurnPhase


TRAINING_CHECKPOINT_FORMAT = "hansa-shared-q-training"
TRAINING_CHECKPOINT_VERSION = 3
PRESTIGE_REWARD_MULTIPLIER = 100
END_GAME_WINNER_BONUS = 150


class TrainingRunError(RuntimeError):
    """Raised when a training game cannot safely continue."""


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.0001
    max_actions: int = 500
    disable_move_action: bool = True
    seed: int = 124
    gamma: float = 0.99
    tier_top_k: tuple[int | None, ...] = (2, 5, 10, 15, None)
    tier_epsilons: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 1.00)
    three_player_tiers: tuple[int, ...] = (1, 3, 5)
    four_player_tiers: tuple[int, ...] = (1, 2, 4, 5)
    five_player_tiers: tuple[int, ...] = (1, 2, 3, 4, 5)

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_actions < 1:
            raise ValueError("max_actions must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
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
    reward_to_go: float | None = None


@dataclass(frozen=True)
class CompletedTrajectory:
    decisions: tuple[TrainingDecision, ...]
    terminal_rewards: tuple[float, ...]
    final_scores: tuple[int, ...]
    winner_indices: tuple[int, ...]
    action_trace: tuple[int, ...]
    seat_tiers: tuple[int, ...]


@dataclass
class TrainingProgress:
    completed_games: int = 0
    training_updates: int = 0
    decisions: int = 0
    invalid_action_attempts: int = 0
    game_completion_failures: int = 0
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


def training_action_mask(game, *, disable_move_action: bool) -> torch.Tensor:
    """Prefer non-Move interactions, restoring Move when it is the only legal choice."""
    mask = torch.tensor(game.ai_action_mask(), dtype=torch.bool)
    if not disable_move_action or game.turn_phase is not TurnPhase.ACTIONS:
        return mask

    acting_player = game.players[game.active_player]
    original_mask = mask.clone()
    for index in mask.nonzero(as_tuple=False).flatten().tolist():
        action = DEFAULT_ACTION_CODEC.decode(index)
        if isinstance(action, PostInteraction):
            post = _post_at(game, action.post_slot)
            if post is not None and post.owner is acting_player:
                mask[index] = False
    return mask if mask.any() else original_mask


def assign_reward_to_go(decisions, terminal_rewards, gamma):
    """Discount each player's reward stream without leaking another player's reward."""
    if not decisions:
        return ()
    reward_events = [list(decision.player_reward_deltas) for decision in decisions]
    for player_index, reward in enumerate(terminal_rewards):
        reward_events[-1][player_index] += reward
    running = [0.0] * len(terminal_rewards)
    completed = list(decisions)
    for index in range(len(decisions) - 1, -1, -1):
        running = [
            reward_events[index][player_index] + gamma * running[player_index]
            for player_index in range(len(running))
        ]
        player_index = decisions[index].acting_player_index
        completed[index] = replace(decisions[index], reward_to_go=running[player_index])
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

    def _select_action(self, scores, legal_indices, tier):
        legal_scores = scores[legal_indices]
        ranked_positions = torch.argsort(legal_scores, descending=True, stable=True).tolist()
        ranked = [legal_indices[position] for position in ranked_positions]
        if len(ranked) == 1:
            return ActionSelection(ranked[0], False, 1, 1)
        if self.rng.random() < tier.epsilon:
            selected = self.rng.choice(legal_indices)
            used_epsilon = True
        else:
            effective_k = min(tier.top_k or len(ranked), len(ranked))
            selected = self.rng.choice(ranked[:effective_k])
            used_epsilon = False
        return ActionSelection(
            selected,
            used_epsilon,
            ranked.index(selected) + 1,
            len(legal_indices),
        )

    def collect_game(self, starting_state, *, quiet=True) -> CompletedTrajectory:
        """Play one exact starting state without changing model weights."""
        game = load_game(starting_state)
        game.interactive_errors = False
        seat_tiers = self._assign_tiers(len(game.players))
        decisions = []
        action_trace = []
        game_end_trigger_player = None
        output = redirect_stdout(io.StringIO()) if quiet else nullcontext()

        self.model.eval()
        with output, torch.no_grad():
            for _ in range(self.config.max_actions):
                if game.game_end:
                    break
                observation = self.encoder.build(game)
                mask = training_action_mask(
                    game, disable_move_action=self.config.disable_move_action
                )
                legal_indices = mask.nonzero(as_tuple=False).flatten().tolist()
                if not legal_indices:
                    self.progress.game_completion_failures += 1
                    raise TrainingRunError(
                        "The training policy removed every legal interaction at "
                        f"turn {game.turn_number}, phase {game.turn_phase.value}"
                    )
                scores = self.model(observation.features.float().unsqueeze(0).to(device))[0]
                tier = seat_tiers[observation.observer_index]
                selection = self._select_action(scores, legal_indices, tier)
                action_index = selection.action_index
                projected_before = game.projected_scores()
                end_was_pending = game.game_end or game.game_end_pending_immediate_resolution
                action_trace.append(action_index)
                try:
                    game.apply_ai_action(action_index)
                    validate_game(game)
                except Exception:
                    self.progress.invalid_action_attempts += 1
                    raise
                projected_after = game.projected_scores()
                player_reward_deltas = tuple(
                    float(PRESTIGE_REWARD_MULTIPLIER * (after - before))
                    for before, after in zip(projected_before, projected_after)
                )
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
                    )
                )
                end_is_pending = game.game_end or game.game_end_pending_immediate_resolution
                if game_end_trigger_player is None and end_is_pending and not end_was_pending:
                    game_end_trigger_player = observation.observer_index
            else:
                self.progress.game_completion_failures += 1
                raise TrainingRunError(
                    f"Game did not finish within {self.config.max_actions} actions"
                )

        winners = tuple(player.order - 1 for player in game.end_the_game())
        terminal_rewards = calculate_terminal_rewards(game, winners, game_end_trigger_player)
        completed_decisions = assign_reward_to_go(decisions, terminal_rewards, self.config.gamma)
        trajectory = CompletedTrajectory(
            completed_decisions,
            terminal_rewards,
            tuple(player.final_score for player in game.players),
            winners,
            tuple(action_trace),
            tuple(tier.number for tier in seat_tiers),
        )
        self.progress.completed_games += 1
        self.progress.decisions += len(decisions)
        self._record_tier_metrics(trajectory)
        return trajectory

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
        """Perform one Monte Carlo action-value update between completed games."""
        samples = [decision for trajectory in trajectories for decision in trajectory.decisions]
        if not samples:
            raise TrainingRunError("Cannot train from an empty trajectory batch")

        observations = torch.stack([sample.observation for sample in samples]).float().to(device)
        actions = torch.tensor([sample.action_index for sample in samples], device=device)
        targets = torch.tensor(
            [sample.reward_to_go for sample in samples], dtype=torch.float32, device=device
        )

        self.model.train()
        predictions = self.model(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = functional.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()

        value = float(loss.detach().cpu())
        self.progress.training_updates += 1
        self.progress.last_loss = value
        self.loss_total += value
        self.progress.mean_loss = self.loss_total / self.progress.training_updates
        return value

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

        batch = []
        trajectories = []
        starting_game_count = self.progress.completed_games
        for episode in range(episodes):
            state = states[(starting_game_count + episode) % len(states)]
            trajectory = self.collect_game(state, quiet=quiet)
            trajectories.append(trajectory)
            batch.append(trajectory)
            if len(batch) == batch_size or episode == episodes - 1:
                self.update_model(batch)
                batch.clear()
        return tuple(trajectories)

    def save_checkpoint(self, path, starting_states):
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

        config = TrainingConfig(**checkpoint["training_config"])
        trainer = cls(config=config)
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.progress = TrainingProgress(**checkpoint["training_progress"])
        trainer.progress.checkpoint_loads += 1
        trainer.rng.setstate(checkpoint["policy_rng_state"])
        trainer.loss_total = checkpoint["loss_total"]
        trainer.source_state_sha256 = checkpoint["source_state_sha256"]
        trainer.model.eval()
        return trainer
