"""Exhaustive consistency checks for one reachable Hansa action state."""

from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
import io
import pickle

from game.action_codec import ACTION_SPACE_SIZE, DEFAULT_ACTION_CODEC
from game.action_schema import ACTION_SCHEMA_VERSION
from game.invariants import validate_game
from game.turn_state import TurnPhase


class ActionValidationError(AssertionError):
    """Raised when legality, codec, mask, or execution disagree."""


@dataclass(frozen=True)
class ActionValidationResult:
    legal_action_count: int
    enabled_index_count: int
    phase: TurnPhase


def state_summary(game) -> str:
    return (
        f"schema={ACTION_SCHEMA_VERSION}, seed={game.seed}, turn={game.turn_number}, "
        f"round={game.round_number}, "
        f"player={game.current_player_index}, active={game.active_player}, "
        f"phase={game.turn_phase.value}, actions={game.current_player.actions_remaining}"
    )


def state_fingerprint(game):
    """Serialize all engine state, including RNG and pending workflows."""
    return pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL)


def _fail(game, message, index=None, action=None):
    details = [message, state_summary(game)]
    if index is not None:
        details.append(f"index={index}")
    if action is not None:
        details.append(f"action={action!r}")
        try:
            details.append(f"description={DEFAULT_ACTION_CODEC.describe(index)}")
        except (TypeError, ValueError):
            pass
    raise ActionValidationError("; ".join(details))


def validate_action_state(game, quiet: bool = False) -> ActionValidationResult:
    """Exhaustively validate and execute every enabled interaction on clones.

    If `quiet` is True, game stdout will be captured and suppressed.
    """
    output = redirect_stdout(io.StringIO()) if quiet else nullcontext()
    with output:
        validate_game(game)
        legal_actions = tuple(game.get_legal_actions())

        if len(legal_actions) != len(set(legal_actions)):
            _fail(game, "duplicate structured legal actions")
        if game.turn_phase != TurnPhase.GAME_OVER and not legal_actions:
            _fail(game, "nonterminal state has no legal actions")
        if game.turn_phase == TurnPhase.GAME_OVER and legal_actions:
            _fail(game, "terminal state exposes gameplay actions")

        mask = game.ai_action_mask()
        if len(mask) != ACTION_SPACE_SIZE:
            _fail(game, f"mask length is {len(mask)}, expected {ACTION_SPACE_SIZE}")

        encoded = {}
        for action in legal_actions:
            index = DEFAULT_ACTION_CODEC.encode(action)
            if not mask[index]:
                _fail(game, "encoded legal action is disabled", index, action)
            if index in encoded:
                _fail(game, f"duplicate index also used by {encoded[index]!r}", index, action)
            encoded[index] = action
            if DEFAULT_ACTION_CODEC.decode(index) != action:
                _fail(game, "encode/decode changed action meaning", index, action)

        enabled = tuple(index for index, value in enumerate(mask) if value)
        if set(enabled) != set(encoded):
            _fail(game, "enabled mask indices differ from encoded legal actions")

        for index in range(ACTION_SPACE_SIZE):
            if DEFAULT_ACTION_CODEC.is_reserved(index) and mask[index]:
                _fail(game, "reserved index is enabled", index)

        snapshot = pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL)
        for index in enabled:
            action = DEFAULT_ACTION_CODEC.decode(index)
            if action not in legal_actions:
                _fail(game, "enabled index decodes to an illegal action", index, action)
            first = pickle.loads(snapshot)
            second = pickle.loads(snapshot)
            try:
                first.apply_ai_action(index)
                second.apply_ai_action(index)
                validate_game(first)
                validate_game(second)
            except Exception as error:
                _fail(
                    game, f"enabled action failed: {type(error).__name__}: {error}", index, action
                )
            try:
                if state_fingerprint(first) != state_fingerprint(second):
                    _fail(game, "enabled action produced nondeterministic state", index, action)
            except ActionValidationError:
                raise
            except Exception as error:
                _fail(
                    game,
                    f"could not compare resulting states: {type(error).__name__}: {error}",
                    index,
                    action,
                )

        return ActionValidationResult(len(legal_actions), len(enabled), game.turn_phase)
