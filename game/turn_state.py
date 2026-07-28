from enum import Enum


class TurnPhase(str, Enum):
    ACTIONS = "actions"
    DISPLACEMENT = "displacement"
    MOVE_PIECES = "move_pieces"
    BONUS_MARKER_CHOICE = "bonus_marker_choice"
    BUY_TILE_PAYMENT = "buy_tile_payment"
    REPLACE_BONUS_MARKERS = "replace_bonus_markers"
    TURN_COMPLETE = "turn_complete"
    GAME_OVER = "game_over"


class TurnStateError(RuntimeError):
    """Raised when a turn transition would violate the game state machine."""
