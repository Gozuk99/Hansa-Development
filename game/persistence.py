"""Versioned, exact snapshots for trusted local Hansa save files."""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile

from game.action_schema import action_schema_metadata, validate_action_schema_metadata
from game.game_info import Game
from game.invariants import validate_game
from game.loaded_state_validation import validate_loaded_game


SAVE_FORMAT = "hansa-exact-game"
SAVE_FORMAT_VERSION = 1
SAVE_EXTENSION = ".hansa"


class SaveGameError(ValueError):
    """Raised when a saved game is invalid or incompatible."""


_SAFE_GLOBALS = {
    ("builtins", "bytearray"),
    ("builtins", "complex"),
    ("builtins", "frozenset"),
    ("builtins", "set"),
    ("builtins", "slice"),
    ("collections", "deque"),
    ("random", "Random"),
    ("pygame.rect", "Rect"),
}
_PROJECT_MODULE_PREFIXES = ("game.", "map_data.", "player_info.")


class _GameSavePickler(pickle.Pickler):
    def persistent_id(self, obj):
        object_type = type(obj)
        if object_type.__module__ == "ai.ai_model" and object_type.__name__ == "HansaNN":
            return ("external_ai_model",)
        return None

    def reducer_override(self, obj):
        if isinstance(obj, Enum):
            return type(obj), (obj.value,)
        return NotImplemented


class _GameSaveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE_GLOBALS or module.startswith(_PROJECT_MODULE_PREFIXES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Save file references forbidden type {module}.{name}")

    def persistent_load(self, persistent_id):
        if persistent_id == ("external_ai_model",):
            return None
        raise pickle.UnpicklingError("Save file contains an unknown external reference")


def _serialize_snapshot(game, controller_rng_state) -> bytes:
    output = io.BytesIO()
    _GameSavePickler(output, protocol=pickle.HIGHEST_PROTOCOL).dump(
        {
            "game": game,
            "controller_rng_state": controller_rng_state,
        }
    )
    return output.getvalue()


def _deserialize_snapshot(payload: bytes):
    return _GameSaveUnpickler(io.BytesIO(payload)).load()


def _restore_office_printed_privileges(game: Game) -> None:
    """Backfill immutable printed office data absent from older exact saves."""
    if all(
        hasattr(office, "printed_privilege")
        for city in game.selected_map.cities
        for office in city.offices
    ):
        return

    template = Game(game.map_num, len(game.players), seed=game.seed).selected_map
    template_cities = {city.name: city for city in template.cities}
    for city in game.selected_map.cities:
        printed = iter(template_cities[city.name].offices)
        for office in city.offices:
            if office.place_adjacent_office:
                office.printed_privilege = None
            else:
                template_office = next(printed, None)
                office.printed_privilege = (
                    template_office.printed_privilege if template_office else "WHITE"
                )


def _remove_legacy_player_models(game: Game) -> None:
    for player in game.players:
        if hasattr(player, "hansa_nn"):
            del player.hansa_nn


def default_save_directory() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "Hansa Teutonica" / "saves"
    return Path.home() / ".local" / "share" / "hansa-teutonica" / "saves"


def suggested_save_name(game: Game) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"hansa-map{game.map_num}-turn{game.turn_number}-{timestamp}{SAVE_EXTENSION}"


def _metadata(game: Game, payload_hash: str) -> dict[str, object]:
    return {
        "save_format": SAVE_FORMAT,
        "save_format_version": SAVE_FORMAT_VERSION,
        **action_schema_metadata(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "map_num": game.map_num,
        "player_count": len(game.players),
        "turn_number": game.turn_number,
        "round_number": game.round_number,
        "current_player_index": game.current_player_index,
        "turn_phase": game.turn_phase.value,
        "game_end": game.game_end,
        "payload_sha256": payload_hash,
    }


def save_game(
    game: Game,
    filename: str | Path,
    *,
    controller_rng_state=None,
) -> Path:
    """Atomically save the complete engine object to a trusted local file."""
    try:
        validate_loaded_game(game)
    except Exception as error:
        raise SaveGameError(f"Cannot save invalid game state: {error}") from error
    target = Path(filename)
    if target.suffix.lower() != SAVE_EXTENSION:
        target = target.with_suffix(SAVE_EXTENSION)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload = _serialize_snapshot(game, controller_rng_state)
    except (pickle.PickleError, TypeError, AttributeError) as error:
        raise SaveGameError(f"Could not serialize game state: {error}") from error
    payload_hash = hashlib.sha256(payload).hexdigest()
    document = {
        "metadata": _metadata(game, payload_hash),
        "payload": base64.b64encode(payload).decode("ascii"),
    }

    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = Path(temporary.name)
            json.dump(document, temporary, indent=2)
            temporary.flush()
        temporary_name.replace(target)
    finally:
        if temporary_name is not None and temporary_name.exists():
            temporary_name.unlink()
    return target


def load_game(filename: str | Path) -> Game:
    """Load an exact game snapshot created by :func:`save_game`.

    Save files contain Python object data and must only be opened when they
    were created locally or came from a trusted source.
    """
    source = Path(filename)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SaveGameError(f"Could not read saved game: {error}") from error
    if not isinstance(document, dict):
        raise SaveGameError("Saved game must contain a JSON object")
    metadata = document.get("metadata")
    encoded_payload = document.get("payload")
    if not isinstance(metadata, dict):
        raise SaveGameError("Saved game metadata must be a JSON object")
    if not isinstance(encoded_payload, str):
        raise SaveGameError("Saved game payload must be text")

    if metadata.get("save_format") != SAVE_FORMAT:
        raise SaveGameError("This is not a Hansa exact-game save file")
    if metadata.get("save_format_version") != SAVE_FORMAT_VERSION:
        raise SaveGameError(
            "Saved game uses an incompatible save format: "
            f"{metadata.get('save_format_version')!r}; expected {SAVE_FORMAT_VERSION}"
        )
    try:
        validate_action_schema_metadata(metadata, "Saved game")
    except ValueError as error:
        raise SaveGameError(str(error)) from error

    try:
        payload = base64.b64decode(encoded_payload, validate=True)
    except (ValueError, TypeError) as error:
        raise SaveGameError("Saved game payload is not valid") from error
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != metadata.get("payload_sha256"):
        raise SaveGameError("Saved game is damaged or has been modified")

    try:
        restored = _deserialize_snapshot(payload)
    except Exception as error:
        raise SaveGameError(f"Could not restore saved game: {error}") from error
    if not isinstance(restored, dict) or not isinstance(restored.get("game"), Game):
        raise SaveGameError("Saved payload does not contain a Hansa game")
    game = restored["game"]
    _remove_legacy_player_models(game)
    _restore_office_printed_privileges(game)
    game._saved_controller_rng_state = restored.get("controller_rng_state")
    try:
        validate_loaded_game(game)
    except Exception as error:
        raise SaveGameError(f"Saved game contains invalid engine state: {error}") from error

    expected = {
        "map_num": game.map_num,
        "player_count": len(game.players),
        "turn_number": game.turn_number,
        "round_number": game.round_number,
        "current_player_index": game.current_player_index,
        "turn_phase": game.turn_phase.value,
        "game_end": game.game_end,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise SaveGameError(
            "Saved-game summary does not match its engine state: " + ", ".join(mismatches)
        )
    configuration = getattr(game, "configuration", None)
    if configuration is not None:
        try:
            game.ai_model = configuration._load_ai_model() if configuration.has_ai_players else None
        except Exception as error:
            raise SaveGameError(f"Could not restore AI model: {error}") from error
    return game
