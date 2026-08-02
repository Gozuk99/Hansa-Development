"""
CLI helper to validate action schema metadata in Hansa saves or model checkpoints.
Usage:
  python tools/check_schema_compat.py --game path/to/position.hansa
  python tools/check_schema_compat.py --model path/to/hansa_nn_model.pth

Exit code 0 on success, non-zero on incompatibility.
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception:
    torch = None

from game.action_schema import ActionSchemaCompatibilityError, validate_action_schema_metadata
from ai.observation_schema import (
    ObservationSchemaCompatibilityError,
    validate_observation_schema_metadata,
)


def check_game(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    try:
        validate_action_schema_metadata(metadata, f"Game file {path}")
    except ActionSchemaCompatibilityError as error:
        print(f"INCOMPATIBLE: {error}")
        return 3
    print("OK: saved-game schema matches runtime")
    return 0


def check_model(path: str) -> int:
    if torch is None:
        print("PyTorch not available; cannot check model files")
        return 4
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        try:
            validate_action_schema_metadata(loaded, f"Model checkpoint {path}")
            validate_observation_schema_metadata(loaded, f"Model checkpoint {path}")
        except (ActionSchemaCompatibilityError, ObservationSchemaCompatibilityError) as error:
            print(f"INCOMPATIBLE: {error}")
            return 3
        print("OK: model checkpoint schema matches runtime")
        return 0
    print("INCOMPATIBLE: legacy model checkpoint has no schema metadata")
    return 3


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--game", help="Path to saved game (.hansa)")
    p.add_argument("--model", help="Path to model checkpoint (.pth)")
    args = p.parse_args()
    if not args.game and not args.model:
        p.print_help()
        sys.exit(1)
    rc = 0
    if args.game:
        if not os.path.isfile(args.game):
            print("Game file not found")
            sys.exit(1)
        rc = max(rc, check_game(args.game))
    if args.model:
        if not os.path.isfile(args.model):
            print("Model file not found")
            sys.exit(1)
        rc = max(rc, check_model(args.model))
    sys.exit(rc)
