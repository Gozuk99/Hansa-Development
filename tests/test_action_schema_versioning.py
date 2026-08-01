import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from ai.ai_model import HansaNN
from game.action_codec import DEFAULT_ACTION_CODEC
from game.game_runner import ReplayRecord, load_replay, save_replay
from game.action_schema import (
    ACTION_SCHEMA_FINGERPRINT,
    ACTION_SCHEMA_VERSION,
    ACTION_SPACE_SIZE,
    ActionSchemaCompatibilityError,
    action_schema_metadata,
    validate_action_schema_metadata,
)


class TestActionSchemaVersioning(unittest.TestCase):
    def test_fingerprint_locks_every_assigned_index_and_semantic_action(self):
        assigned = (
            f"{index}:{DEFAULT_ACTION_CODEC.decode(index)!r}"
            for index in range(ACTION_SPACE_SIZE)
            if not DEFAULT_ACTION_CODEC.is_reserved(index)
        )
        fingerprint = hashlib.sha256("\n".join(assigned).encode()).hexdigest()
        self.assertEqual(fingerprint, ACTION_SCHEMA_FINGERPRINT)

    def test_current_metadata_is_compatible(self):
        validate_action_schema_metadata(action_schema_metadata(), "test artifact")

    def test_missing_old_future_size_and_semantic_mismatches_are_rejected(self):
        cases = (
            {},
            {**action_schema_metadata(), "action_schema_version": ACTION_SCHEMA_VERSION - 1},
            {**action_schema_metadata(), "action_schema_version": ACTION_SCHEMA_VERSION + 1},
            {**action_schema_metadata(), "action_space_size": 620},
            {**action_schema_metadata(), "action_schema_fingerprint": "remapped"},
        )
        for metadata in cases:
            with self.subTest(metadata=metadata), self.assertRaises(ActionSchemaCompatibilityError):
                validate_action_schema_metadata(metadata, "test artifact")

    def test_model_checkpoint_requires_exact_schema_metadata(self):
        model = HansaNN(2, ACTION_SPACE_SIZE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save({"state_dict": model.state_dict(), **action_schema_metadata()}, path)
            HansaNN(2, ACTION_SPACE_SIZE, model_file=path)

            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "missing"):
                HansaNN(2, ACTION_SPACE_SIZE, model_file=path)

    def test_replay_round_trip_requires_exact_schema_metadata(self):
        record = ReplayRecord(2, 3, 124, (1, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            save_replay(record, path)
            self.assertEqual(load_replay(path), record)

            data = record.to_dict()
            data["action_schema_version"] -= 1
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ActionSchemaCompatibilityError):
                load_replay(path)


if __name__ == "__main__":
    unittest.main()
