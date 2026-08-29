import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from ai.ai_model import (
    LEGACY_MODEL_CHECKPOINT_VERSION,
    MODEL_CHECKPOINT_FORMAT,
    MODEL_CHECKPOINT_VERSION,
    HansaNN,
)
from ai.observation_schema import (
    LEGACY_OBSERVATION_SCHEMA_V2_FINGERPRINT,
    LEGACY_OBSERVATION_SCHEMA_V3_FINGERPRINT,
    LEGACY_OBSERVATION_SCHEMA_V4_FINGERPRINT,
    LEGACY_OBSERVATION_SIZE,
    LEGACY_OBSERVATION_SIZE_V3,
    LEGACY_OBSERVATION_SIZE_V4,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    observation_schema_metadata,
)
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
        model = HansaNN()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            model.save_model(path)
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["observation_size"], OBSERVATION_SIZE)
            self.assertEqual(checkpoint["observation_schema_version"], OBSERVATION_SCHEMA_VERSION)
            self.assertEqual(checkpoint["action_space_size"], ACTION_SPACE_SIZE)
            HansaNN(model_file=path)

            legacy = torch.load(path, map_location="cpu")
            legacy["state_dict"]["layer1.weight"] = legacy["state_dict"]["layer1.weight"][
                :, :LEGACY_OBSERVATION_SIZE
            ]
            legacy["observation_schema_version"] = 2
            legacy["observation_size"] = LEGACY_OBSERVATION_SIZE
            legacy["observation_schema_fingerprint"] = LEGACY_OBSERVATION_SCHEMA_V2_FINGERPRINT
            torch.save(legacy, path)
            migrated = HansaNN(model_file=path)
            self.assertTrue(migrated.migrated_observation_schema)
            self.assertTrue(
                torch.equal(
                    migrated.layer1.weight[:, :LEGACY_OBSERVATION_SIZE].cpu(),
                    legacy["state_dict"]["layer1.weight"],
                )
            )
            self.assertFalse(migrated.layer1.weight[:, LEGACY_OBSERVATION_SIZE:].any())
            migrated.save_model(path)
            self.assertEqual(
                torch.load(path, map_location="cpu")["observation_schema_version"],
                OBSERVATION_SCHEMA_VERSION,
            )

            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "missing"):
                HansaNN(model_file=path)

            incompatible = {
                "state_dict": model.state_dict(),
                **action_schema_metadata(),
                **observation_schema_metadata(),
                "observation_size": OBSERVATION_SIZE + 1,
            }
            torch.save(incompatible, path)
            with self.assertRaisesRegex(ValueError, "observation schema"):
                HansaNN(model_file=path)

            incompatible["observation_size"] = OBSERVATION_SIZE
            incompatible["action_space_size"] = ACTION_SPACE_SIZE - 1
            torch.save(incompatible, path)
            with self.assertRaisesRegex(ValueError, "action schema"):
                HansaNN(model_file=path)

    def test_legacy_observation_migration_preserves_zero_extended_outputs(self):
        source = HansaNN()
        state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
        legacy_weight = state["layer1.weight"][:, :LEGACY_OBSERVATION_SIZE].clone()
        state["layer1.weight"] = legacy_weight
        legacy_observation = torch.randn((3, LEGACY_OBSERVATION_SIZE))
        with torch.no_grad():
            legacy_features = source.relu(
                torch.nn.functional.linear(
                    legacy_observation,
                    legacy_weight,
                    state["layer1.bias"],
                )
            )
            legacy_features = source.relu(
                torch.nn.functional.linear(
                    legacy_features,
                    state["layer2.weight"],
                    state["layer2.bias"],
                )
            )
            expected_q = torch.nn.functional.linear(
                legacy_features,
                state["layer3.weight"],
                state["layer3.bias"],
            )
            expected_policy = torch.nn.functional.linear(
                legacy_features,
                state["policy_head.weight"],
                state["policy_head.bias"],
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-observation.pth"
            torch.save(
                {
                    "model_checkpoint_format": MODEL_CHECKPOINT_FORMAT,
                    "model_checkpoint_version": MODEL_CHECKPOINT_VERSION,
                    "state_dict": state,
                    **action_schema_metadata(),
                    "observation_schema_version": 2,
                    "observation_size": LEGACY_OBSERVATION_SIZE,
                    "observation_schema_fingerprint": (LEGACY_OBSERVATION_SCHEMA_V2_FINGERPRINT),
                },
                path,
            )
            migrated = HansaNN(model_file=path)

        expanded = torch.zeros((3, OBSERVATION_SIZE))
        expanded[:, :LEGACY_OBSERVATION_SIZE] = legacy_observation
        with torch.no_grad():
            actual = migrated(expanded)
        self.assertTrue(
            torch.equal(
                migrated.layer1.weight.cpu()[:, :LEGACY_OBSERVATION_SIZE],
                legacy_weight,
            )
        )
        self.assertFalse(migrated.layer1.weight.cpu()[:, LEGACY_OBSERVATION_SIZE:].any())
        self.assertLessEqual(
            torch.max(torch.abs(actual.q_values.cpu() - expected_q)).item(),
            1e-6,
        )
        self.assertLessEqual(
            torch.max(torch.abs(actual.policy_logits.cpu() - expected_policy)).item(),
            1e-6,
        )

    def test_version_three_observation_migration_zero_extends_paid_action_history(self):
        source = HansaNN()
        state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
        legacy_weight = state["layer1.weight"][:, :LEGACY_OBSERVATION_SIZE_V3].clone()
        state["layer1.weight"] = legacy_weight
        legacy_observation = torch.randn((8, LEGACY_OBSERVATION_SIZE_V3))
        expanded_observation = torch.zeros((8, OBSERVATION_SIZE))
        expanded_observation[:, :LEGACY_OBSERVATION_SIZE_V3] = legacy_observation
        with torch.no_grad():
            expected = source(expanded_observation.to(source.layer1.weight.device))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version-three-observation.pth"
            torch.save(
                {
                    "model_checkpoint_format": MODEL_CHECKPOINT_FORMAT,
                    "model_checkpoint_version": MODEL_CHECKPOINT_VERSION,
                    "state_dict": state,
                    **action_schema_metadata(),
                    "observation_schema_version": 3,
                    "observation_size": LEGACY_OBSERVATION_SIZE_V3,
                    "observation_schema_fingerprint": (LEGACY_OBSERVATION_SCHEMA_V3_FINGERPRINT),
                },
                path,
            )
            migrated = HansaNN(model_file=path)

        with torch.no_grad():
            actual = migrated(expanded_observation.to(migrated.layer1.weight.device))
        self.assertTrue(
            torch.equal(
                migrated.layer1.weight.cpu()[:, :LEGACY_OBSERVATION_SIZE_V3],
                legacy_weight,
            )
        )
        self.assertFalse(migrated.layer1.weight.cpu()[:, LEGACY_OBSERVATION_SIZE_V3:].any())
        for key, expected_parameter in state.items():
            if key == "layer1.weight":
                continue
            self.assertTrue(torch.equal(migrated.state_dict()[key].cpu(), expected_parameter))
        self.assertLessEqual(
            torch.max(torch.abs(actual.q_values - expected.q_values)).item(),
            1e-6,
        )
        self.assertLessEqual(
            torch.max(torch.abs(actual.policy_logits - expected.policy_logits)).item(),
            1e-6,
        )

    def test_version_four_observation_migration_zero_extends_route_reward_history(self):
        source = HansaNN()
        state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
        legacy_weight = state["layer1.weight"][:, :LEGACY_OBSERVATION_SIZE_V4].clone()
        state["layer1.weight"] = legacy_weight
        legacy_observation = torch.randn((8, LEGACY_OBSERVATION_SIZE_V4))
        expanded_observation = torch.zeros((8, OBSERVATION_SIZE))
        expanded_observation[:, :LEGACY_OBSERVATION_SIZE_V4] = legacy_observation
        with torch.no_grad():
            expected = source(expanded_observation.to(source.layer1.weight.device))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version-four-observation.pth"
            torch.save(
                {
                    "model_checkpoint_format": MODEL_CHECKPOINT_FORMAT,
                    "model_checkpoint_version": MODEL_CHECKPOINT_VERSION,
                    "state_dict": state,
                    **action_schema_metadata(),
                    "observation_schema_version": 4,
                    "observation_size": LEGACY_OBSERVATION_SIZE_V4,
                    "observation_schema_fingerprint": (LEGACY_OBSERVATION_SCHEMA_V4_FINGERPRINT),
                },
                path,
            )
            migrated = HansaNN(model_file=path)

        with torch.no_grad():
            actual = migrated(expanded_observation.to(migrated.layer1.weight.device))
        self.assertTrue(
            torch.equal(
                migrated.layer1.weight.cpu()[:, :LEGACY_OBSERVATION_SIZE_V4],
                legacy_weight,
            )
        )
        self.assertFalse(migrated.layer1.weight.cpu()[:, LEGACY_OBSERVATION_SIZE_V4:].any())
        for key, expected_parameter in state.items():
            if key != "layer1.weight":
                self.assertTrue(torch.equal(migrated.state_dict()[key].cpu(), expected_parameter))
        self.assertLessEqual(
            torch.max(torch.abs(actual.q_values - expected.q_values)).item(),
            1e-6,
        )
        self.assertLessEqual(
            torch.max(torch.abs(actual.policy_logits - expected.policy_logits)).item(),
            1e-6,
        )

    def test_legacy_architecture_migrates_through_exact_identity_layer(self):
        source = HansaNN()
        legacy_state = {
            key: value.detach().cpu().clone()
            for key, value in source.state_dict().items()
            if not key.startswith("shared_layer3.")
        }
        observations = torch.randn((16, OBSERVATION_SIZE), device=source.layer1.weight.device)
        with torch.no_grad():
            features = torch.relu(source.layer1(observations))
            features = torch.relu(source.layer2(features))
            expected_q = source.layer3(features)
            expected_policy = source.policy_head(features)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-architecture.pth"
            torch.save(
                {
                    "model_checkpoint_format": MODEL_CHECKPOINT_FORMAT,
                    "model_checkpoint_version": LEGACY_MODEL_CHECKPOINT_VERSION,
                    "state_dict": legacy_state,
                    **action_schema_metadata(),
                    **observation_schema_metadata(),
                },
                path,
            )
            migrated = HansaNN(model_file=path)

        self.assertTrue(migrated.migrated_shared_layer)
        self.assertTrue(
            torch.equal(
                migrated.shared_layer3.weight.cpu(),
                torch.eye(1024),
            )
        )
        self.assertFalse(migrated.shared_layer3.bias.any())
        for key, expected in legacy_state.items():
            self.assertTrue(torch.equal(migrated.state_dict()[key].cpu(), expected), key)
        with torch.no_grad():
            actual = migrated(observations)
        self.assertLessEqual(torch.max(torch.abs(actual.q_values - expected_q)).item(), 1e-6)
        self.assertLessEqual(
            torch.max(torch.abs(actual.policy_logits - expected_policy)).item(),
            1e-6,
        )

        old_parameter_count = sum(value.numel() for value in legacy_state.values())
        new_parameter_count = sum(parameter.numel() for parameter in migrated.parameters())
        self.assertEqual(new_parameter_count - old_parameter_count, 1_049_600)

    def test_model_accepts_observation_and_is_deterministic(self):
        observation = torch.zeros((1, OBSERVATION_SIZE))
        first_model = HansaNN()
        second_model = HansaNN()
        first = first_model(observation)
        second = second_model(observation)

        self.assertEqual(tuple(first.q_values.shape), (1, ACTION_SPACE_SIZE))
        self.assertEqual(tuple(first.policy_logits.shape), (1, ACTION_SPACE_SIZE))
        self.assertTrue(torch.equal(first.q_values, second.q_values))
        self.assertTrue(torch.equal(first.policy_logits, second.policy_logits))
        self.assertFalse(hasattr(first_model, "optimizer"))
        self.assertFalse(first_model.training)

    def test_q_only_model_checkpoint_migrates_only_the_policy_head(self):
        model = HansaNN()
        original = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if not key.startswith("policy_head.")
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q-only.pth"
            torch.save(
                {
                    "state_dict": original,
                    **action_schema_metadata(),
                    **observation_schema_metadata(),
                },
                path,
            )
            migrated = HansaNN(model_file=path)

        self.assertTrue(migrated.migrated_policy_head)
        for key, expected in original.items():
            self.assertTrue(torch.equal(migrated.state_dict()[key], expected), key)

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
