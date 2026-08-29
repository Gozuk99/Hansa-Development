"""Shared inference model for Hansa Teutonica."""

from pathlib import Path
import tempfile
from typing import NamedTuple
import warnings

import torch
import torch.nn as nn

from ai.observation_schema import (
    LEGACY_OBSERVATION_SIZE,
    LEGACY_OBSERVATION_SIZE_V3,
    LEGACY_OBSERVATION_SIZE_V4,
    OBSERVATION_SIZE,
    observation_schema_metadata,
    validate_model_observation_schema_metadata,
)
from game.action_schema import (
    ACTION_SPACE_SIZE,
    action_schema_metadata,
    validate_action_schema_metadata,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHARED_MODEL_FILE = "hansa_nn_model.pth"
MODEL_INITIALIZATION_SEED = 124
MODEL_CHECKPOINT_FORMAT = "hansa-dual-head-model"
MODEL_CHECKPOINT_VERSION = 3
LEGACY_MODEL_CHECKPOINT_VERSION = 2


class HansaNNOutput(NamedTuple):
    """The independently interpreted outputs of the shared representation."""

    q_values: torch.Tensor
    policy_logits: torch.Tensor


class HansaNN(nn.Module):
    """Shared trunk with Q-value and shadow policy heads for every AI seat."""

    def __init__(self, model_file=None):
        super().__init__()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(MODEL_INITIALIZATION_SEED)
            self.layer1 = nn.Linear(OBSERVATION_SIZE, 2048).to(device)
            self.layer2 = nn.Linear(2048, 1024).to(device)
            self.layer3 = nn.Linear(1024, ACTION_SPACE_SIZE).to(device)
            self.policy_head = nn.Linear(1024, ACTION_SPACE_SIZE).to(device)
            self.shared_layer3 = nn.Linear(1024, 1024).to(device)
            nn.init.eye_(self.shared_layer3.weight)
            nn.init.zeros_(self.shared_layer3.bias)
        self.relu = nn.ReLU()
        self.migrated_observation_schema = False
        self.migrated_policy_head = False
        self.migrated_shared_layer = False

        if model_file and Path(model_file).is_file():
            self.load_model(model_file)
        self.eval()

    def forward(self, observation):
        if observation.shape[-1] != OBSERVATION_SIZE:
            raise ValueError(
                f"HansaNN expected {OBSERVATION_SIZE} observation values, "
                f"received {observation.shape[-1]}"
            )
        observation = observation.to(device)
        features = self.relu(self.layer1(observation))
        features = self.relu(self.layer2(features))
        features = self.relu(self.shared_layer3(features))
        return HansaNNOutput(
            q_values=self.layer3(features),
            policy_logits=self.policy_head(features.detach()),
        )

    def forward_q(self, observation):
        """Run the shared representation and Q head without the shadow policy head."""
        if observation.shape[-1] != OBSERVATION_SIZE:
            raise ValueError(
                f"HansaNN expected {OBSERVATION_SIZE} observation values, "
                f"received {observation.shape[-1]}"
            )
        observation = observation.to(device)
        features = self.relu(self.layer1(observation))
        features = self.relu(self.layer2(features))
        features = self.relu(self.shared_layer3(features))
        return self.layer3(features)

    def _load_checkpoint_state(self, checkpoint, artifact):
        state_dict = checkpoint["state_dict"]
        legacy_layer1 = state_dict.get("layer1.weight")
        if legacy_layer1 is not None and legacy_layer1.shape[1] != checkpoint.get(
            "observation_size"
        ):
            raise ValueError(
                f"{artifact} observation metadata does not match its input-layer width"
            )
        if (
            legacy_layer1 is not None
            and legacy_layer1.shape[0] == self.layer1.out_features
            and (
                legacy_layer1.shape[1]
                in (
                    LEGACY_OBSERVATION_SIZE,
                    LEGACY_OBSERVATION_SIZE_V3,
                    LEGACY_OBSERVATION_SIZE_V4,
                )
            )
        ):
            migrated = dict(state_dict)
            expanded = torch.zeros_like(self.layer1.weight)
            expanded[:, : legacy_layer1.shape[1]].copy_(legacy_layer1)
            migrated["layer1.weight"] = expanded
            state_dict = migrated
            self.migrated_observation_schema = True
        shared_keys = {"shared_layer3.weight", "shared_layer3.bias"}
        missing_shared = shared_keys.difference(state_dict)
        if missing_shared and missing_shared != shared_keys:
            raise ValueError(f"{artifact} contains an incomplete shared layer")
        if missing_shared:
            migrated = dict(state_dict)
            current = self.state_dict()
            for key in shared_keys:
                migrated[key] = current[key]
            state_dict = migrated
            self.migrated_shared_layer = True
        policy_keys = {"policy_head.weight", "policy_head.bias"}
        missing_policy = policy_keys.difference(state_dict)
        if missing_policy and missing_policy != policy_keys:
            raise ValueError(f"{artifact} contains an incomplete policy head")
        if missing_policy:
            migrated = dict(state_dict)
            current = self.state_dict()
            for key in policy_keys:
                migrated[key] = torch.zeros_like(current[key])
            state_dict = migrated
            self.migrated_policy_head = True
            warnings.warn(
                f"{artifact} was migrated from Q-only schema; initialized a new policy head",
                RuntimeWarning,
                stacklevel=2,
            )
        self.load_state_dict(state_dict)

    def load_model(self, model_file) -> None:
        checkpoint = torch.load(model_file, map_location=device)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(f"Model checkpoint {model_file} is missing its state_dict")
        validate_action_schema_metadata(checkpoint, f"Model checkpoint {model_file}")
        self.migrated_observation_schema = validate_model_observation_schema_metadata(
            checkpoint, f"Model checkpoint {model_file}"
        )
        checkpoint_version = checkpoint.get("model_checkpoint_version")
        if checkpoint_version not in (
            None,
            LEGACY_MODEL_CHECKPOINT_VERSION,
            MODEL_CHECKPOINT_VERSION,
        ):
            raise ValueError(
                f"Model checkpoint {model_file} uses incompatible model schema "
                f"version {checkpoint_version}"
            )
        if (
            checkpoint_version in (LEGACY_MODEL_CHECKPOINT_VERSION, MODEL_CHECKPOINT_VERSION)
            and checkpoint.get("model_checkpoint_format") != MODEL_CHECKPOINT_FORMAT
        ):
            raise ValueError(f"Model checkpoint {model_file} has an incompatible model format")
        self._load_checkpoint_state(checkpoint, f"Model checkpoint {model_file}")

    def save_model(self, model_file=SHARED_MODEL_FILE) -> Path:
        model_path = Path(model_file)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=model_path.parent, delete=False) as output:
                temporary = Path(output.name)
            torch.save(
                {
                    "model_checkpoint_format": MODEL_CHECKPOINT_FORMAT,
                    "model_checkpoint_version": MODEL_CHECKPOINT_VERSION,
                    "state_dict": self.state_dict(),
                    **action_schema_metadata(),
                    **observation_schema_metadata(),
                },
                temporary,
            )
            temporary.replace(model_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return model_path
