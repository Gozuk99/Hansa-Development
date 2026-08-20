"""Shared inference model for Hansa Teutonica."""

from pathlib import Path
import tempfile

import torch
import torch.nn as nn

from ai.observation_schema import (
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


class HansaNN(nn.Module):
    """Shared action-value network for every AI-controlled player."""

    def __init__(self, model_file=None):
        super().__init__()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(MODEL_INITIALIZATION_SEED)
            self.layer1 = nn.Linear(OBSERVATION_SIZE, 2048).to(device)
            self.layer2 = nn.Linear(2048, 1024).to(device)
            self.layer3 = nn.Linear(1024, ACTION_SPACE_SIZE).to(device)
        self.relu = nn.ReLU()
        self.migrated_observation_schema = False

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
        observation = self.relu(self.layer1(observation))
        observation = self.relu(self.layer2(observation))
        return self.layer3(observation)

    def load_model(self, model_file) -> None:
        checkpoint = torch.load(model_file, map_location=device)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(f"Model checkpoint {model_file} is missing its state_dict")
        validate_action_schema_metadata(checkpoint, f"Model checkpoint {model_file}")
        self.migrated_observation_schema = validate_model_observation_schema_metadata(
            checkpoint, f"Model checkpoint {model_file}"
        )
        self.load_state_dict(checkpoint["state_dict"])

    def save_model(self, model_file=SHARED_MODEL_FILE) -> Path:
        model_path = Path(model_file)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=model_path.parent, delete=False) as output:
                temporary = Path(output.name)
            torch.save(
                {
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
