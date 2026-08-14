"""Version identity for the player-visible neural-network observation."""

OBSERVATION_SCHEMA_VERSION = 2
OBSERVATION_SIZE = 4241
# SHA-256 identity of the documented version-2 feature layout and visibility rules.
OBSERVATION_SCHEMA_FINGERPRINT = "031e5dd116422cb7f62fda548ecf7b8a73ab5ac2c9b4e01926f01beedc53193e"
LEGACY_OBSERVATION_SCHEMA_V1_FINGERPRINT = (
    "cd787bc9c502e5254de4ecf9c682546d29f97d8a1dd38912887b3bd74fab8136"
)


class ObservationSchemaCompatibilityError(ValueError):
    """Raised when a checkpoint uses a different observation contract."""


def observation_schema_metadata() -> dict[str, int | str]:
    return {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "observation_schema_fingerprint": OBSERVATION_SCHEMA_FINGERPRINT,
    }


def validate_observation_schema_metadata(metadata, artifact="artifact") -> None:
    expected = observation_schema_metadata()
    missing = [key for key in expected if metadata.get(key) is None]
    if missing:
        raise ObservationSchemaCompatibilityError(
            f"{artifact} is missing observation-schema metadata: {', '.join(missing)}"
        )
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ObservationSchemaCompatibilityError(
            f"{artifact} uses an incompatible observation schema: {'; '.join(mismatches)}. "
            "Use an explicit migration or the matching runtime."
        )


def validate_model_observation_schema_metadata(metadata, artifact="model checkpoint") -> bool:
    """Validate a model, explicitly accepting the shape-compatible v1 transfer."""
    try:
        validate_observation_schema_metadata(metadata, artifact)
        return False
    except ObservationSchemaCompatibilityError:
        legacy_v1 = (
            metadata.get("observation_schema_version") == 1
            and metadata.get("observation_size") == OBSERVATION_SIZE
            and metadata.get("observation_schema_fingerprint")
            == LEGACY_OBSERVATION_SCHEMA_V1_FINGERPRINT
        )
        if not legacy_v1:
            raise
    return True
