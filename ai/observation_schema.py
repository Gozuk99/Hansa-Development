"""Version identity for the player-visible neural-network observation."""

OBSERVATION_SCHEMA_VERSION = 5
OBSERVATION_SIZE = 4724
# SHA-256 identity of the documented version-5 feature layout and visibility rules.
OBSERVATION_SCHEMA_FINGERPRINT = "9342c0713df5c05e3b5038879ad47ee709ec8935144203d620405e5d58914b1d"
LEGACY_OBSERVATION_SIZE_V4 = 4644
LEGACY_OBSERVATION_SCHEMA_V4_FINGERPRINT = (
    "468ca46dc53571a41741b424175d91562bc21bfcad0636a4c880b6600a3a6fb5"
)
LEGACY_OBSERVATION_SIZE_V3 = 4641
LEGACY_OBSERVATION_SCHEMA_V3_FINGERPRINT = (
    "7d57fea6f00406b2d558b4257cf4f36c2e6b396a465fb79ceeb591a174068d99"
)
LEGACY_OBSERVATION_SIZE = 4241
LEGACY_OBSERVATION_SCHEMA_V2_FINGERPRINT = (
    "031e5dd116422cb7f62fda548ecf7b8a73ab5ac2c9b4e01926f01beedc53193e"
)
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
    """Validate a model, accepting documented legacy input schemas for migration."""
    try:
        validate_observation_schema_metadata(metadata, artifact)
        return False
    except ObservationSchemaCompatibilityError:
        legacy_schema = (
            (
                metadata.get("observation_size") == LEGACY_OBSERVATION_SIZE_V4
                and metadata.get("observation_schema_version") == 4
                and metadata.get("observation_schema_fingerprint")
                == LEGACY_OBSERVATION_SCHEMA_V4_FINGERPRINT
            )
            or (
                metadata.get("observation_size") == LEGACY_OBSERVATION_SIZE_V3
                and metadata.get("observation_schema_version") == 3
                and metadata.get("observation_schema_fingerprint")
                == LEGACY_OBSERVATION_SCHEMA_V3_FINGERPRINT
            )
            or (
                metadata.get("observation_size") == LEGACY_OBSERVATION_SIZE
                and (
                    (
                        metadata.get("observation_schema_version") == 1
                        and metadata.get("observation_schema_fingerprint")
                        == LEGACY_OBSERVATION_SCHEMA_V1_FINGERPRINT
                    )
                    or (
                        metadata.get("observation_schema_version") == 2
                        and metadata.get("observation_schema_fingerprint")
                        == LEGACY_OBSERVATION_SCHEMA_V2_FINGERPRINT
                    )
                )
            )
        )
        if not legacy_schema:
            raise
    return True
