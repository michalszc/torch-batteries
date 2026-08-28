"""Model Checkpoint Callback for torch-batteries."""

from __future__ import annotations


def _optional_string(value: object) -> str | None:
    """Validate an optional serialized path."""
    if value is None or isinstance(value, str):
        return value
    msg = "checkpoint path must be a string or None"
    raise TypeError(msg)


def _string_float_dict(value: object) -> dict[str, float]:
    """Validate serialized checkpoint ranking data."""
    if not isinstance(value, dict):
        msg = "best_k_models must be a dictionary"
        raise TypeError(msg)
    return {str(path): _serialized_float(score) for path, score in value.items()}


def _serialized_float(value: object) -> float:
    """Validate a serialized numeric value."""
    if not isinstance(value, (int, float)):
        msg = "checkpoint score must be numeric"
        raise TypeError(msg)
    return float(value)


def _validate_save_weights_only(value: object, *, expected: bool) -> None:
    """Validate the serialized checkpoint format configuration."""
    if value != expected:
        msg = "save_weights_only configuration does not match"
        raise ValueError(msg)
