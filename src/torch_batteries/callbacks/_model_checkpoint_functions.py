"""Model Checkpoint Callback for torch-batteries."""

from __future__ import annotations

from pathlib import Path


def _optional_existing_path(value: object) -> str | None:
    """Validate an optional serialized checkpoint file path."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = "checkpoint path must be a string or None"
        raise TypeError(msg)
    if not Path(value).is_file():
        msg = f"checkpoint path does not exist: {value}"
        raise ValueError(msg)
    return value


def _string_float_dict(value: object) -> dict[str, float]:
    """Validate serialized checkpoint ranking data."""
    if not isinstance(value, dict):
        msg = "best_k_models must be a dictionary"
        raise TypeError(msg)
    return {
        _required_existing_path(path): _serialized_float(score)
        for path, score in value.items()
    }


def _required_existing_path(value: object) -> str:
    """Validate one serialized checkpoint file path."""
    path = _optional_existing_path(value)
    if path is None:
        msg = "checkpoint path must not be None"
        raise TypeError(msg)
    return path


def _serialized_float(value: object) -> float:
    """Validate a serialized numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = "checkpoint score must be numeric"
        raise TypeError(msg)
    return float(value)


def _validate_save_weights_only(value: object, *, expected: bool | None = None) -> bool:
    """Validate checkpoint output mode from any configuration source."""
    if not isinstance(value, bool):
        msg = "save_weights_only must be a boolean"
        raise TypeError(msg)
    if expected is not None and value != expected:
        msg = "save_weights_only configuration does not match"
        raise ValueError(msg)
    return value
