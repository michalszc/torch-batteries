"""Model Checkpoint Callback for torch-batteries."""

from __future__ import annotations


def _optional_path(value: object) -> str | None:
    """Validate a serialized path independently of its current availability."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = "checkpoint path must be a string or None"
        raise TypeError(msg)
    return value


def _string_float_dict(value: object) -> dict[str, float]:
    """Validate serialized checkpoint ranking data."""
    if not isinstance(value, dict):
        msg = "best_k_models must be a dictionary"
        raise TypeError(msg)
    return {
        _required_path(path): _serialized_float(score) for path, score in value.items()
    }


def _required_path(value: object) -> str:
    """Validate one non-null serialized checkpoint path."""
    path = _optional_path(value)
    if path is None:
        msg = "checkpoint path must not be None"
        raise TypeError(msg)
    return path


def _serialized_float(value: object, *, empty_score: float | None = None) -> float:
    """Validate a serialized numeric value."""
    if value is None and empty_score is not None:
        return empty_score
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = "checkpoint score must be numeric"
        raise TypeError(msg)
    return float(value)


def _validate_save_weights_only(value: object, *, expected: bool | None = None) -> bool:
    """Validate checkpoint output mode from any configuration source."""
    value = _validate_bool(value, name="save_weights_only")
    if expected is not None and value != expected:
        msg = "save_weights_only configuration does not match"
        raise ValueError(msg)
    return value


def _validate_bool(value: object, *, name: str) -> bool:
    """Require an explicit boolean checkpoint option."""
    if not isinstance(value, bool):
        msg = f"{name} must be a boolean"
        raise TypeError(msg)
    return value
