"""Tests for private ModelCheckpoint state validators."""

import pytest

from torch_batteries.callbacks._model_checkpoint_functions import (
    _optional_existing_path,
    _required_existing_path,
    _validate_save_weights_only,
)


def test_optional_checkpoint_path_accepts_none() -> None:
    assert _optional_existing_path(None) is None


def test_required_checkpoint_path_rejects_none() -> None:
    with pytest.raises(TypeError, match="must not be None"):
        _required_existing_path(None)


def test_checkpoint_output_mode_requires_boolean() -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        _validate_save_weights_only(1)
