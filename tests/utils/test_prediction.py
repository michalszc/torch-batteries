"""Tests for structured prediction utilities."""

from typing import NamedTuple

import pytest
import torch

from torch_batteries.utils.prediction import concatenate_predictions


class _PredictionPair(NamedTuple):
    inputs: torch.Tensor
    shifted: torch.Tensor


def test_concatenation_rejects_incompatible_structures() -> None:
    with pytest.raises(ValueError, match="dictionary structures differ"):
        concatenate_predictions([{"scores": torch.ones(1)}, {"other": torch.ones(1)}])
    with pytest.raises(TypeError, match="supports tensors"):
        concatenate_predictions([1, 2])


def test_concatenation_preserves_named_tuple_type() -> None:
    result = concatenate_predictions(
        [
            _PredictionPair(torch.ones(1, 2), torch.zeros(1, 2)),
            _PredictionPair(torch.ones(2, 2), torch.zeros(2, 2)),
        ]
    )

    assert isinstance(result, _PredictionPair)
    assert result.inputs.shape == (3, 2)
    assert result.shifted.shape == (3, 2)
