"""Tests for torch_batteries.utils.batch module."""

import torch

from torch_batteries.utils.batch import get_batch_size


class TestGetBatchSize:
    """Test cases for get_batch_size function."""

    def test_single_tensor_batch(self) -> None:
        """Test batch size from a single tensor."""
        batch = torch.randn(32, 10)
        assert get_batch_size(batch) == 32

    def test_empty_tensor_batch(self) -> None:
        """Test batch size from an empty tensor."""
        batch = torch.empty(0, 10)
        assert get_batch_size(batch) == 0

    def test_1d_tensor_batch(self) -> None:
        """Test batch size from a 1D tensor."""
        batch = torch.randn(5)
        assert get_batch_size(batch) == 5

    def test_list_batch(self) -> None:
        """Test batch size from a list of tensors."""
        batch = [torch.randn(16, 5), torch.randn(16, 3)]
        assert get_batch_size(batch) == 16

    def test_tuple_batch(self) -> None:
        """Test batch size from a tuple of tensors."""
        batch = (torch.randn(24, 8), torch.randn(24, 1))
        assert get_batch_size(batch) == 24

    def test_dict_batch(self) -> None:
        """Test batch size from a dictionary of tensors."""
        batch = {
            "input": torch.randn(12, 10),
            "target": torch.randn(12, 1),
        }
        assert get_batch_size(batch) == 12

    def test_nested_dict_batch(self) -> None:
        """Test batch size from a nested dictionary."""
        batch = {
            "data": {"x": torch.randn(8, 5), "y": torch.randn(8, 3)},
            "labels": torch.randn(8, 1),
        }
        assert get_batch_size(batch) == 8

    def test_empty_list_batch(self) -> None:
        """Test batch size from an empty list."""
        batch: list[torch.Tensor] = []
        assert get_batch_size(batch) == 1  # fallback

    def test_empty_dict_batch(self) -> None:
        """Test batch size from an empty dictionary."""
        batch: dict[str, torch.Tensor] = {}
        assert get_batch_size(batch) == 1  # fallback

    def test_non_tensor_batch(self) -> None:
        """Test batch size from non-tensor data."""
        batch = "not a tensor"
        assert get_batch_size(batch) == 1  # fallback

    def test_mixed_types_batch(self) -> None:
        """Test batch size from mixed types in list."""
        batch = [torch.randn(10, 5), "string", 123]
        assert get_batch_size(batch) == 10  # uses first tensor

    def test_scalar_tensor_batch(self) -> None:
        """Test batch size from a scalar tensor."""
        batch = torch.tensor(5.0)
        assert get_batch_size(batch) == 1  # scalar has no batch dimension
