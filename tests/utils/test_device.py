"""Tests for torch_batteries.utils.device module."""

from unittest.mock import MagicMock, patch

import torch

from torch_batteries.utils.device import get_device, move_to_device


class TestGetDevice:
    """Test cases for get_device function."""

    def test_auto_device_cpu_fallback(self) -> None:
        """Test auto device detection falls back to CPU."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            device = get_device("auto")
            assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True, create=True)
    @patch("torch.cuda.is_built", return_value=True, create=True)
    def test_auto_device_cuda(
        self, mock_cuda_available: MagicMock, mock_cuda_built: MagicMock
    ) -> None:
        """Test auto device detection uses MPS when CUDA unavailable."""
        with (
            patch("torch.backends.mps.is_available", return_value=False, create=True),
            patch("torch.backends.mps.is_built", return_value=False, create=True),
        ):
            device = get_device("auto")
            assert device.type == "cuda"

    @patch("torch.cuda.is_available", return_value=False, create=True)
    @patch("torch.cuda.is_built", return_value=False, create=True)
    def test_auto_device_mps(
        self, mock_cuda_available: MagicMock, mock_cuda_built: MagicMock
    ) -> None:
        """Test auto device detection uses MPS when CUDA unavailable."""
        with (
            patch("torch.backends.mps.is_available", return_value=True, create=True),
            patch("torch.backends.mps.is_built", return_value=True, create=True),
        ):
            device = get_device("auto")
            assert device.type == "mps"

    def test_explicit_device_string(self) -> None:
        """Test explicit device string."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_explicit_device_object(self) -> None:
        """Test explicit device object."""
        input_device = torch.device("cpu")
        device = get_device(input_device)
        assert device == input_device

    def test_cuda_device_string(self) -> None:
        """Test CUDA device string."""
        device = get_device("cuda:0")
        assert device.type == "cuda"
        assert device.index == 0

    def test_default_auto(self) -> None:
        """Test default parameter is auto."""
        device = get_device()
        assert isinstance(device, torch.device)


class TestMoveToDevice:
    """Test cases for move_to_device function."""

    def test_move_tensor_to_device(self) -> None:
        """Test moving a single tensor to device."""
        tensor = torch.randn(3, 4)
        device = torch.device("cpu")
        result = move_to_device(tensor, device)
        assert result.device == device

    def test_move_list_to_device(self) -> None:
        """Test moving a list of tensors to device."""
        tensors = [torch.randn(2, 3), torch.randn(2, 1)]
        device = torch.device("cpu")
        result = move_to_device(tensors, device)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(t.device == device for t in result)

    def test_move_tuple_to_device(self) -> None:
        """Test moving a tuple of tensors to device."""
        tensors = (torch.randn(2, 3), torch.randn(2, 1))
        device = torch.device("cpu")
        result = move_to_device(tensors, device)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(t.device == device for t in result)

    def test_move_dict_to_device(self) -> None:
        """Test moving a dictionary of tensors to device."""
        tensors = {
            "input": torch.randn(2, 3),
            "target": torch.randn(2, 1),
        }
        device = torch.device("cpu")
        result = move_to_device(tensors, device)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(t.device == device for t in result.values())

    def test_move_nested_dict_to_device(self) -> None:
        """Test moving a nested dictionary to device."""
        tensors = {
            "data": {"x": torch.randn(2, 3), "y": torch.randn(2, 1)},
            "labels": torch.randn(2, 1),
        }
        device = torch.device("cpu")
        result = move_to_device(tensors, device)
        assert isinstance(result, dict)
        assert isinstance(result["data"], dict)
        assert result["data"]["x"].device == device
        assert result["data"]["y"].device == device
        assert result["labels"].device == device

    def test_move_mixed_types_to_device(self) -> None:
        """Test moving mixed types (tensors and non-tensors)."""
        data = [torch.randn(2, 3), "string", 123, torch.randn(2, 1)]
        device = torch.device("cpu")
        result = move_to_device(data, device)
        assert isinstance(result, list)
        assert len(result) == 4
        assert result[0].device == device  # tensor moved
        assert result[1] == "string"  # string unchanged
        assert result[2] == 123  # int unchanged
        assert result[3].device == device  # tensor moved

    def test_move_non_tensor_to_device(self) -> None:
        """Test moving non-tensor data returns unchanged."""
        data = "not a tensor"
        device = torch.device("cpu")
        result = move_to_device(data, device)
        assert result == data
