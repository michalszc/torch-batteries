"""Device utility functions for torch-batteries package."""

from typing import Any

import torch


def get_device(device: str | torch.device = "auto") -> torch.device:
    """Get PyTorch device for model and data placement.

    Automatically detects and returns the best available device (CUDA > MPS > CPU)
    when set to 'auto', or returns the specified device.

    Args:
        device: Device specification. Can be:
            - 'auto': Automatically detect best available device
            - str: Device string like 'cuda', 'cpu', 'mps', 'cuda:0'
            - torch.device: PyTorch device object

    Returns:
        torch.device object ready for use

    Examples:
        ```python
        # Auto-detect best device
        device = get_device('auto')  # Returns cuda, mps, or cpu

        # Explicitly specify device
        device = get_device('cuda')
        device = get_device('cpu')

        # Pass device object
        device = get_device(torch.device('cuda:1'))
        ```
    """

    def _get_auto_device() -> torch.device:
        """Detect best available device.

        Returns:
            torch.device: CUDA if available, else MPS if available, else CPU.
        """
        if (
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_built")
            and torch.cuda.is_built()
        ):
            return torch.device("cuda")
        if (
            torch.backends.mps.is_available()
            and hasattr(torch.backends.mps, "is_built")
            and torch.backends.mps.is_built()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    match device:
        case "auto":
            return _get_auto_device()
        case str():
            return torch.device(device)
        case torch.device():
            return device


def move_to_device(data: Any, device: torch.device) -> Any:
    """Move data to device handling different data types.

    Args:
        data: Data to move to device (tensors, lists, tuples, etc.)
        device: Target device to move the data to

    Returns:
        Data moved to the specified device

    Examples:
        ```python
        device = torch.device('cuda')
        batch = (torch.randn(32, 10), torch.randn(32, 1))
        batch_on_device = move_to_device(batch, device)
        ```
    """
    if isinstance(data, (list, tuple)):
        moved_data = [move_to_device(x, device) for x in data]
        # Preserve the original type (list or tuple)
        return type(data)(moved_data)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data
