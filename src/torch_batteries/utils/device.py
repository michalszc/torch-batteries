"""Device utility functions for torch-batteries package."""

from typing import Any

import torch


def get_device(device: str | torch.device = "auto") -> torch.device:
    """
    Get the available device (CUDA, MPS or CPU).

    Returns:
        torch.device: The available device.
    """

    def _get_auto_device() -> torch.device:
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

    Example:
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
