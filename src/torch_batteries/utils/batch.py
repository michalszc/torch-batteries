"""Batch utility functions for torch-batteries package."""

from typing import Any

import torch


def get_batch_size(batch: Any) -> int:
    """Get the number of samples in a batch from a DataLoader.

    This function handles various batch formats commonly used in PyTorch:
    - Single tensor
    - List or tuple of tensors
    - Dictionary with tensor values
    - Other data types (returns 1 as fallback)

    Args:
        batch: Batch data from a DataLoader

    Returns:
        Number of samples in the batch

    Examples:
        ```python
        # Single tensor batch
        batch = torch.randn(32, 10)
        size = get_batch_size(batch)  # Returns 32

        # List/tuple batch (common for (x, y) pairs)
        batch = (torch.randn(32, 10), torch.randn(32, 1))
        size = get_batch_size(batch)  # Returns 32

        # Dictionary batch
        batch = {"input": torch.randn(32, 10), "target": torch.randn(32, 1)}
        size = get_batch_size(batch)  # Returns 32
        ```
    """
    if isinstance(batch, torch.Tensor):
        # Handle scalar tensors (0-dimensional)
        if batch.dim() == 0:
            return 1
        return batch.size(0)

    if isinstance(batch, (list, tuple)):
        tensor = next((x for x in batch if isinstance(x, torch.Tensor)), None)
        return tensor.size(0) if tensor is not None else 1

    if isinstance(batch, dict):
        tensor = next((v for v in batch.values() if isinstance(v, torch.Tensor)), None)
        return tensor.size(0) if tensor is not None else 1

    # Fallback for unknown data types
    return 1
