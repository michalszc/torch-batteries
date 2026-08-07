"""Supporting utilities used by the public training workflows.

## Modules

- **`batch`** — batch-size inference for tensors and nested containers.
- **`device`** — automatic device selection and recursive data movement.
- **`formatting`** — human-readable metric formatting.
- **`logging`** — package logger configuration.
- **`metrics`** — callable, stateful, and collected metric support.
- **`prediction`** — recursive prediction concatenation.
- **`progress`** — silent, bar, and text progress implementations.
"""

from . import batch, device, formatting, logging, metrics, progress

__all__ = [
    "batch",
    "device",
    "formatting",
    "logging",
    "metrics",
    "progress",
]
