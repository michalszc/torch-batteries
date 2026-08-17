"""Event-driven dataset and DataLoader construction.

## Public API

- **`DataPack`** — base contract for charged data lifecycle methods.
- **`DataPackHandler`** — discovers and dispatches charged DataPack methods.
- **`DatasetBundle`** and **`DataLoaderBundle`** — resolved data containers.
- **`DataLoaderConfig`** — validated DataLoader construction options.
- **`DataContext`** and **`ResolvedData`** — workflow context and resolution result.
"""

from .base import DataPack
from .handler import DataPackHandler
from .types import (
    DataContext,
    DataLoaderBundle,
    DataLoaderConfig,
    DatasetBundle,
    ResolvedData,
)

__all__ = [
    "DataContext",
    "DataLoaderBundle",
    "DataLoaderConfig",
    "DataPack",
    "DataPackHandler",
    "DatasetBundle",
    "ResolvedData",
]
