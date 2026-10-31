"""Atomic checkpoint file operations shared by weights and ranking metadata."""

from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def _temporary_path(destination: Path) -> Generator[Path]:
    """Create a closed temporary file beside a destination and always clean it up."""
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        path = Path(temporary.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def _checkpoint_file_transaction(destination: Path) -> Generator[None]:
    """Preserve a static destination until both checkpoint and manifest commit."""
    with _temporary_path(destination) as backup:
        existed = destination.is_file()
        if existed:
            shutil.copyfile(destination, backup)
        try:
            yield
        except BaseException:
            if existed:
                backup.replace(destination)
            else:
                destination.unlink(missing_ok=True)
            raise


def _save_weights(state: dict[str, Any], destination: Path) -> None:
    """Atomically write raw model weights without leaving partial checkpoint files."""
    with _temporary_path(destination) as temporary:
        torch.save(state, temporary)
        temporary.replace(destination)


def _write_json(payload: dict[str, object], destination: Path) -> None:
    """Atomically replace a ranking manifest after serialization succeeds."""
    with _temporary_path(destination) as temporary:
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)
