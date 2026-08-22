"""Weights & Biases (wandb) tracker implementation."""

from typing import Protocol


class _WandbArtifact(Protocol):  # noqa: PYI046
    """Runtime shape of the W&B artifact methods used by this tracker."""

    def add_file(self, local_path: str, name: str) -> None: ...
