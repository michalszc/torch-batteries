"""Weights & Biases (wandb) tracker implementation."""

from collections.abc import Callable
from typing import Any, Protocol

from ._artifact import _WandbArtifact
from ._run import _WandbRun


class _WandbModule(Protocol):  # noqa: PYI046
    """Runtime shape of the dynamically imported W&B module."""

    Artifact: Callable[..., _WandbArtifact]

    def init(self, **kwargs: Any) -> _WandbRun: ...
