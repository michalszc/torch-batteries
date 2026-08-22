"""Weights & Biases (wandb) tracker implementation."""

from collections.abc import MutableMapping
from typing import Any, Protocol


class _WandbRun(Protocol):  # noqa: PYI046
    """Runtime shape of the W&B run methods used by this tracker."""

    id: Any
    url: Any
    summary: MutableMapping[str, Any]

    def log(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def finish(self, exit_code: int = 0) -> None: ...

    def log_artifact(self, artifact: Any, aliases: list[str]) -> None: ...
