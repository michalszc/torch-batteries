"""Weights & Biases (wandb) tracker implementation."""

import importlib
import tempfile
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from torch_batteries.tracking.base import ExperimentTracker
from torch_batteries.tracking.types import (
    Run,
)
from torch_batteries.utils.logging import get_logger

logger = get_logger("tracking.wandb")


class _WandbRun(Protocol):
    """Runtime shape of the W&B run methods used by this tracker."""

    id: Any
    url: Any
    summary: MutableMapping[str, Any]

    def log(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def finish(self, exit_code: int = 0) -> None: ...

    def log_artifact(self, artifact: Any, aliases: list[str]) -> None: ...


class _WandbArtifact(Protocol):
    """Runtime shape of the W&B artifact methods used by this tracker."""

    def add_file(self, local_path: str, name: str) -> None: ...


class _WandbModule(Protocol):
    """Runtime shape of the dynamically imported W&B module."""

    Artifact: Callable[..., _WandbArtifact]

    def init(self, **kwargs: Any) -> _WandbRun: ...


class WandbTracker(ExperimentTracker):
    """
    Weights & Biases experiment tracker implementation.

    Example:
    ```python
    tracker = WandbTracker(project="your-wandb-project")
    tracker.init(
        run=Run(config={"lr": 0.001})
    )

    # During training
    tracker.log_metrics({"train/loss": 0.5}, step=100)

    tracker.finish()
    ```
    """

    __slots__ = (
        "_entity",
        "_is_initialized",
        "_project",
        "_run",
        "_run_id",
        "_wandb",
    )

    def __init__(self, project: str, entity: str | None = None) -> None:
        """
        Initialize the wandb tracker.

        Args:
            project: Wandb project name
            entity: Optional wandb entity (username or team name)
        """
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as e:
            msg = (
                "wandb is not installed. "
                "Install it with `pip install torch-batteries[wandb]`."
            )
            raise ImportError(msg) from e

        self._wandb = cast("_WandbModule", wandb)
        self._project = project
        self._entity = entity
        self._run: _WandbRun | None = None
        self._run_id: str | None = None
        self._is_initialized = False

    @property
    def run(self) -> _WandbRun | None:
        """Get the tracked wandb run."""
        return self._run

    @property
    def entity(self) -> str | None:
        """Get the wandb entity."""
        return self._entity

    @property
    def project(self) -> str:
        """Get the wandb project."""
        return self._project

    def init(
        self,
        run: Run,
    ) -> None:
        """
        Initialize wandb tracking session.

        Args:
            run: Name, grouping, tags, description, job type, and configuration sent
                to W&B. Project and entity come from this tracker.

        Raises:
            RuntimeError: If it is already initialized
        """

        if self.is_initialized:
            msg = "WandbTracker is already initialized."
            raise RuntimeError(msg)

        wandb_config = {
            "project": self._project,
            "entity": self._entity,
            "group": run.group,
            "notes": run.description,
            "tags": run.tags,
            "job_type": run.job_type,
            "name": run.name,
            "config": run.config,
        }

        wandb_run = self._wandb.init(**wandb_config)

        self._run = wandb_run
        self._is_initialized = True

        logger.info("Initialized wandb run: run_id=%s", self.run_id)

    @property
    def is_initialized(self) -> bool:
        """
        Check if the tracker has been initialized.

        Returns:
            bool: True if initialized, False otherwise
        """
        return self._is_initialized

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
            prefix: Optional prefix for metric names

        Raises:
            RuntimeError: If the tracker is not initialized
        """
        wandb_run = self._require_run()

        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        logger.debug(
            "Logging wandb metrics: keys=%s, step=%s",
            sorted(metrics),
            step,
        )

        if step is not None:
            wandb_run.log(metrics, step=step)
        else:
            wandb_run.log(metrics)

    def finish(self, exit_code: int = 0) -> None:
        """
        Finish the wandb run.

        Args:
            exit_code: Exit code (0 for success, non-zero for failure)

        Raises:
            RuntimeError: If the tracker is not initialized
        """
        wandb_run = self._require_run()

        wandb_run.finish(exit_code=exit_code)
        self._is_initialized = False
        logger.info("wandb run finished")

    def log_summary(self, summary: dict[str, Any]) -> None:
        """
        Log summary statistics.

        Args:
            summary: Summary dictionary

        Raises:
            RuntimeError: If the tracker is not initialized
        """
        wandb_run = self._require_run()
        logger.debug("Logging wandb summary: keys=%s", sorted(summary))
        for key, value in summary.items():
            wandb_run.summary[key] = value

    def log_model(
        self,
        model: nn.Module,
        name: str = "model",
        *,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a model checkpoint as a W&B artifact.
        Args:
            model: Trained PyTorch model
            name: Name of the model artifact
            aliases: Optional list of aliases for the artifact
            metadata: Optional metadata dictionary for the artifact
        Raises:
            RuntimeError: If the tracker is not initialized
        """
        wandb_run = self._require_run()

        artifact_name = f"{name}-{self.run_id}" if self.run_id else name
        artifact_metadata: dict[str, Any] = {
            "torch_version": getattr(torch, "__version__", None),
            **(metadata or {}),
        }

        with tempfile.TemporaryDirectory(prefix="torch_batteries_") as tmpdir:
            model_path = Path(tmpdir) / f"{name}.pt"
            torch.save({"state_dict": model.state_dict()}, model_path)

            artifact = self._wandb.Artifact(
                name=artifact_name,
                type="model",
                metadata=artifact_metadata,
            )
            artifact.add_file(str(model_path), name=f"{name}.pt")

            wandb_run.log_artifact(artifact, aliases=aliases or ["latest"])

        logger.info("Logged wandb model artifact: name=%s", artifact_name)

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        if self._run:
            return str(self._run.id)
        return None

    @property
    def run_url(self) -> str | None:
        """Get the wandb run URL."""
        if self._run:
            return str(self._run.url)
        return None

    def _require_run(self) -> _WandbRun:
        if not self.is_initialized or self._run is None:
            msg = "WandbTracker is not initialized. Call init()."
            raise RuntimeError(msg)
        return self._run
