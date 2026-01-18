"""Weights & Biases (wandb) tracker implementation."""

from typing import Any

import wandb

from torch_batteries.tracking.base import ExperimentTracker
from torch_batteries.tracking.types import (
    Run,
)
from torch_batteries.utils.logging import get_logger

logger = get_logger("wandb_tracker")


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
    )

    def __init__(self, project: str, entity: str | None = None) -> None:
        """
        Initialize the wandb tracker.

        Args:
            project: Wandb project name
            entity: Optional wandb entity (username or team name)
        """
        self._project = project
        self._entity = entity
        self._run: Any = None
        self._run_id: str | None = None
        self._is_initialized = False

    @property
    def run(self) -> wandb.sdk.wandb_run.Run | None:
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
            project: The project configuration
            experiment: Optional experiment grouping
            run: Run configuration

        Raises:
            ImportError: If wandb is not installed
            RuntimeError: If it is already initialized
        """
        try:
            import wandb  # noqa: PLC0415
        except ImportError as e:
            msg = "wandb is not installed."
            raise ImportError(msg) from e

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

        self._run = wandb.init(**wandb_config)
        self._is_initialized = True

        logger.info(
            "Initialized wandb: project=%s, entity=%s, run_id=%s",
            wandb_config["project"],
            wandb_config["entity"],
            self.run_id,
        )

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
        self._assert_initialized()

        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        if step is not None:
            self._run.log(metrics, step=step)
        else:
            self._run.log(metrics)

    def finish(self, exit_code: int = 0) -> None:
        """
        Finish the wandb run.

        Args:
            exit_code: Exit code (0 for success, non-zero for failure)

        Raises:
            RuntimeError: If the tracker is not initialized
        """
        self._assert_initialized()

        self._run.finish(exit_code=exit_code)
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
        self._assert_initialized()
        for key, value in summary.items():
            self._run.summary[key] = value

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

    def _assert_initialized(self) -> None:
        if not self.is_initialized:
            msg = "WandbTracker is not initialized. Call init()."
            raise RuntimeError(msg)
