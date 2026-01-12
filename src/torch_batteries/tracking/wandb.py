"""Weights & Biases (wandb) tracker implementation."""

from datetime import datetime
from pathlib import Path
from typing import Any

from torch import nn

from torch_batteries.tracking.base import ExperimentTracker
from torch_batteries.tracking.types import (
    Experiment,
    Project,
    Run,
)
from torch_batteries.utils.logging import get_logger

logger = get_logger("wandb_tracker")


class WandbTracker(ExperimentTracker):
    """
    Weights & Biases experiment tracker implementation.

    Requires wandb to be installed: `pip install wandb`

    Example:
        ```python
        tracker = WandbTracker(entity="your-wandb-username")
        tracker.init(
            project=Project("mnist-research"),
            experiment=Experiment("early-stopping-study"),
            run=Run(name="patience-5", config={"patience": 5})
        )

        # During training
        tracker.log_metrics({"train/loss": 0.5}, step=100)

        # Save model
        tracker.save_artifact(
            Artifact("best-model", ArtifactType.MODEL, "model.pth")
        )

        tracker.finish()
        ```
    """

    def __init__(self, entity: str | None = None) -> None:
        """
        Initialize the wandb tracker.

        Args:
            entity: Optional wandb entity (username or team name)
        """
        self._entity = entity
        self._wandb: Any = None
        self._run: Any = None
        self._run_id: str | None = None
        self._is_initialized = False

    def init(
        self,
        project: Project,
        experiment: Experiment | None = None,
        run: Run | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize wandb tracking session.

        Args:
            project: The project configuration
            experiment: Optional experiment grouping
            run: Run configuration
            **kwargs: Additional wandb.init() arguments

        Raises:
            ImportError: If wandb is not installed
        """
        try:
            import wandb
        except ImportError as e:
            msg = (
                "wandb is not installed. Install it with: pip install wandb\n"
                "Or use: pip install torch-batteries[wandb]"
            )
            raise ImportError(msg) from e

        self._wandb = wandb

        # Build wandb config
        wandb_config = {
            "project": project.name,
            "entity": self._entity,
            "notes": project.description,
        }

        # Add experiment info
        if experiment:
            wandb_config["group"] = experiment.name
            wandb_config["notes"] = experiment.description or wandb_config.get("notes")

        # Add run info
        if run:
            wandb_config["name"] = run.name
            if run.group:
                wandb_config["group"] = run.group
            if run.job_type:
                wandb_config["job_type"] = run.job_type
            if run.description:
                wandb_config["notes"] = run.description

            # Merge run config
            if "config" not in wandb_config:
                wandb_config["config"] = {}
            wandb_config["config"].update(run.config)

        # Allow overriding with kwargs
        wandb_config.update(kwargs)

        # Initialize wandb
        logger.info(f"Initializing wandb: project={project.name}")
        self._run = wandb.init(**wandb_config)
        self._run_id = self._run.id
        self._is_initialized = True

        logger.info(f"wandb run initialized: {self._run.url}")

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
        """
        if not self._is_initialized:
            logger.warning("Tracker not initialized. Call init() first.")
            return

        # Add prefix if provided
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Log to wandb
        if step is not None:
            self._wandb.log(metrics, step=step)
        else:
            self._wandb.log(metrics)

    def log_config(self, config: dict[str, Any]) -> None:
        """
        Log configuration to wandb.

        Args:
            config: Configuration dictionary
        """
        if not self._is_initialized:
            logger.warning("Tracker not initialized. Call init() first.")
            return

        self._wandb.config.update(config)

    def finish(self, exit_code: int = 0) -> None:
        """
        Finish the wandb run.

        Args:
            exit_code: Exit code (0 for success, non-zero for failure)
        """
        if not self._is_initialized:
            return

        self._run.finish(exit_code=exit_code)
        self._is_initialized = False
        logger.info("wandb run finished")

    def log_summary(self, summary: dict[str, Any]) -> None:
        """
        Log summary statistics.

        Args:
            summary: Summary dictionary
        """
        if not self._is_initialized:
            logger.warning("Tracker not initialized. Call init() first.")
            return

        for key, value in summary.items():
            self._run.summary[key] = value

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        return self._run_id

    @property
    def run_url(self) -> str | None:
        """Get the wandb run URL."""
        if self._run:
            return self._run.url
        return None
