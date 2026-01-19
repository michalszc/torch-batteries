"""TensorBoard tracker implementation."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torch_batteries.tracking.base import ExperimentTracker
from torch_batteries.tracking.types import Run
from torch_batteries.utils.logging import get_logger

logger = get_logger("tensorboard_tracker")


class TensorBoardTracker(ExperimentTracker):
    """
    TensorBoard experiment tracker implementation.

    Example:
    ```python
    tracker = TensorBoardTracker(output_dir="runs/experiment_1")
    tracker.init(
        run=Run(config={"lr": 0.001})
    )

    # During training
    tracker.log_metrics({"train/loss": 0.5}, step=100)

    tracker.finish()
    ```
    """

    __slots__ = (
        "_comment",
        "_is_initialized",
        "_output_dir",
        "_run_config",
        "_writer",
    )

    def __init__(
        self,
        output_dir: str | Path | None = None,
        comment: str = "",
    ) -> None:
        """
        Initialize the TensorBoard tracker.

        Args:
            output_dir: Directory to save TensorBoard logs. If None, uses default
                TensorBoard directory (runs/CURRENT_DATETIME_HOSTNAME).
            comment: Comment to append to the default output_dir. Ignored if
                output_dir is specified.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
        except ImportError as e:
            msg = "tensorboard is not installed. Install with: pip install tensorboard"
            raise ImportError(msg) from e

        self._output_dir = Path(output_dir) if output_dir else None
        self._comment = comment
        self._writer: SummaryWriter | None = None
        self._is_initialized = False
        self._run_config: dict[str, Any] = {}

    @property
    def output_dir(self) -> Path | None:
        """Get the TensorBoard output directory."""
        if self._writer:
            return Path(self._writer.log_dir)
        return self._output_dir

    @property
    def writer(self) -> SummaryWriter | None:
        """Get the TensorBoard SummaryWriter."""
        return self._writer

    def init(
        self,
        run: Run,
    ) -> None:
        """
        Initialize TensorBoard tracking session.

        Args:
            run: Run configuration

        Raises:
            RuntimeError: If it is already initialized
        """
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        if self.is_initialized:
            msg = "TensorBoardTracker is already initialized."
            raise RuntimeError(msg)

        # Build output_dir with run name if provided
        output_dir = self._output_dir
        if output_dir and run.name:
            output_dir = output_dir / run.name

        # Create SummaryWriter
        self._writer = SummaryWriter(
            log_dir=str(output_dir) if output_dir else None,
            comment=self._comment,
        )
        self._is_initialized = True
        self._run_config = run.config.copy()

        # Log hyperparameters if config provided
        if run.config:
            self._log_hparams(run.config)

        # Log run metadata as text
        if run.description:
            self._writer.add_text("run/description", run.description)
        if run.tags:
            self._writer.add_text("run/tags", ", ".join(run.tags))
        if run.group:
            self._writer.add_text("run/group", run.group)
        if run.job_type:
            self._writer.add_text("run/job_type", run.job_type)

        logger.info(
            "Initialized TensorBoard: log_dir=%s",
            self._writer.log_dir,
        )

    def _log_hparams(self, config: dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        assert self._writer is not None

        # Filter config to only include scalar types supported by TensorBoard
        hparam_dict: dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value
            elif value is None:
                hparam_dict[key] = "None"
            else:
                # Convert complex types to string representation
                hparam_dict[key] = str(value)

        if hparam_dict:
            self._writer.add_hparams(hparam_dict, {}, run_name=".")

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
        Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (global_step in TensorBoard)
            prefix: Optional prefix for metric names

        Raises:
            RuntimeError: If the tracker is not initialized
        """
        writer = self._require_writer()

        for name, value in metrics.items():
            tag = f"{prefix}{name}" if prefix else name
            if step is not None:
                writer.add_scalar(tag, value, global_step=step)
            else:
                writer.add_scalar(tag, value)

    def finish(self, exit_code: int = 0) -> None:
        """
        Finish the TensorBoard session.

        Args:
            exit_code: Exit code (0 for success, non-zero for failure).
                Note: TensorBoard doesn't use exit_code, but it's included
                for interface compatibility.

        Raises:
            RuntimeError: If the tracker is not initialized
        """
        writer = self._require_writer()

        # Log exit code as text for reference
        if exit_code != 0:
            writer.add_text("run/exit_code", str(exit_code))

        writer.flush()
        writer.close()
        self._writer = None
        self._is_initialized = False
        logger.info("TensorBoard session finished")

    def log_summary(self, summary: dict[str, Any]) -> None:
        """
        Log summary statistics.

        Args:
            summary: Summary dictionary

        Raises:
            RuntimeError: If the tracker is not initialized
        """
        writer = self._require_writer()

        # Log scalar summaries
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"summary/{key}", value)
            elif isinstance(value, dict):
                # Handle nested dicts (e.g., train_metrics, val_metrics)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        writer.add_scalar(f"summary/{key}/{sub_key}", sub_value)

        writer.flush()

    def log_model(
        self,
        model: nn.Module,
        name: str = "model",
        *,
        aliases: list[str] | None = None,  # noqa: ARG002
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a model checkpoint.

        TensorBoard doesn't have native artifact support like W&B, so this
        saves the model to the log directory.

        Args:
            model: Trained PyTorch model
            name: Name of the model file
            aliases: Ignored (included for interface compatibility)
            metadata: Optional metadata to save with the model

        Raises:
            RuntimeError: If the tracker is not initialized
        """
        writer = self._require_writer()

        # Save model to log_dir
        model_dir = Path(writer.log_dir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{name}.pt"
        save_dict: dict[str, Any] = {
            "state_dict": model.state_dict(),
            "torch_version": getattr(torch, "__version__", None),
        }
        if metadata:
            save_dict["metadata"] = metadata

        torch.save(save_dict, model_path)
        logger.info("Saved model to: %s", model_path)

    def _require_writer(self) -> "SummaryWriter":
        if not self.is_initialized or self._writer is None:
            msg = "TensorBoardTracker is not initialized. Call init()."
            raise RuntimeError(msg)
        return self._writer
