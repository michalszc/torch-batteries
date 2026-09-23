"""Filesystem experiment tracking backend."""

import csv
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from torch import nn

from torch_batteries.tracking.base import ExperimentTracker
from torch_batteries.tracking.types import Run
from torch_batteries.utils.logging import get_logger

logger = get_logger("tracking.local")


class LocalTracker(ExperimentTracker):
    """Store run configuration, metrics, and summary beneath a local directory.

    Each ``init`` allocates ``save_dir/model_name/version_N``. Metric rows are
    indexed by epoch, with a column for each metric supplied by the callback.
    Model artifacts are not saved by this backend.

    Args:
        model_name: Directory name used to group this model's runs.
        save_dir: Root directory for local experiments.
    """

    __slots__ = (
        "_fields",
        "_is_initialized",
        "_model_name",
        "_rows",
        "_run_dir",
        "_save_dir",
    )

    def __init__(
        self, model_name: str, save_dir: str | Path = "my_experiments"
    ) -> None:
        if (
            not model_name
            or model_name in {".", ".."}
            or "/" in model_name
            or "\\" in model_name
        ):
            msg = "model_name must be a non-empty directory name."
            raise ValueError(msg)
        self._model_name = model_name
        self._save_dir = Path(save_dir)
        self._run_dir: Path | None = None
        self._is_initialized = False
        self._rows: dict[int, dict[str, float | int]] = {}
        self._fields: list[str] = []

    @property
    def metric_granularity(self) -> str:
        """Record completed epochs instead of individual optimization steps."""
        return "epoch"

    @property
    def is_initialized(self) -> bool:
        """Whether a run is currently active."""
        return self._is_initialized

    @property
    def run_dir(self) -> Path | None:
        """Directory allocated for the current or most recent run."""
        return self._run_dir

    def _require_run_dir(self) -> Path:
        if not self._is_initialized or self._run_dir is None:
            msg = "LocalTracker is not initialized. Call init()."
            raise RuntimeError(msg)
        return self._run_dir

    def init(self, run: Run) -> None:
        """Create a new version directory and write run hyperparameters.

        Args:
            run: Run metadata and explicit hyperparameters.
        """
        if self._is_initialized:
            msg = "LocalTracker is already initialized."
            raise RuntimeError(msg)
        model_dir = self._save_dir / self._model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        version = 0
        while True:
            run_dir = model_dir / f"version_{version}"
            try:
                run_dir.mkdir()
                break
            except FileExistsError:
                version += 1
        hparams = {
            "name": run.name,
            "group": run.group,
            "job_type": run.job_type,
            "description": run.description,
            "tags": run.tags,
            **run.config,
        }
        with (run_dir / "hparams.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(hparams, stream, sort_keys=True)
        with (run_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            csv.writer(stream).writerow(["epoch"])
        self._rows = {}
        self._fields = []
        self._run_dir = run_dir
        self._is_initialized = True
        logger.info("Initialized local experiment run: path=%s", run_dir)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        """Merge prefixed metrics into one CSV row for the specified epoch.

        Args:
            metrics: Metric names and scalar values.
            step: Epoch number; omitted values allocate the next epoch.
            prefix: Optional metric name prefix.
        """
        run_dir = self._require_run_dir()
        if step is not None and (isinstance(step, bool) or step < 0):
            msg = "LocalTracker metric epoch must be a non-negative integer."
            raise ValueError(msg)
        epoch = step if step is not None else max(self._rows, default=0) + 1
        row = self._rows.setdefault(epoch, {"epoch": epoch})
        for name, value in metrics.items():
            if name == "epoch":
                continue
            column = f"{prefix or ''}{name}"
            row[column] = value
            if column not in self._fields:
                self._fields.append(column)
        with (run_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", *self._fields])
            writer.writeheader()
            writer.writerows(self._rows[key] for key in sorted(self._rows))
        logger.debug(
            "Logged local epoch metrics: path=%s, epoch=%d, keys=%s",
            run_dir,
            epoch,
            sorted(metrics),
        )

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Write the latest run summary as YAML.

        Args:
            summary: Summary values to save.
        """
        run_dir = self._require_run_dir()

        def latest(value: Any) -> Any:
            if isinstance(value, list):
                return latest(value[-1]) if value else None
            if isinstance(value, dict):
                return {key: latest(item) for key, item in value.items()}
            return value

        latest_summary = {key: latest(value) for key, value in summary.items()}
        with (run_dir / "summary.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(latest_summary, stream, sort_keys=True)
        logger.debug("Logged local summary: path=%s", run_dir)

    def log_model(
        self,
        model: nn.Module,  # noqa: ARG002
        name: str = "model",
        *,
        aliases: list[str] | None = None,  # noqa: ARG002
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """Accept the common tracker hook without saving model artifacts.

        Args:
            model: Model whose artifacts are intentionally skipped.
            name: Artifact name supplied by the callback.
            aliases: Optional artifact aliases, unused here.
            metadata: Optional artifact metadata, unused here.
        """
        self._require_run_dir()
        logger.info("LocalTracker does not save model artifacts: name=%s", name)

    def finish(self, exit_code: int = 0) -> None:
        """Close the run and write its exit status to the summary.

        Args:
            exit_code: Zero for success, nonzero for failure.
        """
        run_dir = self._require_run_dir()
        summary_path = run_dir / "summary.yaml"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as stream:
                summary = yaml.safe_load(stream) or {}
        summary["exit_code"] = exit_code
        with summary_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(summary, stream, sort_keys=True)
        self._is_initialized = False
        logger.info(
            "Finished local experiment run: path=%s, exit_code=%d", run_dir, exit_code
        )
