"""Model Checkpoint Callback for torch-batteries."""

import re
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from torch_batteries.callbacks.base import Callback
from torch_batteries.events import Event, EventContext, charge
from torch_batteries.utils.logging import get_logger

logger = get_logger("ModelCheckpoint")


def _optional_string(value: object) -> str | None:
    """Validate an optional serialized path."""
    if value is None or isinstance(value, str):
        return value
    msg = "checkpoint path must be a string or None"
    raise TypeError(msg)


def _string_float_dict(value: object) -> dict[str, float]:
    """Validate serialized checkpoint ranking data."""
    if not isinstance(value, dict):
        msg = "best_k_models must be a dictionary"
        raise TypeError(msg)
    return {str(path): _serialized_float(score) for path, score in value.items()}


def _serialized_float(value: object) -> float:
    """Validate a serialized numeric value."""
    if not isinstance(value, (int, float)):
        msg = "checkpoint score must be numeric"
        raise TypeError(msg)
    return float(value)


class ModelCheckpoint(Callback):
    """Saves the model when a monitored metric improves.

    Args:
        stage: One of 'train' or 'val' to indicate which stage's metric to monitor
        metric: The name of the metric to monitor
        mode: One of 'min' or 'max'. In 'min' mode, the model is saved when the
              monitored metric decreases. In 'max' mode, it is saved when the
              metric increases
        save_dir: Directory to save the model checkpoints (defaults to current directory)
        save_path: Filename for the saved model. If None, defaults to
                   'epochs-metric=value.pth'
        save_top_k: Saves specified number of best models (defaults to 1)

    Missing directories are created automatically. A `.pth` suffix is added only
    when `save_path` has no explicit suffix. Static templates gain an epoch field
    when `save_top_k` is greater than one to avoid overwriting retained weights.

    Examples:
        ```python
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            mode="max",
            save_path="best_model.pth"
        )
        battery = Battery(model=model, callbacks=[checkpoint])
        ```
    """  # noqa: E501

    def __init__(  # noqa: PLR0913
        self,
        stage: Literal["train", "val"],
        metric: str,
        mode: Literal["min", "max"] = "max",
        save_dir: str = ".",
        save_path: str | None = None,
        save_top_k: int = 1,
    ) -> None:
        if stage not in {"train", "val"}:
            msg = "stage must be one of 'train' or 'val'"
            raise ValueError(msg)
        if save_top_k < 1:
            msg = "save_top_k must be greater than or equal to one"
            raise ValueError(msg)

        self._stage = stage
        self._metric = metric
        self._save_dir = save_dir
        self._save_path = save_path
        self._save_top_k = save_top_k
        self._best_k_models: dict[str, float] = {}

        self._best_model_path: str | None = None
        self._kth_best_model_path: str | None = None

        if mode not in {"min", "max"}:
            msg = "mode must be one of 'min' or 'max'"
            raise ValueError(msg)
        self._mode = mode
        if self._mode == "min":
            self._monitor_op = lambda current, best: current < best
            self._best_score = float("inf")
            self._kth_best_score = float("inf")
        else:
            self._monitor_op = lambda current, best: current > best
            self._best_score = float("-inf")
            self._kth_best_score = float("-inf")

        self.CHECKPOINT_JOIN_CHAR = "-"
        self.CHECKPOINT_EQUALS_CHAR = "="

    @property
    def best_model_path(self) -> str | None:
        """Returns the path of the best saved model."""
        return self._best_model_path

    @property
    def best_score(self) -> float | None:
        """Returns the best score achieved by the monitored metric."""
        return self._best_score

    @property
    def best_k_models(self) -> dict[str, float]:
        """Returns a dictionary of the top K saved models and their scores."""
        return self._best_k_models

    def state_dict(self) -> dict[str, object]:
        """Return checkpoint ranking state for training resumption."""
        state: dict[str, object] = {
            "best_k_models": dict(self._best_k_models),
            "best_model_path": self._best_model_path,
            "kth_best_model_path": self._kth_best_model_path,
            "best_score": self._best_score,
            "kth_best_score": self._kth_best_score,
        }
        logger.debug(
            "Serialized model checkpoint state with %d retained models.",
            len(self._best_k_models),
        )
        return state

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore checkpoint ranking state."""
        try:
            self._best_k_models = _string_float_dict(state_dict["best_k_models"])
            self._best_model_path = _optional_string(state_dict["best_model_path"])
            self._kth_best_model_path = _optional_string(
                state_dict["kth_best_model_path"]
            )
            self._best_score = _serialized_float(state_dict["best_score"])
            self._kth_best_score = _serialized_float(state_dict["kth_best_score"])
        except (KeyError, TypeError, ValueError) as error:
            logger.exception("Invalid model checkpoint state.")
            msg = "Invalid ModelCheckpoint checkpoint state."
            raise ValueError(msg) from error
        logger.info(
            "Restored model checkpoint state with %d retained models.",
            len(self._best_k_models),
        )

    @charge(Event.AFTER_TRAIN_EPOCH)
    def run_on_train_epoch_end(self, context: EventContext) -> None:
        """Save model checkpoint after training epoch if metric improved.

        Args:
            context: Event context containing training metrics and model.
        """
        if self._stage != "train":
            return

        metrics = {**context["train_metrics"], "epoch": context["epoch"]}

        if not self._save_best_model(context["model"], metrics):
            self._save_top_k_model(context["model"], metrics)

    @charge(Event.AFTER_VALIDATION)
    def run_on_validation_end(self, context: EventContext) -> None:
        """Save model checkpoint after validation if metric improved.

        Args:
            context: Event context containing validation metrics and model.
        """
        if self._stage != "val":
            return

        metrics = {**context["val_metrics"], "epoch": context["epoch"]}

        if not self._save_best_model(context["model"], metrics):
            self._save_top_k_model(context["model"], metrics)

    def _save_best_model(self, model: nn.Module, metrics: dict[str, float]) -> bool:
        """Save model if it achieves new best score.

        Args:
            model: The PyTorch model to save.
            metrics: Dictionary of current metrics.

        Returns:
            True if model was saved as new best, False otherwise.
        """
        current_score = metrics.get(self._metric)
        if current_score is None:
            logger.warning(
                "Checkpoint monitor metric '%s' is missing; checkpoint was skipped.",
                self._metric,
            )
            return False

        logger.debug(
            "Checkpoint candidate: metric=%s, score=%s, best=%s, mode=%s",
            self._metric,
            current_score,
            self._best_score,
            self._mode,
        )

        if self._monitor_op(current_score, self._best_score):
            self._best_score = current_score
            self._best_model_path = self._save_model(model, metrics, current_score)
            return True
        return False

    def _save_top_k_model(self, model: nn.Module, metrics: dict[str, float]) -> None:
        """Save model if it's in top-k best models.

        Args:
            model: The PyTorch model to save.
            metrics: Dictionary of current metrics.
        """
        current_score = metrics.get(self._metric)
        if current_score is None:
            return

        if len(self._best_k_models) < self._save_top_k or self._monitor_op(
            current_score, self._kth_best_score
        ):
            self._save_model(model, metrics, current_score)

        if len(self._best_k_models) == self._save_top_k:
            if self._mode == "min":
                self._kth_best_model_path = max(
                    self._best_k_models,
                    key=self._best_k_models.get,  # type: ignore[arg-type]
                )
                self._kth_best_score = self._best_k_models[self._kth_best_model_path]
            else:
                self._kth_best_model_path = min(
                    self._best_k_models,
                    key=self._best_k_models.get,  # type: ignore[arg-type]
                )
                self._kth_best_score = self._best_k_models[self._kth_best_model_path]
        logger.debug(
            "Checkpoint ranking updated: retained=%d, save_top_k=%d, kth_score=%s",
            len(self._best_k_models),
            self._save_top_k,
            self._kth_best_score,
        )

    def _save_model(
        self, model: nn.Module, metrics: dict[str, float], current_score: float
    ) -> str:
        """Save model to disk and update top-k tracking.

        Args:
            model: The PyTorch model to save.
            metrics: Dictionary of current metrics.
            current_score: The current metric score.

        Returns:
            Path to the saved model file.
        """
        filename_template = self._ensure_unique_template(self._save_path)
        filename = self._format_checkpoint_name(
            filename_template,
            metrics,
            auto_insert_metric_name=True,
        )
        if not self._has_explicit_suffix(filename_template):
            filename = f"{filename}.pth"
        path = Path(self._save_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        filepath = str(path)
        torch.save(model.state_dict(), filepath)
        logger.info(
            "Saved model checkpoint at: %s with %s: %.2f",
            filepath,
            self._metric,
            current_score,
        )

        self._update_top_k_models(filepath, current_score)
        return filepath

    def _ensure_unique_template(self, filename: str | None) -> str | None:
        """Add an epoch field when top-k checkpoints would share one filename."""
        if filename is None or self._save_top_k == 1 or "{epoch" in filename:
            return filename

        suffix_match = re.search(r"(\.[A-Za-z][A-Za-z0-9]*)$", filename)
        if suffix_match is None:
            return f"{filename}-{{epoch}}"
        suffix_start = suffix_match.start()
        return f"{filename[:suffix_start]}-{{epoch}}{filename[suffix_start:]}"

    @staticmethod
    def _has_explicit_suffix(filename: str | None) -> bool:
        """Check whether a template ends in a user-provided file suffix."""
        return filename is not None and bool(
            re.search(r"\.[A-Za-z][A-Za-z0-9]*$", filename)
        )

    def _update_top_k_models(self, filepath: str, current_score: float) -> None:
        """Update top-k models tracking and remove worst model if needed.

        Args:
            filepath: Path to the newly saved model.
            current_score: The metric score of the saved model.
        """
        self._best_k_models[filepath] = current_score

        if len(self._best_k_models) > self._save_top_k:
            if self._mode == "min":
                worst_model = max(self._best_k_models, key=self._best_k_models.get)  # type: ignore[arg-type]
            else:
                worst_model = min(self._best_k_models, key=self._best_k_models.get)  # type: ignore[arg-type]
            self._delete_saved_model(worst_model)

    def _format_checkpoint_name(
        self,
        filename: str | None,
        metrics: dict[str, float],
        prefix: str | None = None,
        *,
        auto_insert_metric_name: bool = True,
    ) -> str:
        """Format checkpoint filename with metrics values.

        Args:
            filename: Template filename with placeholders like {epoch}, {metric}.
            metrics: Dictionary of metric values to insert.
            prefix: Optional prefix to add to filename.
            auto_insert_metric_name: Whether to add metric names to values.

        Returns:
            Formatted checkpoint filename.
        """
        if not filename:
            filename = "{epoch}"

        groups = re.findall(r"(\{.*?)[:\}]", filename)

        groups = sorted(groups, key=len, reverse=True)

        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                filename = filename.replace(
                    group, name + self.CHECKPOINT_EQUALS_CHAR + "{" + name
                )

            filename = filename.replace(group, f"{{0[{name}]")

        filename = filename.format(metrics)

        if prefix is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename

    def _delete_saved_model(self, filepath: str) -> None:
        """Delete a saved model file from disk and tracking.

        Args:
            filepath: Path to the model file to delete.
        """
        del self._best_k_models[filepath]
        path = Path(filepath)
        if path.exists():
            path.unlink()
            logger.info("Deleted model checkpoint at: %s", filepath)
        else:
            logger.warning(
                "Checkpoint file was already missing during cleanup: %s", filepath
            )
