import re
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from torch_batteries import Event, EventContext, charge
from torch_batteries.utils.logging import get_logger

logger = get_logger("ModelCheckpoint")


class ModelCheckpoint:
    """
    Saves the model when a monitored metric improves.

    Args:
        stage: One of 'train' or 'val' to indicate which stage's metric to monitor.
        metric: The name of the metric to monitor.
        mode: One of 'min' or 'max'. In 'min' mode, the model is saved when the
                    monitored metric decreases. In 'max' mode, it is saved when the
                    metric increases.
        save_dir: Directory to save the model checkpoints. Defaults to the current
                    directory.
        save_path: Filename for the saved model. If None, defaults to
                    'epochs-metric=value.pth'.
        save_top_k: Saves specified number of best models. Defaults to 1.

    Example:
        ```python
        checkpoint = ModelCheckpoint(
                        stage="val",
                        metric="accuracy",
                        mode="max",
                        save_path="best_model.pth"
                    )
        battery = Battery(model=model, callbacks=[checkpoint])
        ```
    """

    def __init__(  # noqa: PLR0913
        self,
        stage: Literal["train", "val"],
        metric: str,
        mode: str = "max",
        save_dir: str = ".",
        save_path: str | None = None,
        save_top_k: int = 1,
        *,
        verbose: bool = False,
    ) -> None:
        if stage not in {"train", "val"}:
            msg = "stage must be one of 'train' or 'val'"
            raise ValueError(msg)

        self._stage = stage
        self._metric = metric
        self._save_dir = save_dir
        self._save_path = save_path
        self._save_top_k = save_top_k
        self._best_k_models: dict[str, float] = {}
        self._verbose = verbose

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

    @charge(Event.AFTER_TRAIN_EPOCH)
    def run_on_train_epoch_end(self, context: EventContext) -> None:
        if self._stage != "train":
            return

        metrics = context["train_metrics"]
        metrics["epoch"] = context["epoch"]

        if not self._save_best_model(context["model"], metrics):
            self._save_top_k_model(context["model"], metrics)

    @charge(Event.AFTER_VALIDATION)
    def run_on_validation_end(self, context: EventContext) -> None:
        if self._stage != "val":
            return

        metrics = context["val_metrics"]
        metrics["epoch"] = context["epoch"]

        if not self._save_best_model(context["model"], metrics):
            self._save_top_k_model(context["model"], metrics)

    def _save_best_model(self, model: nn.Module, metrics: dict[str, float]) -> bool:
        current_score = metrics.get(self._metric)
        if current_score is None:
            return False

        if self._monitor_op(current_score, self._best_score):
            self._best_score = current_score
            self._best_model_path = self._save_model(model, metrics, current_score)
            return True
        return False

    def _save_top_k_model(self, model: nn.Module, metrics: dict[str, float]) -> None:
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

    def _save_model(
        self, model: nn.Module, metrics: dict[str, float], current_score: float
    ) -> str:
        filename = self._format_checkpoint_name(
            self._save_path,
            metrics,
            auto_insert_metric_name=True,
        )
        filepath = f"{self._save_dir}/{filename}.pth"
        torch.save(model.state_dict(), filepath)
        if self._verbose:
            logger.info(
                "Saved model checkpoint at: %s with %s: %.2f",
                filepath,
                self._metric,
                current_score,
            )

        self._update_top_k_models(filepath, current_score)
        return filepath

    def _update_top_k_models(self, filepath: str, current_score: float) -> None:
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
        if not filename:
            filename = "{epoch}"

        groups = re.findall(r"(\{.*?)[:\}]", filename)

        groups = sorted(groups, key=lambda x: len(x), reverse=True)

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
        del self._best_k_models[filepath]
        Path(filepath).unlink()
