import re
from pathlib import Path

import torch
from torch import nn

from torch_batteries import Event, EventContext, charge
from torch_batteries.utils.logging import get_logger


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
        stage: str,
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

        self.stage = stage
        self.metric = metric
        self.save_dir = save_dir
        self.save_path = save_path
        self.save_top_k = save_top_k
        self.best_k_models: dict[str, float] = {}
        self.verbose = verbose

        self.best_model_path: str | None = None
        self.kth_best_model_path: str | None = None

        if mode not in {"min", "max"}:
            msg = "mode must be one of 'min' or 'max'"
            raise ValueError(msg)
        self.mode = mode
        if self.mode == "min":
            self.monitor_op = lambda current, best: current < best
            self.best_score = float("inf")
            self.kth_best_score = float("inf")
        else:
            self.monitor_op = lambda current, best: current > best
            self.best_score = float("-inf")
            self.kth_best_score = float("-inf")

        self.log = get_logger("ModelCheckpoint")

        self.CHECKPOINT_JOIN_CHAR = "-"
        self.CHECKPOINT_EQUALS_CHAR = "="

    @charge(Event.AFTER_TRAIN_EPOCH)
    def run_on_train_epoch_end(self, context: EventContext) -> None:
        if self.stage != "train":
            return

        metrics = context["train_metrics"]
        metrics["epoch"] = context["epoch"]

        if not self._save_best_model(context["model"], metrics):
            self._save_top_k_model(context["model"], metrics)

    @charge(Event.AFTER_VALIDATION)
    def run_on_validation_end(self, context: EventContext) -> None:
        if self.stage != "val":
            return

        metrics = context["val_metrics"]
        metrics["epoch"] = context["epoch"]

        if not self._save_best_model(context["model"], metrics):
            self._save_top_k_model(context["model"], metrics)

    def _save_best_model(self, model: nn.Module, metrics: dict[str, float]) -> bool:
        current_score = metrics.get(self.metric)
        if current_score is None:
            return False

        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.best_model_path = self._save_model(model, metrics, current_score)
            return True
        return False

    def _save_top_k_model(self, model: nn.Module, metrics: dict[str, float]) -> None:
        current_score = metrics.get(self.metric)
        if current_score is None:
            return

        if len(self.best_k_models) < self.save_top_k or self.monitor_op(
            current_score, self.kth_best_score
        ):
            self._save_model(model, metrics, current_score)

        if len(self.best_k_models) == self.save_top_k:
            if self.mode == "min":
                self.kth_best_model_path = max(
                    self.best_k_models,
                    key=self.best_k_models.get,  # type: ignore
                )
                self.kth_best_score = self.best_k_models[self.kth_best_model_path]
            else:
                self.kth_best_model_path = min(
                    self.best_k_models,
                    key=self.best_k_models.get,  # type: ignore
                )
                self.kth_best_score = self.best_k_models[self.kth_best_model_path]

    def _save_model(
        self, model: nn.Module, metrics: dict[str, float], current_score: float
    ) -> str:
        filename = self._format_checkpoint_name(
            self.save_path,
            metrics,
            auto_insert_metric_name=True,
        )
        filepath = f"{self.save_dir}/{filename}.pth"
        torch.save(model.state_dict(), filepath)
        if self.verbose:
            self.log.info(
                "Saved model checkpoint at: %s with %s: %.2f",
                filepath,
                self.metric,
                current_score,
            )

        self._update_top_k_models(filepath, current_score)
        return filepath

    def _update_top_k_models(self, filepath: str, current_score: float) -> None:
        self.best_k_models[filepath] = current_score

        if len(self.best_k_models) > self.save_top_k:
            if self.mode == "min":
                worst_model = max(self.best_k_models, key=self.best_k_models.get)  # type: ignore
            else:
                worst_model = min(self.best_k_models, key=self.best_k_models.get)  # type: ignore
            del self.best_k_models[worst_model]
            Path(worst_model).unlink()

    def _format_checkpoint_name(
        self,
        filename: str | None,
        metrics: dict[str, float],
        prefix: str | None = None,
        *,
        auto_insert_metric_name: bool = True,
    ) -> str:
        if not filename:
            filename = "{epoch}" + self.CHECKPOINT_JOIN_CHAR

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
