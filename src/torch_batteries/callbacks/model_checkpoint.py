"""Model Checkpoint Callback for torch-batteries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID, uuid4

from torch_batteries.callbacks._monitor import (
    MonitorPhase,
    require_metric,
    resolve_monitor_phase,
)
from torch_batteries.callbacks.base import Callback
from torch_batteries.events import Event, EventContext, charge
from torch_batteries.utils.logging import get_logger

from ._checkpoint_files import _checkpoint_file_transaction, _save_weights, _write_json
from ._model_checkpoint_functions import (
    _optional_path,
    _serialized_float,
    _string_float_dict,
    _validate_bool,
    _validate_save_weights_only,
)

if TYPE_CHECKING:
    from torch import nn

    from torch_batteries.trainer import Battery

logger = get_logger("callbacks.model_checkpoint")


class ModelCheckpoint(Callback):
    """Saves the model when a monitored metric improves.

    Args:
        phase: ``"train"`` or ``"validation"``.
        metric: The name of the metric to monitor
        mode: One of 'min' or 'max'. In 'min' mode, the model is saved when the
              monitored metric decreases. In 'max' mode, it is saved when the
              metric increases
        save_dir: Directory to save the model checkpoints (defaults to current directory)
        save_path: Filename for the saved model. If None, defaults to
                   'epochs-metric=value.pth'
        save_top_k: Saves specified number of best models (defaults to 1)
        save_weights_only: Save only model weights instead of full Battery state.

    Missing directories are created automatically. A `.pth` suffix is added only
    when `save_path` has no explicit suffix. Static templates gain an epoch field
    when `save_top_k` is greater than one to avoid overwriting retained weights.
    The `{epoch}` filename field uses the one-based epoch number from the event
    context, matching progress output.

    Full-state restoration uses this run's ranking manifest in ``save_dir`` to
    retain the current best files while restoring historical training state.
    Missing or invalid manifests fall back to surviving snapshot entries with a
    warning. Saving a replacement is transactional; each ranking needs one writer.

    Examples:
        ```python
        checkpoint = ModelCheckpoint(
            phase="validation",
            metric="accuracy",
            mode="max",
            save_path="best_model.pth"
        )
        battery = Battery(model=model, callbacks=[checkpoint])
        ```
    """  # noqa: E501

    def __init__(  # noqa: PLR0913
        self,
        phase: MonitorPhase | None = None,
        metric: str | None = None,
        mode: Literal["min", "max"] = "max",
        save_dir: str = ".",
        save_path: str | None = None,
        *,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        save_every_n_epochs: int = 1,
        save_final: bool = False,
        final_save_path: str | None = None,
    ) -> None:
        phase = resolve_monitor_phase(phase, required=True)
        assert phase is not None
        metric = require_metric(metric)
        if type(save_top_k) is not int or (save_top_k != -1 and save_top_k < 1):
            msg = "save_top_k must be -1 or a positive integer"
            raise ValueError(msg)
        if type(save_every_n_epochs) is not int or save_every_n_epochs < 1:
            msg = "save_every_n_epochs must be a positive integer"
            raise ValueError(msg)
        save_weights_only = _validate_save_weights_only(save_weights_only)
        save_final = _validate_bool(save_final, name="save_final")
        final_save_path = _optional_path(final_save_path)

        self._phase = phase
        self._metric = metric
        self._save_dir = save_dir
        self._save_path = save_path
        self._save_top_k = save_top_k
        self._save_every_n_epochs = save_every_n_epochs
        self._save_final = save_final
        self._final_save_path = final_save_path
        self._final_model_path: str | None = None
        self._save_weights_only = save_weights_only
        self._best_k_models: dict[str, float] = {}
        self._ranking_id: str | None = uuid4().hex

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
    def final_model_path(self) -> str | None:
        """Return the latest additional final checkpoint path, if saved."""
        return self._final_model_path

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
            "best_score": self._best_score if self._best_k_models else None,
            "kth_best_score": self._kth_best_score if self._best_k_models else None,
            "final_model_path": self._final_model_path,
            "save_weights_only": self._save_weights_only,
            "ranking_id": self._ranking_id,
            "ranking_config": self._ranking_config(),
        }
        logger.debug(
            "Serialized model checkpoint state with %d retained models.",
            len(self._best_k_models),
        )
        return state

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore checkpoint ranking state.

        Args:
            state_dict: State returned by :meth:`state_dict`.
        """
        (
            best_k_models,
            best_model_path,
            kth_best_model_path,
            best_score,
            kth_best_score,
        ) = self._validate_checkpoint_state(state_dict)
        self._best_k_models = best_k_models
        self._best_model_path = best_model_path
        self._kth_best_model_path = kth_best_model_path
        self._best_score = best_score
        self._kth_best_score = kth_best_score
        self._ranking_id = cast("str | None", state_dict.get("ranking_id"))
        self._final_model_path = cast("str | None", state_dict.get("final_model_path"))
        logger.info(
            "Restored model checkpoint state with %d retained models.",
            len(self._best_k_models),
        )

    def _validate_checkpoint_state(
        self, state_dict: dict[str, Any]
    ) -> tuple[dict[str, float], str | None, str | None, float, float]:
        """Validate and normalize all checkpoint ranking state without mutation."""
        try:
            best_k_models = _string_float_dict(state_dict["best_k_models"])
            best_model_path = _optional_path(state_dict["best_model_path"])
            kth_best_model_path = _optional_path(state_dict["kth_best_model_path"])
            empty_score = (
                (float("inf") if self._mode == "min" else float("-inf"))
                if not best_k_models
                else None
            )
            best_score = _serialized_float(
                state_dict["best_score"], empty_score=empty_score
            )
            kth_best_score = _serialized_float(
                state_dict["kth_best_score"], empty_score=empty_score
            )
            _optional_path(state_dict.get("final_model_path"))
            _validate_save_weights_only(
                state_dict["save_weights_only"],
                expected=self._save_weights_only,
            )
            self._validate_ranking_metadata(state_dict)
        except (KeyError, TypeError, ValueError) as error:
            logger.exception("Invalid model checkpoint state.")
            msg = "Invalid ModelCheckpoint checkpoint state."
            raise ValueError(msg) from error
        expected_best_path = (
            min(best_k_models, key=best_k_models.get)  # type: ignore[arg-type]
            if self._mode == "min" and best_k_models
            else max(best_k_models, key=best_k_models.get)  # type: ignore[arg-type]
            if best_k_models
            else None
        )
        expected_kth_path = (
            max(best_k_models, key=best_k_models.get)  # type: ignore[arg-type]
            if self._mode == "min" and best_k_models
            else min(best_k_models, key=best_k_models.get)  # type: ignore[arg-type]
            if best_k_models
            else None
        )
        expected_empty_score = float("inf") if self._mode == "min" else float("-inf")
        if (
            best_model_path != expected_best_path
            or kth_best_model_path != expected_kth_path
            or best_score
            != (
                best_k_models[expected_best_path]
                if expected_best_path is not None
                else expected_empty_score
            )
            or kth_best_score
            != (
                best_k_models[expected_kth_path]
                if expected_kth_path is not None
                else expected_empty_score
            )
        ):
            logger.error("Model checkpoint ranking state is inconsistent.")
            msg = "Invalid ModelCheckpoint checkpoint state."
            raise ValueError(msg)
        return (
            best_k_models,
            best_model_path,
            kth_best_model_path,
            best_score,
            kth_best_score,
        )

    def _ranking_config(self) -> dict[str, object]:
        """Identify the monitor and retention policy owning one manifest."""
        return {
            "phase": self._phase,
            "metric": self._metric,
            "mode": self._mode,
            "save_path": self._save_path,
            "save_top_k": self._save_top_k,
            "save_every_n_epochs": self._save_every_n_epochs,
            "save_final": self._save_final,
            "final_save_path": self._final_save_path,
            "save_weights_only": self._save_weights_only,
        }

    def _validate_ranking_metadata(self, state: dict[str, object]) -> None:
        """Accept legacy metadata or validate a run identity and fixed policy."""
        ranking_id = state.get("ranking_id")
        if ranking_id is not None and (
            not isinstance(ranking_id, str) or UUID(ranking_id).hex != ranking_id
        ):
            msg = "Invalid checkpoint ranking identity."
            raise ValueError(msg)
        config = state.get("ranking_config")
        if isinstance(config, dict):
            config = {
                "save_every_n_epochs": 1,
                "save_final": False,
                "final_save_path": None,
                **config,
            }
        if "ranking_config" in state and config != self._ranking_config():
            msg = "Checkpoint ranking configuration does not match."
            raise ValueError(msg)

    def _manifest_path(self) -> Path:
        """Locate this run's private ranking metadata in the configured directory."""
        return Path(self._save_dir) / f".torch-batteries-top-k-{self._ranking_id}.json"

    def _write_manifest(self) -> None:
        """Commit the current ranking after its checkpoint is safely written."""
        _write_json(
            {"schema_version": 1, "state": self.state_dict()}, self._manifest_path()
        )

    def _read_manifest(self) -> dict[str, Any]:
        """Read and validate a manifest belonging to this callback's run."""
        if self._ranking_id is None:
            msg = "Legacy checkpoint has no ranking identity."
            raise ValueError(msg)
        payload = json.loads(self._manifest_path().read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            msg = "Unsupported checkpoint ranking manifest."
            raise ValueError(msg)
        state = payload["state"]
        if not isinstance(state, dict) or state.get("ranking_id") != self._ranking_id:
            msg = "Checkpoint ranking manifest identity does not match."
            raise ValueError(msg)
        ranking = self._validate_checkpoint_state(state)[0]
        if self._save_top_k != -1 and len(ranking) > self._save_top_k:
            msg = "Checkpoint manifest exceeds the configured retention limit."
            raise ValueError(msg)
        return state

    def _reconcile_checkpoint_state(self) -> None:
        """Read current ranking metadata while keeping restoration free of writes."""
        try:
            state = self._read_manifest()
        except (OSError, ValueError, TypeError, KeyError) as error:
            state = self.state_dict()
            logger.warning(
                "Using surviving checkpoint snapshot entries; newer ranking entries "
                "cannot be recovered from the manifest: %s",
                error,
            )
        self._best_k_models = {
            path: score
            for path, score in state["best_k_models"].items()
            if Path(path).is_file()
        }
        final_path = state.get("final_model_path")
        self._final_model_path = (
            final_path
            if final_path is not None and Path(final_path).is_file()
            else None
        )
        self._refresh_ranking()

    def _refresh_ranking(self) -> None:
        """Recompute both cached extrema after filesystem reconciliation."""
        if self._best_k_models:
            select = min if self._mode == "min" else max
            self._best_model_path = select(
                self._best_k_models, key=self._best_k_models.__getitem__
            )
            self._best_score = self._best_k_models[self._best_model_path]
        else:
            self._best_model_path = None
            self._best_score = float("inf") if self._mode == "min" else float("-inf")
        self._refresh_kth_best()

    @charge(Event.AFTER_TRAIN_EPOCH)
    def run_on_train_epoch_end(self, context: EventContext) -> None:
        """Save model checkpoint after training epoch if metric improved.

        Args:
            context: Event context containing training metrics and model.
        """
        if self._phase != "train" or context["epoch"] % self._save_every_n_epochs:
            return

        metrics = {**context["train_metrics"], "epoch": context["epoch"]}

        battery = context.get("battery")
        self._save_top_k_model(battery, context["model"], metrics)

    @charge(Event.AFTER_VALIDATION_EPOCH)
    def run_on_validation_end(self, context: EventContext) -> None:
        """Save model checkpoint after validation if metric improved.

        Args:
            context: Event context containing validation metrics and model.
        """
        if self._phase != "validation" or context["epoch"] % self._save_every_n_epochs:
            return

        metrics = {**context["val_metrics"], "epoch": context["epoch"]}

        battery = context.get("battery")
        self._save_top_k_model(battery, context["model"], metrics)

    @charge(Event.AFTER_TRAIN)
    def run_on_train_end(self, context: EventContext) -> None:
        """Save the latest final model independently of periodic retention.

        Args:
            context: Completed training context, including optional validation metrics.
        """
        if not self._save_final:
            return
        template = self._final_save_path
        if template is None:
            original = Path(self._save_path or "{epoch}")
            template = str(original.with_name(f"final-{original.name}"))
        phase_metrics = (
            context.get("train_metrics", {})
            if self._phase == "train"
            else context.get("val_metrics", {})
        )
        metrics = {**phase_metrics, "epoch": context["epoch"]}
        path = self._checkpoint_path(
            template, metrics, replace_path=self._final_model_path
        )
        previous_state = self.state_dict()
        previous_final = self._final_model_path
        if self._ranking_id is None:
            self._ranking_id = uuid4().hex
        self._final_model_path = str(path)
        self._commit_checkpoint(
            context.get("battery"), context["model"], path, previous_state
        )
        if previous_final is not None and previous_final != str(path):
            self._delete_checkpoint_file(previous_final)
        logger.info("Saved final model checkpoint at: %s", path)

    def _save_top_k_model(
        self,
        battery: Battery | None,
        model: nn.Module,
        metrics: dict[str, float],
    ) -> None:
        """Save model if it's in top-k best models.

        Args:
            model: The PyTorch model to save.
            metrics: Dictionary of current metrics.
        """
        current_score = metrics.get(self._metric)
        if current_score is None:
            logger.error(
                "Checkpoint metric is unavailable: phase=%s, metric=%s",
                self._phase,
                self._metric,
            )
            msg = (
                f"ModelCheckpoint metric '{self._metric}' is unavailable "
                f"for phase '{self._phase}'."
            )
            raise ValueError(msg)

        is_best = self._monitor_op(current_score, self._best_score)
        qualifies = (
            self._save_top_k == -1
            or len(self._best_k_models) < self._save_top_k
            or self._monitor_op(current_score, self._kth_best_score)
        )
        logger.debug(
            "Checkpoint candidate: metric=%s, score=%s, best=%s, kth=%s, qualifies=%s",
            self._metric,
            current_score,
            self._best_score,
            self._kth_best_score,
            qualifies,
        )
        if not qualifies:
            logger.debug(
                "Checkpoint candidate skipped because it is outside the top %d.",
                self._save_top_k,
            )
            return

        self._save_model(
            battery,
            model,
            metrics,
            current_score,
            is_best=is_best,
        )

    def _save_model(
        self,
        battery: Battery | None,
        model: nn.Module,
        metrics: dict[str, float],
        current_score: float,
        *,
        is_best: bool,
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
        path = self._checkpoint_path(
            filename_template,
            metrics,
            replace_path=self._best_model_path if self._save_top_k == 1 else None,
        )
        filepath = str(path)

        previous_state = self.state_dict()
        if self._ranking_id is None:
            self._ranking_id = uuid4().hex
        if is_best:
            self._best_score = current_score
            self._best_model_path = filepath
        displaced_path = self._update_top_k_models(filepath, current_score)
        self._commit_checkpoint(battery, model, path, previous_state)

        if displaced_path is not None:
            self._delete_checkpoint_file(displaced_path)
        logger.info(
            "Saved model checkpoint at: %s with %s: %.2f",
            filepath,
            self._metric,
            current_score,
        )

        return filepath

    def _checkpoint_path(
        self,
        template: str | None,
        metrics: dict[str, float],
        *,
        replace_path: str | None,
    ) -> Path:
        """Format a destination while preserving files not owned by this save."""
        filename = self._format_checkpoint_name(template, metrics)
        if not self._has_explicit_suffix(template):
            filename = f"{filename}.pth"
        path = Path(self._save_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        original = path
        version = 1
        while path.exists() and str(path) != replace_path:
            path = original.with_name(f"{original.stem}-v{version}{original.suffix}")
            version += 1
        return path

    def _commit_checkpoint(
        self,
        battery: Battery | None,
        model: nn.Module,
        path: Path,
        previous_state: dict[str, object],
    ) -> None:
        """Commit checkpoint and manifest together, rolling back failed writes."""
        try:
            with _checkpoint_file_transaction(path):
                if self._save_weights_only or battery is None:
                    if battery is None and not self._save_weights_only:
                        logger.warning(
                            "ModelCheckpoint context has no Battery; "
                            "falling back to raw weights."
                        )
                    _save_weights(model.state_dict(), path)
                else:
                    battery.save_checkpoint(str(path))
                self._write_manifest()
        except BaseException:
            self.load_state_dict(previous_state)
            logger.exception("Failed to save model checkpoint at: %s", path)
            raise

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

    def _update_top_k_models(self, filepath: str, current_score: float) -> str | None:
        """Update top-k tracking and return the checkpoint to remove.

        Args:
            filepath: Path to the newly saved model.
            current_score: The metric score of the saved model.
        """
        self._best_k_models[filepath] = current_score
        displaced_path: str | None = None
        if self._save_top_k != -1 and len(self._best_k_models) > self._save_top_k:
            if self._mode == "min":
                displaced_path = max(
                    self._best_k_models,
                    key=self._best_k_models.get,  # type: ignore[arg-type]
                )
            else:
                displaced_path = min(
                    self._best_k_models,
                    key=self._best_k_models.get,  # type: ignore[arg-type]
                )
            del self._best_k_models[displaced_path]

        self._refresh_kth_best()
        logger.debug(
            "Checkpoint ranking updated: retained=%d, save_top_k=%d, kth_score=%s",
            len(self._best_k_models),
            self._save_top_k,
            self._kth_best_score,
        )
        return displaced_path

    def _refresh_kth_best(self) -> None:
        """Synchronize the cached K-th checkpoint with retained models."""
        if not self._best_k_models:
            self._kth_best_model_path = None
            self._kth_best_score = (
                float("inf") if self._mode == "min" else float("-inf")
            )
            return

        if self._mode == "min":
            self._kth_best_model_path = max(
                self._best_k_models,
                key=self._best_k_models.get,  # type: ignore[arg-type]
            )
        else:
            self._kth_best_model_path = min(
                self._best_k_models,
                key=self._best_k_models.get,  # type: ignore[arg-type]
            )
        self._kth_best_score = self._best_k_models[self._kth_best_model_path]

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
        self._refresh_kth_best()
        self._delete_checkpoint_file(filepath)

    @staticmethod
    def _delete_checkpoint_file(filepath: str) -> None:
        """Delete a checkpoint file after its replacement is safely written."""
        path = Path(filepath)
        if path.exists():
            path.unlink()
            logger.info("Deleted model checkpoint at: %s", filepath)
        else:
            logger.warning(
                "Checkpoint file was already missing during cleanup: %s", filepath
            )
