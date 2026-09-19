"""Checkpoint persistence for ``torch_batteries.Battery``."""

import copy
import importlib
import random
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from torch_batteries.callbacks.base import Callback
from torch_batteries.data import DataPack
from torch_batteries.utils.device import move_to_device
from torch_batteries.utils.logging import get_logger

from ._state import BatteryStateMixin

if TYPE_CHECKING:
    from torch_batteries.trainer.types import FitResult

logger = get_logger("trainer._checkpoint")

_CHECKPOINT_SCHEMA_VERSION = 4
_SUPPORTED_CHECKPOINT_SCHEMAS = {_CHECKPOINT_SCHEMA_VERSION}


class CheckpointMixin(BatteryStateMixin):
    """Implement full-state and raw-model checkpoint persistence."""

    __slots__ = ()

    def _checkpoint_metric_states(self) -> dict[str, Any]:
        """Serialize aggregate and named dataset metric states by phase."""
        return {
            "phases": {
                phase: manager.state_dict()
                for phase, manager in self._metric_managers.items()
            },
            "datasets": {
                phase: {
                    name: manager.state_dict() for name, manager in managers.items()
                }
                for phase, managers in self._dataset_metric_managers.items()
            },
        }

    def _validate_metric_states(self, state: object) -> None:
        """Check checkpoint metric names before mutating runtime state.

        Args:
            state: Untrusted serialized metric state.
        """
        if not isinstance(state, dict):
            msg = "Invalid metric state in training checkpoint."
            raise TypeError(msg)
        phases = state.get("phases")
        datasets = state.get("datasets")
        if not isinstance(phases, dict) or not isinstance(datasets, dict):
            msg = "Configured metric states do not match checkpoint state."
            raise ValueError(msg)  # noqa: TRY004
        expected_phases = set(self._metric_managers)
        if set(phases) != expected_phases or set(datasets) != expected_phases:
            msg = "Configured metric states do not match checkpoint state."
            raise ValueError(msg)
        for phase, manager in self._metric_managers.items():
            expected_names = set(manager.state_dict())
            if (
                not isinstance(phases[phase], dict)
                or set(phases[phase]) != expected_names
            ):
                msg = "Configured metric states do not match checkpoint state."
                raise ValueError(msg)
            if not isinstance(datasets[phase], dict):
                msg = "Invalid metric state in training checkpoint."
                raise TypeError(msg)
            for name, saved in datasets[phase].items():
                if not isinstance(name, str) or not isinstance(saved, dict):
                    msg = "Invalid metric state in training checkpoint."
                    raise TypeError(msg)
                if set(saved) != expected_names:
                    msg = "Configured metric states do not match checkpoint state."
                    raise ValueError(msg)

    def _restore_metric_states(self, state: dict[str, Any]) -> None:
        """Restore phase and dataset metric states after validation.

        Args:
            state: Validated serialized metric state.
        """
        for phase, manager in self._metric_managers.items():
            manager.load_state_dict(state["phases"][phase])
        self._dataset_metric_managers = {phase: {} for phase in self._metric_managers}
        for phase, datasets in state["datasets"].items():
            for name, saved in datasets.items():
                self._manager_for_dataset(phase, name).load_state_dict(saved)

    def _checkpoint_callbacks(self) -> list[Callback]:
        """Return configured callbacks participating in checkpoint state."""
        return [
            callback for callback in self._callbacks if isinstance(callback, Callback)
        ]

    @staticmethod
    def _callback_identifier(callback: Callback) -> str:
        """Return a stable module-qualified callback identifier."""
        callback_type = type(callback)
        return f"{callback_type.__module__}.{callback_type.__qualname__}"

    @staticmethod
    def _data_pack_identifier(data_pack: DataPack) -> str:
        """Return the stable qualified identifier stored in checkpoints."""
        data_pack_type = type(data_pack)
        return f"{data_pack_type.__module__}.{data_pack_type.__qualname__}"

    def _checkpoint_data_pack(self) -> dict[str, Any] | None:
        """Build and validate resumable DataPack state."""
        if self._data_pack is None:
            return None
        state: object = self._data_pack.state_dict()
        if not isinstance(state, dict):
            msg = "DataPack state_dict() must return a dictionary."
            raise TypeError(msg)
        return {
            "type": self._data_pack_identifier(self._data_pack),
            "state": state,
        }

    def _validate_checkpoint_data_pack(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Validate and return DataPack state from a current checkpoint."""
        if "data_pack" not in payload:
            msg = "Training checkpoint is missing fields: ['data_pack']."
            raise ValueError(msg)
        saved_data_pack = payload["data_pack"]
        if saved_data_pack is None:
            if self._data_pack is not None:
                msg = "Configured DataPack does not match checkpoint state."
                raise ValueError(msg)
            return None
        if not isinstance(saved_data_pack, dict):
            msg = "Invalid DataPack state in training checkpoint."
            raise TypeError(msg)
        saved_type = saved_data_pack.get("type")
        saved_state = saved_data_pack.get("state")
        if not isinstance(saved_type, str) or not isinstance(saved_state, dict):
            msg = "Invalid DataPack state in training checkpoint."
            raise TypeError(msg)
        if self._data_pack is None:
            msg = "Checkpoint requires a configured DataPack."
            raise ValueError(msg)
        expected_type = self._data_pack_identifier(self._data_pack)
        if saved_type != expected_type:
            msg = (
                "Configured DataPack does not match checkpoint state: "
                f"expected '{saved_type}', got '{expected_type}'."
            )
            raise ValueError(msg)
        return saved_state

    @staticmethod
    def _loader_generators(
        loader: torch.utils.data.DataLoader,
    ) -> dict[str, torch.Generator]:
        """Return distinct reachable loader generators keyed by access path."""
        sampler = getattr(loader, "sampler", None)
        batch_sampler = getattr(loader, "batch_sampler", None)
        batch_sampler_sampler = getattr(batch_sampler, "sampler", None)
        candidates = (
            ("loader.generator", getattr(loader, "generator", None)),
            ("sampler.generator", getattr(sampler, "generator", None)),
            (
                "batch_sampler.sampler.generator",
                getattr(batch_sampler_sampler, "generator", None),
            ),
        )
        generators: dict[str, torch.Generator] = {}
        seen: set[int] = set()
        for path, generator in candidates:
            if not isinstance(generator, torch.Generator) or id(generator) in seen:
                continue
            generators[path] = generator
            seen.add(id(generator))
        return generators

    @staticmethod
    def _loader_state_key(phase: str, dataset_name: str | None) -> str:
        """Return the stable state key for one phase loader."""
        return phase if dataset_name is None else f"{phase}:{dataset_name}"

    def _capture_loader_generator_state(
        self,
        phase: str,
        loader: torch.utils.data.DataLoader,
        *,
        dataset_name: str | None = None,
    ) -> None:
        """Retain current states for distinct generators reachable from a loader."""
        key = self._loader_state_key(phase, dataset_name)
        self._loader_generator_states[key] = {
            path: generator.get_state().clone()
            for path, generator in self._loader_generators(loader).items()
        }
        logger.debug(
            "Captured %d loader generator states for %s.",
            len(self._loader_generator_states[key]),
            key,
        )

    def _restore_loader_generator_state(
        self,
        phase: str,
        loader: torch.utils.data.DataLoader,
        *,
        dataset_name: str | None = None,
    ) -> None:
        """Apply pending generator state before a resumed loader is iterated."""
        key = self._loader_state_key(phase, dataset_name)
        saved_states = self._pending_loader_generator_states.pop(key, None)
        if saved_states is None:
            return
        generators = self._loader_generators(loader)
        for path, state in saved_states.items():
            generator = generators.get(path)
            if generator is None:
                logger.warning(
                    "Saved loader generator is unavailable: phase=%s, path=%s",
                    key,
                    path,
                )
                continue
            try:
                generator.set_state(state.cpu())
            except (RuntimeError, TypeError):
                logger.warning(
                    "Saved loader generator is incompatible: phase=%s, path=%s",
                    key,
                    path,
                    exc_info=True,
                )
        logger.debug("Restored pending loader generator states for %s.", key)

    @staticmethod
    def _capture_global_rng_state() -> dict[str, Any]:
        """Capture available Python, PyTorch, accelerator, and NumPy RNG state."""
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state().clone(),
        }
        if torch.cuda.is_available():
            state["cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
        if torch.backends.mps.is_available():
            state["mps"] = torch.mps.get_rng_state().clone()
        try:
            numpy = importlib.import_module("numpy")
        except ModuleNotFoundError:
            logger.debug("NumPy is unavailable; its RNG state was not checkpointed.")
        else:
            numpy_state = numpy.random.get_state()
            state["numpy"] = {
                "bit_generator": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].copy()),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            }
        return state

    @staticmethod
    def _restore_global_rng_state(state: dict[str, Any]) -> None:
        """Restore every saved RNG source available on the current runtime."""
        random.setstate(state["python"])

        numpy_state = state.get("numpy")
        if numpy_state is not None:
            try:
                numpy = importlib.import_module("numpy")
            except ModuleNotFoundError:
                logger.warning(
                    "Saved NumPy RNG state cannot be restored; NumPy is unavailable."
                )
            else:
                numpy.random.set_state(
                    (
                        numpy_state["bit_generator"],
                        numpy_state["keys"].cpu().numpy(),
                        numpy_state["position"],
                        numpy_state["has_gauss"],
                        numpy_state["cached_gaussian"],
                    )
                )

        cuda_states = state.get("cuda")
        if cuda_states is not None:
            if not torch.cuda.is_available():
                logger.warning(
                    "Saved CUDA RNG state cannot be restored; CUDA is unavailable."
                )
            else:
                device_count = torch.cuda.device_count()
                for device_index, cuda_state in enumerate(cuda_states[:device_count]):
                    torch.cuda.set_rng_state(cuda_state.cpu(), device_index)
                if len(cuda_states) != device_count:
                    logger.warning(
                        "CUDA RNG device count differs: saved=%d, available=%d",
                        len(cuda_states),
                        device_count,
                    )

        mps_state = state.get("mps")
        if mps_state is not None:
            if not torch.backends.mps.is_available():
                logger.warning(
                    "Saved MPS RNG state cannot be restored; MPS is unavailable."
                )
            else:
                torch.mps.set_rng_state(mps_state.cpu())

        torch.set_rng_state(state["torch_cpu"].cpu())

    @staticmethod
    def _validate_global_rng_state(state: object) -> dict[str, Any]:
        """Validate schema-three global RNG state without changing live RNGs."""
        if not isinstance(state, dict):
            msg = "Invalid RNG state in training checkpoint."
            raise TypeError(msg)
        if "python" not in state or not isinstance(
            state.get("torch_cpu"), torch.Tensor
        ):
            msg = "Invalid RNG state in training checkpoint."
            raise TypeError(msg)
        try:
            random.Random().setstate(state["python"])
        except (TypeError, ValueError) as error:
            msg = "Invalid Python RNG state in training checkpoint."
            raise TypeError(msg) from error
        cuda_states = state.get("cuda")
        if cuda_states is not None and (
            not isinstance(cuda_states, list)
            or any(not isinstance(item, torch.Tensor) for item in cuda_states)
        ):
            msg = "Invalid CUDA RNG state in training checkpoint."
            raise TypeError(msg)
        if state.get("mps") is not None and not isinstance(state["mps"], torch.Tensor):
            msg = "Invalid MPS RNG state in training checkpoint."
            raise TypeError(msg)
        numpy_state = state.get("numpy")
        if numpy_state is not None and (
            not isinstance(numpy_state, dict)
            or not isinstance(numpy_state.get("bit_generator"), str)
            or not isinstance(numpy_state.get("keys"), torch.Tensor)
            or not isinstance(numpy_state.get("position"), int)
            or not isinstance(numpy_state.get("has_gauss"), int)
            or not isinstance(numpy_state.get("cached_gaussian"), float)
        ):
            msg = "Invalid NumPy RNG state in training checkpoint."
            raise TypeError(msg)
        return state

    @staticmethod
    def _validate_loader_generator_states(
        states: object,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Validate retained DataLoader generator states."""
        if not isinstance(states, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, dict)
            or any(
                not isinstance(path, str) or not isinstance(state, torch.Tensor)
                for path, state in value.items()
            )
            for key, value in states.items()
        ):
            msg = "Invalid loader generator state in training checkpoint."
            raise TypeError(msg)
        return cast("dict[str, dict[str, torch.Tensor]]", states)

    def _validate_checkpoint_components(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[Callback], dict[str, Any] | None]:
        """Validate component identities and payload structure before mutation."""
        data_pack_state = self._validate_checkpoint_data_pack(payload)
        saved_optimizer = payload["optimizer"]
        if saved_optimizer is not None and self._optimizer is None:
            logger.error("Checkpoint contains optimizer state but Battery does not.")
            msg = "An optimizer is required to resume this checkpoint."
            raise ValueError(msg)
        if saved_optimizer is None and self._optimizer is not None:
            logger.error("Checkpoint optimizer availability does not match Battery.")
            msg = "Configured optimizer does not match checkpoint state."
            raise ValueError(msg)
        if saved_optimizer is not None and not isinstance(saved_optimizer, dict):
            msg = "Invalid optimizer state in training checkpoint."
            raise TypeError(msg)
        if saved_optimizer is not None and self._optimizer is not None:
            saved_groups = saved_optimizer.get("param_groups")
            current_groups = self._optimizer.state_dict().get("param_groups")
            if (
                not isinstance(saved_groups, list)
                or not isinstance(current_groups, list)
                or len(saved_groups) != len(current_groups)
                or any(
                    not isinstance(saved, dict)
                    or not isinstance(current, dict)
                    or not isinstance(saved.get("params"), list)
                    or not isinstance(current.get("params"), list)
                    or len(saved["params"]) != len(current["params"])
                    for saved, current in zip(saved_groups, current_groups, strict=True)
                )
            ):
                logger.error("Checkpoint optimizer parameter groups are incompatible.")
                msg = "Configured optimizer does not match checkpoint state."
                raise ValueError(msg)

        saved_callbacks = payload["callbacks"]
        callbacks = self._checkpoint_callbacks()
        if not isinstance(saved_callbacks, list):
            logger.error("Checkpoint callback state is not a list.")
            msg = "Invalid callback state in training checkpoint."
            raise TypeError(msg)
        expected_ids = [self._callback_identifier(item) for item in callbacks]
        actual_ids = [
            item.get("type") if isinstance(item, dict) else None
            for item in saved_callbacks
        ]
        if actual_ids != expected_ids:
            logger.error(
                "Callback state mismatch: expected=%s, actual=%s",
                expected_ids,
                actual_ids,
            )
            msg = "Configured callbacks do not match checkpoint state."
            raise ValueError(msg)
        if any(
            not isinstance(item, dict) or not isinstance(item.get("state"), dict)
            for item in saved_callbacks
        ):
            msg = "Invalid callback state in training checkpoint."
            raise TypeError(msg)
        for callback, saved in zip(callbacks, saved_callbacks, strict=True):
            callback._validate_checkpoint_state(saved["state"])  # noqa: SLF001

        self._validate_metric_states(payload["metrics"])

        if not isinstance(payload["model"], dict):
            msg = "Invalid model state in training checkpoint."
            raise TypeError(msg)
        if not isinstance(payload["epoch"], int) or not isinstance(
            payload["optimizer_step_idx"], int
        ):
            msg = "Invalid training counters in checkpoint."
            raise TypeError(msg)
        if not isinstance(payload["results"], dict):
            logger.error("Checkpoint training results are not a dictionary.")
            msg = "Invalid training history in checkpoint."
            raise TypeError(msg)

        self._validate_global_rng_state(payload["rng_state"])
        self._validate_loader_generator_states(payload["loader_generator_states"])
        return callbacks, data_pack_state

    def _checkpoint_snapshot(self, callbacks: list[Callback]) -> dict[str, Any]:
        """Capture every mutable checkpoint participant before restoration."""
        rng_state = self._capture_global_rng_state()
        callback_states = [
            copy.deepcopy(callback.state_dict()) for callback in callbacks
        ]
        if any(not isinstance(state, dict) for state in callback_states):
            msg = "Callback state_dict() must return a dictionary."
            raise TypeError(msg)
        data_pack_state: dict[str, Any] | None = None
        if self._data_pack is not None:
            raw_data_pack_state: object = self._data_pack.state_dict()
            if not isinstance(raw_data_pack_state, dict):
                msg = "DataPack state_dict() must return a dictionary."
                raise TypeError(msg)
            data_pack_state = copy.deepcopy(raw_data_pack_state)
        return {
            "model": copy.deepcopy(self._model.state_dict()),
            "optimizer": (
                copy.deepcopy(self._optimizer.state_dict())
                if self._optimizer is not None
                else None
            ),
            "callbacks": callback_states,
            "metrics": copy.deepcopy(self._checkpoint_metric_states()),
            "data_pack": data_pack_state,
            "last_completed_epoch": self._last_completed_epoch,
            "optimizer_step_idx": self._optimizer_step_idx,
            "results": copy.deepcopy(self._train_results),
            "loader_generator_states": copy.deepcopy(self._loader_generator_states),
            "pending_loader_generator_states": copy.deepcopy(
                self._pending_loader_generator_states
            ),
            "resume_loaded": self._resume_loaded,
            "stop_training": self._stop_training,
            "rng_state": rng_state,
        }

    def _apply_checkpoint_internal_state(
        self,
        payload: dict[str, Any],
    ) -> None:
        """Apply Battery-owned counters, history, and deferred loader state."""
        self._last_completed_epoch = payload["epoch"]
        self._optimizer_step_idx = payload["optimizer_step_idx"]
        self._train_results = cast("FitResult", copy.deepcopy(payload["results"]))
        loader_states = self._validate_loader_generator_states(
            payload["loader_generator_states"]
        )
        self._loader_generator_states = copy.deepcopy(loader_states)
        self._pending_loader_generator_states = copy.deepcopy(loader_states)
        self._resume_loaded = True

    @staticmethod
    def _rollback_action(name: str, action: Callable[[], None]) -> None:
        """Attempt one rollback action without masking the load failure."""
        try:
            action()
        except BaseException:
            logger.exception("Checkpoint rollback failed for %s.", name)

    def _rollback_checkpoint(
        self,
        snapshot: dict[str, Any],
        callbacks: list[Callback],
    ) -> None:
        """Restore a pre-load snapshot, attempting every component."""
        self._rollback_action(
            "model",
            lambda: self._model.load_state_dict(snapshot["model"], strict=True),
        )
        optimizer = self._optimizer
        if optimizer is not None and snapshot["optimizer"] is not None:
            self._rollback_action(
                "optimizer",
                lambda: optimizer.load_state_dict(snapshot["optimizer"]),
            )
        for index, (callback, state) in enumerate(
            zip(callbacks, snapshot["callbacks"], strict=True)
        ):

            def restore_callback(
                callback: Callback = callback,
                state: dict[str, Any] = state,
            ) -> None:
                callback.load_state_dict(state)

            self._rollback_action(
                f"callback[{index}]",
                restore_callback,
            )
        self._rollback_action(
            "metrics",
            lambda: self._restore_metric_states(snapshot["metrics"]),
        )
        data_pack = self._data_pack
        if data_pack is not None and snapshot["data_pack"] is not None:
            self._rollback_action(
                "DataPack",
                lambda: data_pack.load_state_dict(snapshot["data_pack"]),
            )

        def restore_internal_state() -> None:
            self._last_completed_epoch = snapshot["last_completed_epoch"]
            self._optimizer_step_idx = snapshot["optimizer_step_idx"]
            self._train_results = copy.deepcopy(snapshot["results"])
            self._loader_generator_states = copy.deepcopy(
                snapshot["loader_generator_states"]
            )
            self._pending_loader_generator_states = copy.deepcopy(
                snapshot["pending_loader_generator_states"]
            )
            self._resume_loaded = snapshot["resume_loaded"]
            self._stop_training = snapshot["stop_training"]

        self._rollback_action("Battery state", restore_internal_state)
        self._rollback_action(
            "global RNG",
            lambda: self._restore_global_rng_state(snapshot["rng_state"]),
        )
        logger.warning("Rolled back failed checkpoint restoration.")

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically save complete resumable training state.

        The payload contains model and optimizer state, resumable callback, metric,
        and DataPack state, the last completed epoch, optimizer-step index, and
        accumulated results. Parent directories are created automatically and the
        final path is replaced only after serialization succeeds.

        Args:
            path: Destination checkpoint path.

        Raises:
            OSError: If the destination cannot be created or replaced.
            Exception: Propagates serialization errors raised by :func:`torch.save`.

        Warning:
            PyTorch checkpoints should only be loaded from trusted sources.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks = self._checkpoint_callbacks()
        payload: dict[str, Any] = {
            "__torch_batteries_checkpoint__": _CHECKPOINT_SCHEMA_VERSION,
            "model": self._model.state_dict(),
            "optimizer": (
                self._optimizer.state_dict() if self._optimizer is not None else None
            ),
            "callbacks": [
                {
                    "type": self._callback_identifier(callback),
                    "state": callback.state_dict(),
                }
                for callback in callbacks
            ],
            "metrics": self._checkpoint_metric_states(),
            "epoch": self._last_completed_epoch,
            "optimizer_step_idx": self._optimizer_step_idx,
            "results": copy.deepcopy(self._train_results),
            "data_pack": self._checkpoint_data_pack(),
            "rng_state": self._capture_global_rng_state(),
            "loader_generator_states": copy.deepcopy(self._loader_generator_states),
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=checkpoint_path.parent,
                prefix=f".{checkpoint_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
            torch.save(payload, temporary_name)
            Path(temporary_name).replace(checkpoint_path)
        except Exception:
            logger.exception("Failed to save checkpoint at %s.", checkpoint_path)
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise
        logger.info(
            "Training checkpoint saved: path=%s, epoch=%d, optimizer_step=%d",
            checkpoint_path,
            self._last_completed_epoch,
            self._optimizer_step_idx,
        )

    @staticmethod
    def _is_raw_model_state(payload: object) -> bool:
        """Return whether a mapping resembles a raw model state dictionary."""
        return (
            isinstance(payload, dict)
            and bool(payload)
            and all(isinstance(key, str) for key in payload)
            and all(isinstance(value, torch.Tensor) for value in payload.values())
        )

    @staticmethod
    def _validate_checkpoint_schema(
        payload: object, checkpoint_path: Path
    ) -> dict[str, Any]:
        """Validate and narrow a full-checkpoint payload."""
        if not isinstance(payload, dict):
            logger.error("Checkpoint at %s is not a mapping.", checkpoint_path)
            msg = "Torch-batteries checkpoint structure must be a mapping."
            raise TypeError(msg)

        schema_version = payload.get("__torch_batteries_checkpoint__")
        if schema_version is None:
            logger.error("Unrecognized checkpoint structure at %s.", checkpoint_path)
            msg = "Unrecognized torch-batteries checkpoint structure."
            raise ValueError(msg)
        if schema_version not in _SUPPORTED_CHECKPOINT_SCHEMAS:
            logger.error(
                "Unsupported checkpoint schema %r at %s; supported schemas are %s.",
                schema_version,
                checkpoint_path,
                sorted(_SUPPORTED_CHECKPOINT_SCHEMAS),
            )
            msg = (
                f"Checkpoint schema {schema_version!r} is unsupported; "
                f"supported schemas are {sorted(_SUPPORTED_CHECKPOINT_SCHEMAS)}."
            )
            raise ValueError(msg)
        return payload

    @staticmethod
    def _move_optimizer_state(value: Any, device: torch.device) -> Any:
        """Move nested optimizer state through the shared device utility."""
        return move_to_device(value, device)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load full training state or auto-detected raw model weights.

        Full checkpoints are restored strictly: the model, optimizer availability,
        ordered resumable callbacks, stateful metrics, and saved DataPack type must
        match the current ``Battery`` configuration. DataPack state is restored before
        a later implicit setup, and optimizer tensors are moved to this battery's
        device. A raw model ``state_dict`` is accepted as weights-only input but does
        not mark training as resumable.

        Args:
            path: Full checkpoint or raw model-state path.

        Raises:
            ValueError: If the schema or configured callback/metric state differs.
            TypeError: If the serialized payload has an invalid structure.
            RuntimeError: If strict model or optimizer restoration fails.

        Warning:
            Load only checkpoints from trusted sources.
        """
        checkpoint_path = Path(path)
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=self._device,
                weights_only=True,
            )
        except Exception:
            logger.exception("Failed to read checkpoint at %s.", checkpoint_path)
            raise

        if self._is_raw_model_state(payload):
            logger.warning(
                "Raw model state detected at %s; training state was not restored.",
                checkpoint_path,
            )
            model_snapshot = copy.deepcopy(self._model.state_dict())
            resume_snapshot = self._resume_loaded
            try:
                self._model.load_state_dict(payload, strict=True)
                self._resume_loaded = False
            except BaseException:
                logger.exception(
                    "Raw model checkpoint restoration failed at %s; rolling back.",
                    checkpoint_path,
                )
                self._rollback_action(
                    "model",
                    lambda: self._model.load_state_dict(model_snapshot, strict=True),
                )
                self._resume_loaded = resume_snapshot
                raise
            return

        payload = self._validate_checkpoint_schema(payload, checkpoint_path)
        required = {
            "model",
            "optimizer",
            "callbacks",
            "metrics",
            "epoch",
            "optimizer_step_idx",
            "results",
            "rng_state",
            "loader_generator_states",
            "data_pack",
        }
        if not required.issubset(payload):
            missing = sorted(required - set(payload))
            logger.error("Checkpoint is missing required fields: %s", missing)
            msg = f"Training checkpoint is missing fields: {missing}."
            raise ValueError(msg)
        callbacks, data_pack_state = self._validate_checkpoint_components(payload)
        snapshot = self._checkpoint_snapshot(callbacks)
        try:
            self._model.load_state_dict(payload["model"], strict=True)
            if self._optimizer is not None:
                self._optimizer.load_state_dict(payload["optimizer"])
                self._optimizer.state = self._move_optimizer_state(
                    self._optimizer.state, self._device
                )
            for callback, saved in zip(callbacks, payload["callbacks"], strict=True):
                callback.load_state_dict(saved["state"])
            self._restore_metric_states(payload["metrics"])
            if self._data_pack is not None and data_pack_state is not None:
                self._data_pack.load_state_dict(data_pack_state)
            self._apply_checkpoint_internal_state(payload)
            self._restore_global_rng_state(payload["rng_state"])
        except BaseException:
            logger.exception(
                "Checkpoint restoration failed at %s; rolling back.",
                checkpoint_path,
            )
            self._rollback_checkpoint(snapshot, callbacks)
            raise
        logger.info(
            "Training checkpoint restored: path=%s, epoch=%d, optimizer_step=%d",
            checkpoint_path,
            self._last_completed_epoch,
            int(payload["optimizer_step_idx"]),
        )
