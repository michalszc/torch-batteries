from torch_batteries import EventContext, Event, charge, Battery
from torch_batteries.utils.logging import get_logger

import torch.nn as nn

class EarlyStopping:
    """Early stops the training if selected metric doesn't improve after a given patience."""

    def __init__(
        self,
        stage: str,
        metric: str,
        min_delta: float = 0.0,
        patience: int = 5,
        verbose: bool = False,
        mode: str = "min",
        restore_best_weights: bool = False,
    ) -> None:
        """
        Args:
            stage (str): One of 'train' or 'val' to indicate which stage's metric to monitor.
            metric (str): The name of the metric to monitor.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            verbose (bool): If True, prints a message when early stopping is triggered.
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the
                        monitored metric stops decreasing. In 'max' mode, it will stop when
                        the metric stops increasing.
        """
        if stage not in {"train", "val"}:
            raise ValueError("stage must be one of 'train' or 'val'")

        self.stage = stage
        self.metric = metric
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.logger = get_logger("EarlyStopping")

        self.best_score: float | None = None
        self.epochs_no_improve = 0

        if mode not in {"min", "max"}:
            raise ValueError("mode must be one of 'min' or 'max'")
        self.mode = mode
        if self.mode == "min":
            self.monitor_op = lambda current, best: current < best - self.min_delta
        else:
            self.monitor_op = lambda current, best: current > best + self.min_delta

    @charge(Event.BEFORE_TRAIN)
    def run_on_train_start(self, _: EventContext) -> None:
        """
        Initialize early stopping parameters at the start of training.

        Args:
            _: The event context (not used here).
        """
        self.best_score = None
        self.epochs_no_improve = 0

    @charge(Event.AFTER_TRAIN_EPOCH)
    def run_on_epoch_end(self, context: EventContext) -> None:
        if self.stage != "train":
            return

        metrics = context["train_metrics"]
        model = context["model"]
        battery = context["battery"]
        self._check_for_early_stop(metrics, model, battery)

    @charge(Event.AFTER_VALIDATION)
    def run_on_validation_end(self, context: EventContext):
        if self.stage != "val":
            return

        metrics = context["val_metrics"]
        model = context["model"]
        battery = context["battery"]
        self._check_for_early_stop(metrics, model, battery)

    def _check_for_early_stop(self, metrics: dict[str, float], model: nn.Module, battery: Battery) -> None:
        """
        Check if early stopping condition is met and update internal state.

        Args:
            metrics: Dictionary of current metrics.
            model: The model being trained.
        """

        if self.metric not in metrics:
            raise ValueError(f"Metric '{self.metric}' not found in training metrics.")

        current_score = metrics[self.metric]
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict()
            return

        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.epochs_no_improve = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict()
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                battery.stop_training = True
                if self.verbose:
                    self.logger.info(
                        f"Early stopping triggered. No improvement in '{self.metric}' for {self.patience} epochs."
                    )

    @charge(Event.AFTER_TRAIN)
    def run_on_train_end(self, context: EventContext) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            context["model"].load_state_dict(self.best_weights)
            if self.verbose:
                self.logger.info("Restored best model weights from early stopping.")
