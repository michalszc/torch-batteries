from torch_batteries import EventContext, Event, charge


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

        self.best_score: float | None = None
        self.epochs_no_improve = 0
        self.early_stop = False

        if mode not in {"min", "max"}:
            raise ValueError("mode must be one of 'min' or 'max'")
        self.mode = mode
        if self.mode == "min":
            self.monitor_op = lambda current, best: current < best - self.min_delta
        else:
            self.monitor_op = lambda current, best: current > best + self.min_delta

    @charge(Event.AFTER_TRAIN_EPOCH)
    def run_on_epoch_end(self, context: EventContext) -> None:
        if self.stage != "train":
            return

    @charge(Event.AFTER_VALIDATION)
    def run_on_validation_end(self, context: EventContext):
        if self.stage != "val":
            return
