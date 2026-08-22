"""Public metric aliases."""

from collections.abc import Callable

import torch

from .state import StatefulMetric

type MetricCallable = Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]
type Metric = MetricCallable | StatefulMetric
