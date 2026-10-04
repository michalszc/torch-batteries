"""Internal types for metric definitions and structured inputs."""

import torch

from .metric_spec import MetricSpec
from .metric_types import Metric

type MetricDefinition = Metric | MetricSpec
type StructuredTensors = torch.Tensor | dict[str, torch.Tensor]
type Selection = tuple[str | None, str | None]
