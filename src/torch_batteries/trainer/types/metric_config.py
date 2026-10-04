"""Metric configuration accepted by Battery."""

from collections.abc import Mapping

from torch_batteries.utils.metrics.types import MetricDefinition

type MetricsConfig = (
    Mapping[str, MetricDefinition] | Mapping[str, Mapping[str, MetricDefinition]]
)

METRIC_PHASES = ("train", "validation", "test")
