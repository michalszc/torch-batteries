"""Per-dataset sample-weighted metric aggregation."""


class DatasetMetricTotals:
    """Accumulate sample-weighted scalar metrics for each named dataset."""

    def __init__(self) -> None:
        self._totals: dict[str, dict[str, float]] = {}
        self._weights: dict[str, dict[str, int]] = {}

    def update(self, name: str, metrics: dict[str, float], samples: int) -> None:
        """Record one batch's metrics for a dataset.

        Args:
            name: Dataset name.
            metrics: Scalar batch metrics.
            samples: Number of samples contributing to the batch.
        """
        totals = self._totals.setdefault(name, {})
        weights = self._weights.setdefault(name, {})
        for metric, value in metrics.items():
            totals[metric] = totals.get(metric, 0.0) + value * samples
            weights[metric] = weights.get(metric, 0) + samples

    def compute(self) -> dict[str, float]:
        """Return ``dataset:metric`` weighted averages."""
        return {
            f"{name}:{metric}": value / self._weights[name][metric]
            for name, totals in self._totals.items()
            for metric, value in totals.items()
        }
