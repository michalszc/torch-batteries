"""Test workflow result contract."""

from typing import TypedDict


class TestResult(TypedDict, total=False):
    """Result from testing.

    Attributes:
        test_loss: Average test loss across all batches.
        test_metrics: Named average test metrics.
    """

    test_loss: float
    test_metrics: dict[str, float]
