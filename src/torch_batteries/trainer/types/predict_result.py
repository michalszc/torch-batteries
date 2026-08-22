"""Prediction workflow result contract."""

from typing import Any, TypedDict


class PredictResult(TypedDict):
    """Collected prediction output.

    Attributes:
        predictions: Batch list or recursively concatenated prediction structure.
    """

    predictions: Any
