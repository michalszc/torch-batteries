"""Progress metric mapping contract."""

from typing import NotRequired, TypedDict


class ProgressMetrics(TypedDict, total=False):
    """Scalar metrics accumulated by a progress implementation.

    Attributes:
        loss: Optional scalar phase loss.
    """

    loss: NotRequired[float]
