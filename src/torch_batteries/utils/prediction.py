"""Utilities for combining structured prediction outputs."""

from typing import Any

import torch

from torch_batteries.utils.logging import get_logger

logger = get_logger("prediction")


def concatenate_predictions(outputs: list[Any]) -> Any:
    """Recursively concatenate matching tensor leaves across prediction batches."""
    if not outputs:
        logger.error("Cannot concatenate an empty prediction result.")
        msg = "Cannot concatenate predictions because no outputs were returned."
        raise ValueError(msg)
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(item, torch.Tensor) for item in outputs):
            logger.error("Prediction tensor structures differ across batches.")
            msg = "Prediction structures differ across batches."
            raise TypeError(msg)
        try:
            result = torch.cat(outputs)
        except RuntimeError as error:
            logger.exception("Prediction tensor shapes cannot be concatenated.")
            msg = "Prediction tensor shapes are incompatible for concatenation."
            raise ValueError(msg) from error
        logger.debug(
            "Prediction tensors concatenated: batches=%d, shape=%s",
            len(outputs),
            tuple(result.shape),
        )
        return result

    if isinstance(first, dict):
        keys = list(first)
        if not all(
            isinstance(item, dict) and set(item) == set(keys) for item in outputs
        ):
            logger.error("Prediction dictionary structures differ across batches.")
            msg = "Prediction dictionary structures differ across batches."
            raise ValueError(msg)
        return {
            key: concatenate_predictions([item[key] for item in outputs])
            for key in keys
        }

    if isinstance(first, tuple):
        if not all(
            isinstance(item, tuple) and len(item) == len(first) for item in outputs
        ):
            logger.error("Prediction tuple structures differ across batches.")
            msg = "Prediction tuple structures differ across batches."
            raise ValueError(msg)
        items = [
            concatenate_predictions([output[index] for output in outputs])
            for index in range(len(first))
        ]
        if hasattr(first, "_fields"):
            return type(first)(*items)
        return tuple(items)

    if isinstance(first, list):
        if not all(
            isinstance(item, list) and len(item) == len(first) for item in outputs
        ):
            logger.error("Prediction list structures differ across batches.")
            msg = "Prediction list structures differ across batches."
            raise ValueError(msg)
        return [
            concatenate_predictions([output[index] for output in outputs])
            for index in range(len(first))
        ]

    logger.error(
        "Unsupported prediction leaf for concatenation: %s", type(first).__name__
    )
    msg = (
        "Prediction concatenation supports tensors and matching nested "
        "dictionaries, tuples, and lists."
    )
    raise TypeError(msg)
