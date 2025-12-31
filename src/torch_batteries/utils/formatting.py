"""Utilities for formatting metrics and other display strings."""


def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    """Format metrics dictionary into a human-readable string.

    Converts metric names from snake_case to Title Case and formats values
    to 4 decimal places. Multiple metrics are joined with commas.

    Args:
        metrics: Dictionary mapping metric names to their float values.
        prefix: Optional prefix to add before each metric name.

    Returns:
        Formatted string of metrics. Returns empty string if metrics is empty.

    Example:
        ```python
        metrics = {'loss': 0.1234, 'mae': 0.5678, 'train_rmse': 0.9012}

        # Without prefix
        format_metrics(metrics)
        # Returns: "Loss: 0.1234, Mae: 0.5678, Train Rmse: 0.9012"

        # With prefix
        format_metrics(metrics, prefix="Train ")
        # Returns: "Train Loss: 0.1234, Train Mae: 0.5678, Train Train Rmse: 0.9012"
        ```

    Note:
        - Metric names are converted from snake_case to Title Case
        - Values are formatted to 4 decimal places
        - Empty metrics dictionary returns empty string
    """
    parts = []
    for key, value in metrics.items():
        metric_name = key.replace("_", " ").title()
        parts.append(f"{prefix}{metric_name}: {value:.4f}")
    return ", ".join(parts)
