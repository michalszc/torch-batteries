"""Tests for torch_batteries.utils.logging module."""

import logging
from unittest.mock import MagicMock, patch

from torch_batteries.utils.logging import get_logger


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_with_name(self) -> None:
        """Test getting a logger with a specific name."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "torch_batteries.test_logger"

    def test_get_logger_without_name(self) -> None:
        """Test getting a logger without a name."""
        logger = get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "torch_batteries"

    def test_get_logger_empty_name(self) -> None:
        """Test getting a logger with empty name."""
        logger = get_logger("")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "torch_batteries."

    def test_get_logger_nested_name(self) -> None:
        """Test getting a logger with nested module name."""
        logger = get_logger("module.submodule")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "torch_batteries.module.submodule"

    def test_get_logger_same_name_returns_same_instance(self) -> None:
        """Test that getting logger with same name returns same instance."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        assert logger1 is logger2

    def test_logger_hierarchy(self) -> None:
        """Test logger hierarchy relationships."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        assert child_logger.parent is not None
        assert child_logger.parent.name == parent_logger.name
        """Test logger has appropriate default level."""
        logger = get_logger("level_test")
        # Logger should inherit from root logger or have INFO level
        assert logger.level in [0, logging.INFO, logging.WARNING]  # 0 means inherit

    @patch("torch_batteries.utils.logging.logging.getLogger")
    def test_get_logger_calls_logging_module(self, mock_get_logger: MagicMock) -> None:
        """Test that get_logger properly calls logging.getLogger."""
        mock_root_logger = MagicMock()
        mock_child_logger = MagicMock()
        mock_root_logger.getChild.return_value = mock_child_logger
        mock_get_logger.return_value = mock_root_logger

        result = get_logger("test")

        # Should call getLogger for the root package
        mock_get_logger.assert_called_once_with("torch_batteries")
        # And then create a child logger
        mock_root_logger.getChild.assert_called_once_with("test")
        assert result == mock_child_logger
