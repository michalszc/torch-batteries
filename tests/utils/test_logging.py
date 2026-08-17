"""Tests for torch_batteries.utils.logging module."""

import logging
from collections.abc import Callable, Generator
from importlib import import_module
from unittest.mock import MagicMock, patch

import pytest

from torch_batteries.utils import logging as package_logging
from torch_batteries.utils.logging import (
    disable_default_handler,
    enable_default_handler,
    enable_explicit_format,
    get_logger,
    reset_format,
    set_verbosity,
    set_verbosity_debug,
    set_verbosity_error,
    set_verbosity_info,
    set_verbosity_warning,
)

LOGGER_MODULES = (
    "callbacks._monitor",
    "callbacks.base",
    "callbacks.early_stopping",
    "callbacks.experiment_tracking",
    "callbacks.gradient_accumulation",
    "callbacks.gradient_clip",
    "callbacks.learning_rate_scheduler",
    "callbacks.mixed_precision",
    "callbacks.model_checkpoint",
    "data.base",
    "data.handler",
    "data.loader",
    "events.core",
    "events.handler",
    "tracking.wandb",
    "trainer._checkpoint",
    "trainer._evaluation",
    "trainer._prediction",
    "trainer._training",
    "trainer.core",
    "utils.device",
    "utils.metrics",
    "utils.prediction",
    "utils.progress.factory",
)


@pytest.fixture(autouse=True)
def restore_package_logger() -> Generator[None]:
    """Keep the module-level logging manager isolated between tests."""
    manager = package_logging._manager  # noqa: SLF001
    root_logger = manager.get_root_logger()
    original_handler = manager.default_handler
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    yield

    manager.reset_default_handler()
    root_logger.handlers[:] = original_handlers
    root_logger.setLevel(original_level)
    manager._default_handler = original_handler  # noqa: SLF001


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

    @pytest.mark.parametrize("module_name", LOGGER_MODULES)
    def test_library_logger_uses_module_qualified_name(self, module_name: str) -> None:
        """Every library logger follows its lowercase module path."""
        module = import_module(f"torch_batteries.{module_name}")
        logger = module.__dict__["logger"]

        assert isinstance(logger, logging.Logger)
        assert logger.name == f"torch_batteries.{module_name}"

    @patch("torch_batteries.utils.logging._manager.logging.getLogger")
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


class TestLoggerConfiguration:
    """Tests for package-level logger configuration."""

    @pytest.mark.parametrize(
        ("configured_level", "expected"),
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
        ],
    )
    def test_environment_configures_default_level(
        self,
        monkeypatch: pytest.MonkeyPatch,
        configured_level: str,
        expected: int,
    ) -> None:
        manager = package_logging._manager  # noqa: SLF001
        manager.reset_default_handler()
        monkeypatch.setenv("TORCH_BATTERIES_LOG_LEVEL", configured_level)

        logger = get_logger()

        assert logger.level == expected
        assert manager.default_handler in logger.handlers

    def test_missing_environment_uses_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = package_logging._manager  # noqa: SLF001
        manager.reset_default_handler()
        monkeypatch.delenv("TORCH_BATTERIES_LOG_LEVEL", raising=False)

        assert get_logger().level == logging.WARNING

    def test_invalid_environment_level_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = package_logging._manager  # noqa: SLF001
        manager.reset_default_handler()
        monkeypatch.setenv("TORCH_BATTERIES_LOG_LEVEL", "TRACE")

        with pytest.raises(ValueError, match="Invalid log level: 'TRACE'"):
            get_logger()

        assert manager.default_handler is None

    def test_setup_is_idempotent(self) -> None:
        manager = package_logging._manager  # noqa: SLF001
        manager.reset_default_handler()

        logger = get_logger()
        handler = manager.default_handler
        get_logger()

        assert handler is not None
        assert logger.handlers.count(handler) == 1

    def test_reset_before_setup_is_a_noop(self) -> None:
        manager = package_logging._manager  # noqa: SLF001
        manager.reset_default_handler()

        manager.reset_default_handler()

        assert manager.default_handler is None

    def test_reset_removes_handler_and_level(self) -> None:
        manager = package_logging._manager  # noqa: SLF001
        manager.reset_default_handler()
        logger = get_logger()
        handler = manager.default_handler

        manager.reset_default_handler()

        assert handler not in logger.handlers
        assert logger.level == logging.NOTSET
        assert manager.default_handler is None

    @pytest.mark.parametrize(
        ("configure", "expected"),
        [
            (set_verbosity_debug, logging.DEBUG),
            (set_verbosity_info, logging.INFO),
            (set_verbosity_warning, logging.WARNING),
            (set_verbosity_error, logging.ERROR),
        ],
    )
    def test_verbosity_helpers(
        self, configure: Callable[[], None], expected: int
    ) -> None:
        get_logger()

        configure()

        assert package_logging._manager.get_root_logger().level == expected  # noqa: SLF001

    def test_set_verbosity_uses_requested_level(self) -> None:
        get_logger()

        set_verbosity(logging.CRITICAL)

        manager = package_logging._manager  # noqa: SLF001
        assert manager.get_root_logger().level == logging.CRITICAL

    def test_default_handler_can_be_disabled_and_enabled(self) -> None:
        logger = get_logger()
        handler = package_logging._manager.default_handler  # noqa: SLF001
        assert handler is not None

        disable_default_handler()
        assert handler not in logger.handlers

        enable_default_handler()
        enable_default_handler()
        assert logger.handlers.count(handler) == 1

    def test_explicit_format_can_be_reset(self) -> None:
        get_logger()
        handler = package_logging._manager.default_handler  # noqa: SLF001
        assert handler is not None
        record = logging.LogRecord(
            "torch_batteries",
            logging.INFO,
            "module.py",
            12,
            "message",
            (),
            None,
        )

        enable_explicit_format()
        assert "module.py:12" in handler.format(record)

        reset_format()
        assert handler.format(record) == "[torch_batteries] INFO: message"
