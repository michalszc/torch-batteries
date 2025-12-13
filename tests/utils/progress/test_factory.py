"""Tests for torch_batteries.utils.progress.ProgressFactory class."""

import pytest

from torch_batteries.utils.progress import (
    BarProgress,
    ProgressFactory,
    SilentProgress,
    SimpleProgress,
)


class TestProgressFactory:
    """Test cases for ProgressFactory class (main tests)."""

    def test_creates_silent_progress(self) -> None:
        """Test ProgressFactory creates SilentProgress for verbose=0."""
        progress = ProgressFactory.create(verbose=0, total_epochs=5)
        assert isinstance(progress, SilentProgress)

    def test_creates_progress_bar(self) -> None:
        """Test ProgressFactory creates BarProgress for verbose=1."""
        progress = ProgressFactory.create(verbose=1, total_epochs=5)
        assert isinstance(progress, BarProgress)

    def test_creates_simple_progress(self) -> None:
        """Test ProgressFactory creates SimpleProgress for verbose=2."""
        progress = ProgressFactory.create(verbose=2, total_epochs=5)
        assert isinstance(progress, SimpleProgress)

    def test_invalid_verbose_raises_error(self) -> None:
        """Test ProgressFactory raises error for invalid verbose level."""
        with pytest.raises(ValueError, match="Invalid verbose level"):
            ProgressFactory.create(verbose=3, total_epochs=1)

    def test_default_values(self) -> None:
        """Test ProgressFactory uses default values."""
        progress = ProgressFactory.create(verbose=1)  # Explicitly set default
        assert isinstance(progress, BarProgress)  # verbose=1 is default


class TestProgressFactoryCreate:
    """Test cases for ProgressFactory.create() static method."""

    def test_create_silent_progress(self) -> None:
        """Test factory creates SilentProgress for verbose=0."""
        progress = ProgressFactory.create(verbose=0, total_epochs=5)
        assert isinstance(progress, SilentProgress)

    def test_create_progress_bar(self) -> None:
        """Test factory creates BarProgress for verbose=1."""
        progress = ProgressFactory.create(verbose=1, total_epochs=5)
        assert isinstance(progress, BarProgress)

    def test_create_simple_progress(self) -> None:
        """Test factory creates SimpleProgress for verbose=2."""
        progress = ProgressFactory.create(verbose=2, total_epochs=5)
        assert isinstance(progress, SimpleProgress)

    def test_create_invalid_verbose(self) -> None:
        """Test factory raises error for invalid verbose level."""
        with pytest.raises(ValueError, match="Invalid verbose level"):
            ProgressFactory.create(verbose=3, total_epochs=1)
