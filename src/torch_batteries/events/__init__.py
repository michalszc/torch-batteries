"""Lifecycle and optimization events used by the trainer and callbacks.

## Public API

- **`Event`** — names every lifecycle, step, and optimization extension point.
- **`EventContext`** — typed mapping of values available to event handlers.
- **`OptimizationStep`** — immutable gradient-operation plan for one train batch.
- **`charge`** — marks a model or callback method as an event handler.
- **`EventHandler`** — discovers handlers and applies broadcast, provider, executor,
  and context-manager dispatch rules.
"""

from .core import Event, EventContext, OptimizationStep, charge
from .handler import EventHandler

__all__ = ["Event", "EventContext", "EventHandler", "OptimizationStep", "charge"]
