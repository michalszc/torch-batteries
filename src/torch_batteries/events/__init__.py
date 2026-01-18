"""Events package for torch-batteries.

Provides event system for training workflow control:
- **Event**: Enumeration of all available training lifecycle events
- **EventContext**: Context dictionary passed to event handlers
- **charge**: Decorator to mark methods as event handlers
- **EventHandler**: Internal handler for discovering and executing event methods
"""

from .core import Event, EventContext, charge
from .handler import EventHandler

__all__ = ["Event", "EventContext", "EventHandler", "charge"]
