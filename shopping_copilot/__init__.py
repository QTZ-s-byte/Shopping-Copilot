"""Core orchestration components for the TechJam Shopping Copilot.

The package deliberately keeps the Agent, retriever, and ranker behind small
protocols.  Teammates can replace the default implementations without having
to rewrite the lifecycle, memory, or evaluation plumbing.
"""

from .contracts import (
    AgentResponse,
    Candidate,
    IntentResult,
    RetrievalResult,
    SessionState,
    StateDiff,
)
from .memory import InMemoryContextMemory
from .orchestrator import ShoppingOrchestrator

__all__ = [
    "AgentResponse",
    "Candidate",
    "IntentResult",
    "RetrievalResult",
    "SessionState",
    "StateDiff",
    "InMemoryContextMemory",
    "ShoppingOrchestrator",
]
