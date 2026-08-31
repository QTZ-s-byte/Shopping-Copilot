"""Canonical intent and session types used by Member A and the core agent."""

from dataclasses import dataclass

from shopping_copilot.contracts import (
    ALLOWED_ATTRIBUTES as ASK_ATTRIBUTES,
    ALLOWED_INTENTS as VALID_INTENTS,
    IntentResult,
    SessionState as _CanonicalSessionState,
)

INTENT_BUYING = "buying"
INTENT_BROWSING = "browsing"
HARD_ATTRIBUTES = ("category", "brand", "color", "size", "material", "style", "budget")
SOFT_ATTRIBUTES = ("feature", "use_case", "preference_tags")
VALID_ASK_ATTRIBUTES = tuple(ASK_ATTRIBUTES) + (None,)


@dataclass
class SessionState(_CanonicalSessionState):
    """A test-friendly A-facing view with browsing as the initial intent."""

    intent: str | None = INTENT_BROWSING

__all__ = [
    "ASK_ATTRIBUTES",
    "VALID_ASK_ATTRIBUTES",
    "VALID_INTENTS",
    "INTENT_BUYING",
    "INTENT_BROWSING",
    "HARD_ATTRIBUTES",
    "SOFT_ATTRIBUTES",
    "IntentResult",
    "SessionState",
]
