"""Shared data structures aligned with ``docs/agent_api_contract.json``.

Members A, B and C import these types so the whole agent speaks the same schema
as the official evaluator.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


DEFAULT_MAX_TURNS = 10

INTENT_BUYING = "buying"
INTENT_BROWSING = "browsing"
VALID_INTENTS = (INTENT_BUYING, INTENT_BROWSING)

# The exact ``ask_attribute`` enum accepted by the evaluator (``null`` handled
# separately as ``None``).
ASK_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
VALID_ASK_ATTRIBUTES = ASK_ATTRIBUTES + (None,)

# Attributes treated as hard constraints during Buying routing.
HARD_ATTRIBUTES = ("category", "brand", "color", "size", "material", "style", "budget")
# Attributes treated as soft preferences.
SOFT_ATTRIBUTES = ("feature", "use_case", "preference_tags")


@dataclass
class IntentResult:
    """What one user turn means.

    ``intent`` is ``buying`` or ``browsing``; ``override`` and ``no_preference``
    are independent flags. A turn can be buying *and* override the previous goal
    (for example "actually, forget shoes, I want a bag").
    """

    intent: str
    confidence: float
    hard_constraints: Dict[str, Any] = field(default_factory=dict)
    soft_preferences: Dict[str, Any] = field(default_factory=dict)
    negative_constraints: Dict[str, Any] = field(default_factory=dict)
    override: bool = False
    override_reason: str = ""
    no_preference: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "hard_constraints": self.hard_constraints,
            "soft_preferences": self.soft_preferences,
            "negative_constraints": self.negative_constraints,
            "override": self.override,
            "override_reason": self.override_reason,
            "no_preference": self.no_preference,
        }


@dataclass
class SessionState:
    """Structured shopping state carried across a session."""

    session_id: str = ""
    intent: str = INTENT_BROWSING
    hard_constraints: Dict[str, Any] = field(default_factory=dict)
    soft_preferences: Dict[str, Any] = field(default_factory=dict)
    negative_constraints: Dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    recent_messages: List[str] = field(default_factory=list)
    summary: str = ""
    user_profile: Dict[str, Any] = field(default_factory=dict)
    asked_attributes: Set[str] = field(default_factory=set)
    no_preference_attributes: Set[str] = field(default_factory=set)
    max_turns: int = DEFAULT_MAX_TURNS
    terminated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "intent": self.intent,
            "hard_constraints": self.hard_constraints,
            "soft_preferences": self.soft_preferences,
            "negative_constraints": self.negative_constraints,
            "turn_count": self.turn_count,
            "recent_messages": list(self.recent_messages),
            "summary": self.summary,
            "user_profile": dict(self.user_profile),
            "asked_attributes": sorted(self.asked_attributes),
            "no_preference_attributes": sorted(self.no_preference_attributes),
            "max_turns": self.max_turns,
            "terminated": self.terminated,
        }
