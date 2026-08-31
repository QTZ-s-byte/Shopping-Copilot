"""Structured clarification questions mapped to the official ``ask_attribute``."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from agent.types import (
    INTENT_BROWSING,
    IntentResult,
    SessionState,
)


@dataclass
class Clarification:
    needed: bool
    message: str = ""
    ask_attribute: Optional[str] = None
    options: List[str] = field(default_factory=list)
    reason: str = ""


BUYING_PRIORITY = ("category", "budget", "brand", "material", "color", "size", "style")
BROWSING_PRIORITY = ("style", "feature", "use_case", "category")

ATTRIBUTE_PROMPTS = {
    "category": "Which category are you interested in?",
    "budget": "What is your budget?",
    "brand": "Do you have a brand preference?",
    "material": "Do you have a material preference?",
    "color": "Do you have a color preference?",
    "size": "What size are you looking for?",
    "style": "What style do you prefer?",
    "feature": "Are there any features you care about?",
    "use_case": "What will you use it for?",
}


class ClarificationPolicy:
    def __init__(
        self,
        max_candidates: int = 100,
        category_options: Optional[List[str]] = None,
    ) -> None:
        self.max_candidates = max_candidates
        self.category_options = category_options or [
            "Shoes",
            "Clothing",
            "Jewelry",
            "Bags",
            "Accessories",
        ]

    def suggest(
        self,
        state: SessionState,
        intent: str,
        hard: dict,
        soft: dict,
        negative: dict,
    ) -> Optional[str]:
        """Return a conservative clarification attribute, or None.

        Unlike :meth:`decide`, this is meant to be embedded in Member A's
        intent result. It only asks when the request is genuinely vague or
        self-conflicting, so a concrete request still lets the orchestrator
        return ranked results immediately.
        """
        for key in hard:
            if key in negative:
                return key

        if intent == INTENT_BROWSING:
            if not soft.get("use_case") and not soft.get("feature") and not hard.get("style"):
                return "style"
            return None

        if not hard.get("category") and not hard.get("brand"):
            return "category"
        return None

    def decide(
        self,
        state: SessionState,
        intent_result: IntentResult,
        candidate_count: Optional[int] = None,
    ) -> Clarification:
        conflict = self._conflicting_field(state, intent_result)
        if conflict:
            return Clarification(
                needed=True,
                message=(
                    f"You asked both for and against '{conflict}'. "
                    "Which one should I prioritize?"
                ),
                ask_attribute=conflict,
                reason="conflicting_constraints",
            )

        if candidate_count is not None and candidate_count > self.max_candidates:
            return self._ask_category("candidate_set_too_broad")

        next_attr = self._next_missing_attribute(state, intent_result)
        if next_attr is None:
            return Clarification(needed=False)

        if next_attr == "category":
            return self._ask_category("missing_attribute")
        return Clarification(
            needed=True,
            message=ATTRIBUTE_PROMPTS[next_attr],
            ask_attribute=next_attr,
            reason="missing_attribute",
        )

    def _ask_category(self, reason: str) -> Clarification:
        numbered = "\n".join(
            f"{index}. {option}"
            for index, option in enumerate(self.category_options, start=1)
        )
        return Clarification(
            needed=True,
            message=(
                "The candidate set is too broad. Would you prefer:\n" + numbered
            ),
            ask_attribute="category",
            options=list(self.category_options),
            reason=reason,
        )

    def _next_missing_attribute(
        self, state: SessionState, intent_result: IntentResult
    ) -> Optional[str]:
        priority = (
            BUYING_PRIORITY if intent_result.intent != INTENT_BROWSING else BROWSING_PRIORITY
        )
        for attr in priority:
            if attr in intent_result.hard_constraints or attr in intent_result.soft_preferences:
                continue
            if attr in state.hard_constraints or attr in state.soft_preferences:
                continue
            if attr in state.negative_constraints:
                continue
            if attr in state.asked_attributes or attr in state.no_preference_attributes:
                continue
            return attr
        return None

    def _conflicting_field(
        self, state: SessionState, intent_result: IntentResult
    ) -> Optional[str]:
        positive = intent_result.hard_constraints
        negative = intent_result.negative_constraints
        for key in positive:
            if key in negative:
                return key
        return None
