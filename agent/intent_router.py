"""Rule-based intent routing with an optional external-LLM enhancement hook."""

from typing import Callable, Optional, Tuple

from agent.slot_extractor import ExtractedSlots, SlotExtractor
from agent.types import INTENT_BROWSING, INTENT_BUYING, IntentResult, SessionState


BUYING_SIGNALS = [
    ("i need", 2.0),
    ("need a", 1.5),
    ("need to", 1.0),
    ("i want to buy", 2.5),
    ("want to buy", 2.5),
    ("looking for", 2.0),
    ("find me", 2.0),
    ("get me", 1.5),
    ("i want", 1.2),
    ("buy", 2.0),
    ("purchase", 2.0),
    ("order", 1.0),
    ("budget", 1.2),
    ("specific", 1.0),
    ("exact", 1.0),
    ("requirements", 1.0),
    ("must have", 1.5),
]

BROWSING_SIGNALS = [
    ("show me", 2.0),
    ("ideas", 2.0),
    ("outfit ideas", 2.5),
    ("gift ideas", 2.5),
    ("recommend", 2.0),
    ("suggest", 2.0),
    ("explore", 2.0),
    ("browse", 2.0),
    ("browsing", 2.0),
    ("trending", 1.8),
    ("what's trending", 2.5),
    ("whats trending", 2.5),
    ("popular", 1.2),
    ("gift", 1.5),
    ("inspiration", 2.0),
    ("inspire", 2.0),
    ("something like", 2.0),
    ("options", 1.2),
    ("maybe", 1.0),
    ("just looking", 2.2),
    ("curious", 1.5),
    ("discover", 1.5),
    ("open to", 1.5),
    ("what should i", 1.5),
    ("help me pick", 1.5),
    ("help me choose", 1.5),
    ("for inspiration", 2.0),
]

OVERRIDE_SIGNALS = [
    "actually",
    "forget",
    "never mind",
    "nevermind",
    "on second thought",
    "change my mind",
    "changed my mind",
    "i'd rather",
    "id rather",
    "scratch that",
    "let's do",
    "lets do",
    "switch",
    "no, i want",
    "no i want",
    "wait, i want",
]

STRONG_OVERRIDE = (
    "never mind",
    "nevermind",
    "change my mind",
    "changed my mind",
    "i'd rather",
    "id rather",
    "scratch that",
)

NO_PREFERENCE_PHRASES = [
    "no preference",
    "no pref",
    "doesn't matter",
    "doesnt matter",
    "don't care",
    "dont care",
    "no idea",
    "any is fine",
    "anything is fine",
    "whatever",
    "up to you",
    "not sure",
    "no specific",
    "doesn't really matter",
]


class IntentRouter:
    def __init__(
        self,
        slot_extractor: Optional[SlotExtractor] = None,
        llm_classifier: Optional[Callable[[str, SessionState], IntentResult]] = None,
    ) -> None:
        self.slot_extractor = slot_extractor or SlotExtractor()
        self.llm_classifier = llm_classifier

    def classify(self, user_message: str, state: SessionState) -> IntentResult:
        if self.llm_classifier is not None:
            try:
                return self.llm_classifier(user_message, state)
            except Exception:
                pass

        slots = self.slot_extractor.extract(user_message)
        override, override_reason = self._detect_override(user_message, state, slots)
        no_preference = self._detect_no_preference(user_message)
        intent, confidence = self._detect_intent(user_message, slots)

        return IntentResult(
            intent=intent,
            confidence=confidence,
            hard_constraints=slots.to_hard_constraints(),
            soft_preferences=slots.to_soft_preferences(),
            negative_constraints=slots.to_negative_constraints(),
            override=override,
            override_reason=override_reason,
            no_preference=no_preference,
        )

    def _detect_override(
        self,
        message: str,
        state: SessionState,
        slots: ExtractedSlots,
    ) -> Tuple[bool, str]:
        text = message.lower().strip()
        hit = next((signal for signal in OVERRIDE_SIGNALS if signal in text), None)
        if hit is None:
            return False, ""

        strong = any(signal in text for signal in STRONG_OVERRIDE)
        has_new_goal = any(
            token in text for token in ("want", "need", "looking for", "get me")
        ) or bool(slots.category or slots.brand)
        if strong or has_new_goal:
            return True, hit
        return False, ""

    def _detect_no_preference(self, message: str) -> bool:
        text = message.lower().strip()
        return any(phrase in text for phrase in NO_PREFERENCE_PHRASES)

    def _detect_intent(
        self, message: str, slots: ExtractedSlots
    ) -> Tuple[str, float]:
        text = message.lower().strip()

        buying_score = 0.0
        for signal, weight in BUYING_SIGNALS:
            if signal in text:
                buying_score += weight

        browsing_score = 0.0
        for signal, weight in BROWSING_SIGNALS:
            if signal in text:
                browsing_score += weight

        constraint_evidence = 0.0
        if slots.budget_min is not None or slots.budget_max is not None:
            constraint_evidence += 2.0
        if slots.brand:
            constraint_evidence += 1.0
        if slots.size:
            constraint_evidence += 1.0
        if slots.style:
            constraint_evidence += 0.5
        if slots.color or slots.material:
            constraint_evidence += 0.5
        buying_score += constraint_evidence

        if buying_score > browsing_score:
            intent = INTENT_BUYING
            winner, loser = buying_score, browsing_score
        else:
            intent = INTENT_BROWSING
            winner, loser = browsing_score, buying_score

        total = winner + loser
        if total <= 0:
            confidence = 0.5
        else:
            confidence = 0.5 + 0.5 * ((winner - loser) / total)
        confidence = max(0.0, min(1.0, confidence))
        return intent, round(confidence, 4)
