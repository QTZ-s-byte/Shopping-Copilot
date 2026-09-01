"""Rule-based intent routing with an optional external-LLM enhancement hook."""

import re
from typing import Callable, Optional, Tuple

from agent.clarification import ClarificationPolicy
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
    ("budget", 1.2),
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

DELETION_VERBS = (
    "forget",
    "remove",
    "drop",
    "clear",
    "delete",
    "reset",
    "no longer need",
    "don't care about",
    "dont care about",
    "ignore",
    "skip",
)

FIELD_ALIASES = {
    "category": ("category", "type", "kind"),
    "brand": ("brand", "make"),
    "color": ("color", "colour"),
    "size": ("size",),
    "material": ("material", "fabric"),
    "style": ("style", "look"),
    "budget": ("budget", "price", "price limit", "price range"),
    "feature": ("feature", "features"),
    "use_case": ("use case", "scenario", "occasion", "purpose"),
}

ATTRIBUTE_NAMES = {
    "category": "category",
    "material": "material",
    "color": "color",
    "size": "size",
    "style": "style",
    "brand": "brand",
    "budget": "budget",
    "feature": "feature",
    "use_case": "use case",
}


class IntentRouter:
    def __init__(
        self,
        slot_extractor: Optional[SlotExtractor] = None,
        llm_classifier: Optional[Callable[[str, SessionState], IntentResult]] = None,
        clarification_policy: Optional[ClarificationPolicy] = None,
    ) -> None:
        self.slot_extractor = slot_extractor or SlotExtractor()
        self.llm_classifier = llm_classifier
        self.clarification_policy = clarification_policy or ClarificationPolicy()

    def classify(self, user_message: str, state: SessionState) -> IntentResult:
        if self.llm_classifier is not None:
            try:
                return self.llm_classifier(user_message, state)
            except Exception:
                pass

        slots = self.slot_extractor.extract(user_message)
        override, override_reason = self._detect_override(user_message, state, slots)
        no_preference = self._detect_no_preference(user_message)
        no_preference_attributes = self._detect_no_preference_attributes(user_message)
        remove_fields = self._detect_remove_fields(user_message)

        if no_preference:
            # A no-preference reply is not a new product intent.
            # Preserve the current intent and do not add filler words such as
            # "additional", "preference", or the attribute name as keywords.
            intent = state.intent or INTENT_BROWSING
            confidence = 1.0
            hard = {}
            soft = {}
            negative = {}
        else:
            intent, confidence = self._detect_intent(user_message, slots)
            hard = slots.to_hard_constraints()
            soft = slots.to_soft_preferences()
            negative = slots.to_negative_constraints()

        replace_fields = {**hard, **soft} if override else {}
        clarification_attribute = (
            None
            if no_preference
            else self.clarification_policy.suggest(state, intent, hard, soft, negative)
        )

        return IntentResult(
            intent=intent,
            confidence=confidence,
            hard_constraints=hard,
            soft_preferences=soft,
            negative_constraints=negative,
            remove_fields=remove_fields,
            replace_fields=replace_fields,
            clarification_attribute=clarification_attribute,
            override=override,
            override_reason=override_reason,
            no_preference=no_preference,
            no_preference_attributes=no_preference_attributes,
            raw=user_message,
        )

    def _detect_override(
        self,
        message: str,
        state: SessionState,
        slots: ExtractedSlots,
    ) -> Tuple[bool, str]:
        text = message.lower().strip()
        strong = next((signal for signal in STRONG_OVERRIDE if signal in text), None)
        if strong:
            return True, strong

        # "forget shoes" is a goal override, while "forget the color" is a
        # field deletion handled through remove_fields.
        if "forget" in text:
            if self._detect_remove_fields(message):
                return False, ""
            return True, "forget"

        hit = next((signal for signal in OVERRIDE_SIGNALS if signal in text), None)
        if hit is None:
            return False, ""
        has_new_goal = any(
            token in text for token in ("want", "need", "looking for", "get me")
        ) or bool(slots.category or slots.brand or slots.material or slots.feature)
        if self._goal_changed(state, slots) or has_new_goal:
            return True, hit
        return False, ""

    def _goal_changed(self, state: SessionState, slots: ExtractedSlots) -> bool:
        old_category = _as_set(state.hard_constraints.get("category"))
        new_category = set(slots.category)
        if new_category and old_category and new_category.isdisjoint(old_category):
            return True

        old_brand = _as_set(state.hard_constraints.get("brand"))
        new_brand = set(slots.brand)
        if new_brand and old_brand and new_brand.isdisjoint(old_brand):
            return True
        return False

    def _detect_no_preference(self, message: str) -> bool:
        text = message.lower().strip()

        if any(phrase in text for phrase in NO_PREFERENCE_PHRASES):
            return True

        # Evaluator replies such as:
        # "I don't have an additional preference for use_case."
        # should not be interpreted as a new product requirement.
        normalized = text.replace("'", "")
        if re.search(
            r"\bdo\s*nt\s+have\s+(?:an?\s+)?(?:additional\s+)?preference\b",
            normalized,
            re.I,
        ):
            return True

        return False

    def _detect_no_preference_attributes(self, message: str) -> Tuple[str, ...]:
        text = message.lower().strip()
        found = []
        for attr, name in ATTRIBUTE_NAMES.items():
            aliases = {name, attr, attr.replace("_", " ")}
            attribute_pattern = "(?:" + "|".join(
                re.escape(alias) for alias in sorted(aliases)
            ) + ")"
            if re.search(r"\b(?:no|without)\s+" + attribute_pattern + r"\b", text):
                found.append(attr)
            elif re.search(
                r"\b" + attribute_pattern + r"\s+(?:doesn'?t|doesnt)\s+matter\b", text
            ):
                found.append(attr)
            elif re.search(
                r"\b(?:additional\s+)?preference\s+for\s+" + attribute_pattern + r"\b",
                text,
            ):
                found.append(attr)
        return tuple(sorted(set(found)))

    def _detect_remove_fields(self, message: str) -> Tuple[str, ...]:
        text = message.lower().strip()
        removed = []
        for canonical, aliases in FIELD_ALIASES.items():
            for alias in aliases:
                for verb in DELETION_VERBS:
                    pattern = (
                        re.escape(verb)
                        + r"\s+(?:the\s+|about\s+the\s+|my\s+)?"
                        + re.escape(alias)
                    )
                    if re.search(r"\b" + pattern, text):
                        removed.append(canonical)
                        break
        return tuple(sorted(set(removed)))

    def _detect_intent(
        self, message: str, slots: ExtractedSlots
    ) -> Tuple[str, float]:
        text = message.lower().strip()

        buying_score = _score_signals(text, BUYING_SIGNALS)
        browsing_score = _score_signals(text, BROWSING_SIGNALS)

        constraint_evidence = 0.0
        if slots.budget_min is not None or slots.budget_max is not None:
            constraint_evidence += 2.0
        if slots.category:
            constraint_evidence += 0.75
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


def _as_set(value) -> set:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).lower() for item in value}
    return {str(value).lower()}


def _score_signals(text: str, signals) -> float:
    score = 0.0
    for signal, weight in signals:
        if re.search(r"\b" + re.escape(signal) + r"\b", text):
            score += weight
    return score
