"""Conversation state machine for accumulating, removing, and overriding slots."""

import re
from typing import Any, Dict, List, Optional, Set

from agent.types import (
    ASK_ATTRIBUTES,
    INTENT_BROWSING,
    IntentResult,
    SessionState,
)


FIELD_ALIASES = {
    "category": ["category", "type", "kind"],
    "brand": ["brand", "make"],
    "color": ["color", "colour"],
    "size": ["size"],
    "material": ["material", "fabric"],
    "style": ["style", "look"],
    "budget": ["budget", "price", "price limit", "price range", "max price", "min price"],
    "feature": ["feature", "features", "function"],
    "use_case": ["use case", "use_case", "scenario", "occasion", "purpose"],
}

DELETION_VERBS = [
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
]


class StateMachine:
    def new_session(self, session_id: str, user_profile: Dict[str, Any]) -> SessionState:
        """Create a state seeded from the anonymized aggregate profile."""
        profile = dict(user_profile or {})
        state = SessionState(session_id=session_id, user_profile=profile)
        tags = profile.get("preference_tags")
        if isinstance(tags, list) and tags:
            state.soft_preferences["preference_tags"] = [str(t) for t in tags]
        return state

    def apply(
        self,
        state: SessionState,
        intent_result: IntentResult,
        message: str = "",
        turn: int = 0,
        asked_attribute: Optional[str] = None,
    ) -> Dict[str, Any]:
        diff: Dict[str, Any] = {
            "added": {},
            "modified": {},
            "removed": {},
            "intent": {"from": state.intent, "to": intent_result.intent},
            "override": False,
            "terminated": False,
            "turn_count": state.turn_count,
        }

        if state.terminated:
            diff["terminated"] = True
            return diff

        state.turn_count = turn if turn else state.turn_count + 1
        diff["turn_count"] = state.turn_count
        if message:
            state.recent_messages.append(message)
            state.recent_messages = state.recent_messages[-3:]

        if asked_attribute is not None and asked_attribute in ASK_ATTRIBUTES:
            state.asked_attributes.add(asked_attribute)

        if intent_result.no_preference:
            if asked_attribute is not None and asked_attribute in ASK_ATTRIBUTES:
                state.no_preference_attributes.add(asked_attribute)
        elif intent_result.override:
            self._apply_override(state, intent_result, diff)
        else:
            self._apply_merge(state, intent_result, message, diff)

        state.summary = self.summarize(state)
        if state.turn_count >= state.max_turns:
            state.terminated = True
            diff["terminated"] = True
        return diff

    def _apply_override(
        self,
        state: SessionState,
        intent_result: IntentResult,
        diff: Dict[str, Any],
    ) -> None:
        old_intent = state.intent
        old_hard = dict(state.hard_constraints)
        old_soft = dict(state.soft_preferences)
        old_negative = dict(state.negative_constraints)

        state.intent = intent_result.intent
        state.hard_constraints = dict(intent_result.hard_constraints)
        state.soft_preferences = dict(intent_result.soft_preferences)
        state.negative_constraints = dict(intent_result.negative_constraints)

        diff["override"] = True
        diff["intent"] = {"from": old_intent, "to": intent_result.intent}
        diff["removed"] = {
            k: v
            for k, v in list(old_hard.items())
            + list(old_soft.items())
            + list(old_negative.items())
        }
        diff["added"] = dict(intent_result.hard_constraints)
        for k, v in intent_result.soft_preferences.items():
            diff["added"][k] = v
        for k, v in intent_result.negative_constraints.items():
            diff["added"][k] = v

    def _apply_merge(
        self,
        state: SessionState,
        intent_result: IntentResult,
        message: str,
        diff: Dict[str, Any],
    ) -> None:
        if intent_result.intent != state.intent:
            diff["intent"] = {"from": state.intent, "to": intent_result.intent}
            state.intent = intent_result.intent

        for field in self.detect_deletions(message):
            self._remove_field(state, field, diff)

        for key, value in intent_result.hard_constraints.items():
            if key not in state.hard_constraints:
                diff["added"][key] = value
            elif state.hard_constraints[key] != value:
                diff["modified"][key] = {
                    "from": state.hard_constraints[key],
                    "to": value,
                }
            state.hard_constraints[key] = value

        for key, value in intent_result.soft_preferences.items():
            if key not in state.soft_preferences:
                diff["added"][key] = value
            elif state.soft_preferences[key] != value:
                diff["modified"][key] = {
                    "from": state.soft_preferences[key],
                    "to": value,
                }
            state.soft_preferences[key] = value

        for key, value in intent_result.negative_constraints.items():
            existing = state.negative_constraints.get(key)
            if existing is None:
                state.negative_constraints[key] = list(value)
                diff["added"][key] = list(value)
            else:
                merged = list(existing)
                for item in value:
                    if item not in merged:
                        merged.append(item)
                state.negative_constraints[key] = merged
                diff["modified"][key] = {"from": list(existing), "to": merged}

    def detect_deletions(self, message: str) -> Set[str]:
        text = message.lower().strip()
        deleted: Set[str] = set()
        for canonical, aliases in FIELD_ALIASES.items():
            for alias in aliases:
                for verb in DELETION_VERBS:
                    pattern = (
                        re.escape(verb)
                        + r"\s+(?:the\s+|about\s+the\s+|my\s+)?"
                        + re.escape(alias)
                    )
                    if re.search(r"\b" + pattern, text):
                        deleted.add(canonical)
        return deleted

    def _remove_field(
        self, state: SessionState, field: str, diff: Dict[str, Any]
    ) -> None:
        keys: List[str] = [field]
        for key in keys:
            for bucket in (
                state.hard_constraints,
                state.soft_preferences,
                state.negative_constraints,
            ):
                if key in bucket:
                    diff["removed"][key] = bucket.pop(key)

    @staticmethod
    def summarize(state: SessionState) -> str:
        parts: List[str] = [state.intent]
        for key in ("category", "brand", "color", "size", "material", "style", "budget"):
            if key in state.hard_constraints:
                parts.append(f"{key}={state.hard_constraints[key]}")
        for key in ("feature", "use_case", "preference_tags"):
            if key in state.soft_preferences:
                parts.append(f"{key}={state.soft_preferences[key]}")
        neg_keys = [k for k in state.negative_constraints if k != "negative_keywords"]
        if neg_keys:
            parts.append(f"not={neg_keys}")
        return "; ".join(parts)
