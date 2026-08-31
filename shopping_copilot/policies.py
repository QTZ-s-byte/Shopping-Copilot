"""Default policies used by the orchestrator.

These policies are deliberately conservative and deterministic.  A teammate
can inject a richer LLM-backed policy without changing the lifecycle code.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .contracts import Candidate, IntentResult, SessionState


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry policy for one pipeline stage."""

    max_retries: int = 1
    backoff_seconds: float = 0.02

    def delays(self) -> Iterable[float]:
        for attempt in range(max(0, int(self.max_retries))):
            yield max(0.0, float(self.backoff_seconds)) * (2**attempt)


class DefaultIntentRouter:
    """Minimal offline router used until member A's implementation is injected."""

    _override = re.compile(r"\b(actually|instead|forget|ignore|change|rather)\b", re.I)
    _buying = re.compile(
        r"\b(need|buy|purchase|looking for|find me|under \$|below \$|budget|size|brand)\b",
        re.I,
    )
    _attributes = {
        "color": re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow)\b", re.I),
        "material": re.compile(r"\b(cotton|polyester|nylon|leather|wool|silk|denim|fabric)\b", re.I),
        "size": re.compile(r"\b(size|small|medium|large|xl|wide|narrow)\b", re.I),
        "brand": re.compile(r"\b(nike|adidas|puma|reebok|coach|gucci|levi)\b", re.I),
        "budget": re.compile(r"(?:\$|under|below|less than)\s*\d", re.I),
        "use_case": re.compile(r"\b(running|hiking|gym|work|formal|casual|outdoor|winter)\b", re.I),
    }

    def classify(self, user_message: str, state: SessionState) -> IntentResult:
        message = str(user_message or "").strip()
        intent = "buying" if self._buying.search(message) else (state.intent or "browsing")
        override = bool(self._override.search(message)) and state.turn_count > 0
        hard: dict[str, str] = {}
        soft: dict[str, str] = {}
        for name, pattern in self._attributes.items():
            match = pattern.search(message)
            if match:
                value = match.group(0).strip().lower()
                (hard if intent == "buying" else soft)[name] = value
        replace = {}
        if override and hard:
            replace.update(hard)
        ask = None
        if not hard and not soft:
            ask = "category" if not state.hard_constraints and not state.soft_preferences else "feature"
        return IntentResult(
            intent=intent,
            confidence=0.65 if hard or soft else 0.45,
            hard_constraints=hard,
            soft_preferences=soft,
            replace_fields=replace,
            clarification_attribute=ask,
            raw=message,
        )


class EmptyRetriever:
    """Safe fallback when member B's catalog retriever is unavailable."""

    def retrieve(self, query: str, state: SessionState, top_k: int):
        return []


class ScoreRanker:
    """Deterministic ranker fallback that preserves stable ordering."""

    def rank(self, query: str, candidates: Sequence[Candidate], state: SessionState):
        return sorted(
            candidates,
            key=lambda item: (-float(item.score), item.parent_asin),
        )


def build_query(state: SessionState, user_message: str) -> str:
    """Build a compact query from current state and the latest message."""

    parts: list[str] = []
    if state.intent:
        parts.append(state.intent)
    for container in (
        state.hard_constraints,
        state.soft_preferences,
        state.negative_constraints,
    ):
        for key, value in container.items():
            parts.append(f"{key} {value}")
    if user_message.strip():
        parts.append(user_message.strip())
    return " ".join(parts)[:4000]


def default_clarification(attribute: str | None, *, broad: bool = False) -> str:
    attribute = attribute if attribute else "category"
    prompts = {
        "category": "What type of product are you looking for (for example, shoes, bags, or clothing)?",
        "material": "Do you have a preferred material, such as leather, cotton, or nylon?",
        "color": "Which color should I prioritize?",
        "size": "What size or fit should I prioritize?",
        "style": "Which style or occasion should I prioritize?",
        "brand": "Do you have a preferred brand?",
        "budget": "What budget range should I use?",
        "feature": "Which feature matters most to you?",
        "use_case": "What will you mainly use the product for?",
        "other": "What is the most important requirement for you?",
    }
    prefix = "There are many possible matches. " if broad else "To narrow this down, "
    return prefix + prompts.get(attribute, prompts["other"])


def safe_sleep(seconds: float) -> None:
    """A separately named sleep hook that is easy to replace in tests."""

    if seconds > 0:
        time.sleep(seconds)


def call_with_retry(
    operation: Callable[[], Any],
    policy: RetryPolicy,
) -> tuple[Any, int]:
    """Run an operation with bounded exponential backoff.

    Exceptions are re-raised after the final attempt so the orchestrator can
    record the stage and select a safe fallback.
    """

    last_error: BaseException | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return operation(), attempt
        except BaseException as exc:  # noqa: BLE001 - boundary intentionally catches plugins
            last_error = exc
            if attempt >= policy.max_retries:
                break
            delays = list(policy.delays())
            safe_sleep(delays[attempt] if attempt < len(delays) else 0.0)
    assert last_error is not None
    raise last_error

