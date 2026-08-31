"""Shared contracts between the three team workstreams.

The official harness only requires ``reset`` and ``respond`` on the exported
Agent.  Internally we use these dataclasses and protocols so that intent
routing (A), retrieval/ranking (B), and orchestration/evaluation (C) remain
independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


ALLOWED_INTENTS = {"buying", "browsing"}
ALLOWED_ATTRIBUTES = {
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
}


@dataclass(frozen=True)
class Candidate:
    """A catalog candidate returned by a retriever or ranker."""

    parent_asin: str
    score: float = 0.0
    source_scores: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        value = str(self.parent_asin).strip()
        if not value:
            raise ValueError("parent_asin must not be empty")
        object.__setattr__(self, "parent_asin", value)
        try:
            object.__setattr__(self, "score", float(self.score))
        except (TypeError, ValueError):
            object.__setattr__(self, "score", 0.0)


@dataclass(frozen=True)
class RetrievalResult:
    """Candidates plus an optional pre-truncation count.

    A plain list of :class:`Candidate` is also accepted by the orchestrator;
    this richer result lets a retriever signal that a query is too broad even
    when it only returns the first ``top_k`` candidates.
    """

    candidates: tuple[Candidate, ...] = ()
    total_count: int | None = None
    exhausted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.total_count is not None:
            object.__setattr__(self, "total_count", max(0, int(self.total_count)))


@dataclass(frozen=True)
class IntentResult:
    """Normalized output from the intent/slot extraction module.

    ``hard_constraints`` and ``soft_preferences`` are additive updates.  A
    field in ``replace_fields`` replaces all previous values for that field;
    ``remove_fields`` removes it.  This explicit representation is what makes
    intent-overwrite scenarios auditable.
    """

    intent: str | None = None
    confidence: float = 0.0
    hard_constraints: Mapping[str, Any] = field(default_factory=dict)
    soft_preferences: Mapping[str, Any] = field(default_factory=dict)
    negative_constraints: Mapping[str, Any] = field(default_factory=dict)
    remove_fields: tuple[str, ...] = ()
    replace_fields: Mapping[str, Any] = field(default_factory=dict)
    clarification_attribute: str | None = None
    raw: str = ""

    def __post_init__(self) -> None:
        normalized_intent = self.intent.lower().strip() if isinstance(self.intent, str) else None
        if normalized_intent not in ALLOWED_INTENTS:
            normalized_intent = None
        object.__setattr__(self, "intent", normalized_intent)
        try:
            confidence = min(1.0, max(0.0, float(self.confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        object.__setattr__(self, "confidence", confidence)
        attribute = self.clarification_attribute
        if attribute not in ALLOWED_ATTRIBUTES:
            object.__setattr__(self, "clarification_attribute", None)
        object.__setattr__(self, "remove_fields", tuple(str(x) for x in self.remove_fields))


@dataclass(frozen=True)
class StateDiff:
    """The observable state change produced by one user turn."""

    added: Mapping[str, Any] = field(default_factory=dict)
    removed: Mapping[str, Any] = field(default_factory=dict)
    replaced: Mapping[str, Any] = field(default_factory=dict)
    intent_before: str | None = None
    intent_after: str | None = None


@dataclass
class SessionState:
    """Mutable, serializable state for one isolated evaluator session."""

    session_id: str
    user_profile: dict[str, Any] = field(default_factory=dict)
    intent: str | None = None
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: dict[str, Any] = field(default_factory=dict)
    negative_constraints: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    recent_messages: list[str] = field(default_factory=list)
    summary: str = ""
    last_candidates: list[str] = field(default_factory=list)
    completed: bool = False
    termination_reason: str | None = None
    version: int = 0

    def clone(self) -> "SessionState":
        """Return a deep-enough copy suitable for a transaction rollback."""

        return SessionState(
            session_id=self.session_id,
            user_profile=dict(self.user_profile),
            intent=self.intent,
            hard_constraints=dict(self.hard_constraints),
            soft_preferences=dict(self.soft_preferences),
            negative_constraints=dict(self.negative_constraints),
            turn_count=self.turn_count,
            recent_messages=list(self.recent_messages),
            summary=self.summary,
            last_candidates=list(self.last_candidates),
            completed=self.completed,
            termination_reason=self.termination_reason,
            version=self.version,
        )


@dataclass(frozen=True)
class AgentResponse:
    """Internal response object; ``to_dict`` matches the official contract."""

    message: str
    ask_attribute: str | None = None
    recommendations: tuple[Candidate | Mapping[str, Any] | str, ...] = ()
    usage: Mapping[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        recommendations: list[dict[str, Any]] = []
        for candidate in self.recommendations:
            if isinstance(candidate, Candidate):
                recommendations.append(
                    {"parent_asin": candidate.parent_asin, "score": candidate.score}
                )
            elif isinstance(candidate, Mapping):
                item = dict(candidate)
                if "parent_asin" in item:
                    recommendations.append(item)
            else:
                recommendations.append({"parent_asin": str(candidate)})
        usage = {
            "prompt_tokens": max(0, int(self.usage.get("prompt_tokens", 0))),
            "completion_tokens": max(0, int(self.usage.get("completion_tokens", 0))),
        }
        return {
            "message": str(self.message),
            "ask_attribute": self.ask_attribute if self.ask_attribute in ALLOWED_ATTRIBUTES else None,
            "recommendations": recommendations,
            "usage": usage,
        }


class IntentRouter(Protocol):
    def classify(self, user_message: str, state: SessionState) -> IntentResult | Mapping[str, Any]:
        ...


class Retriever(Protocol):
    def retrieve(
        self, query: str, state: SessionState, top_k: int
    ) -> RetrievalResult | Sequence[Candidate | Mapping[str, Any] | str]:
        ...


class Ranker(Protocol):
    def rank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        state: SessionState,
    ) -> Sequence[Candidate | Mapping[str, Any] | str]:
        ...


class TraceSink(Protocol):
    def emit(self, event: Mapping[str, Any]) -> None:
        ...

