"""Shared contracts between the three team workstreams.

The official harness only requires ``reset`` and ``respond`` on the exported
Agent.  Internally we use these dataclasses and protocols so that intent
routing (A), retrieval/ranking (B), and orchestration/evaluation (C) remain
independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
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
class Product:
    """Canonical product record loaded from the frozen competition catalog."""

    parent_asin: str
    title: str = ""
    features: tuple[str, ...] = ()
    description: tuple[str, ...] = ()
    price: float | None = None
    categories: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    average_rating: float | None = None
    rating_number: int = 0
    store: str | None = None

    def __post_init__(self) -> None:
        value = str(self.parent_asin).strip()
        if not value:
            raise ValueError("parent_asin must not be empty")
        object.__setattr__(self, "parent_asin", value)
        object.__setattr__(self, "title", str(self.title or ""))
        object.__setattr__(self, "features", tuple(str(item) for item in (self.features or ())))
        object.__setattr__(self, "description", tuple(str(item) for item in (self.description or ())))
        object.__setattr__(self, "categories", tuple(str(item) for item in (self.categories or ())))
        object.__setattr__(self, "details", dict(self.details or {}))
        if self.price is not None:
            try:
                object.__setattr__(self, "price", float(self.price))
            except (TypeError, ValueError):
                object.__setattr__(self, "price", None)
        if self.average_rating is not None:
            try:
                object.__setattr__(self, "average_rating", float(self.average_rating))
            except (TypeError, ValueError):
                object.__setattr__(self, "average_rating", None)
        try:
            object.__setattr__(self, "rating_number", max(0, int(self.rating_number)))
        except (TypeError, ValueError):
            object.__setattr__(self, "rating_number", 0)


@dataclass
class Candidate:
    """The canonical candidate shared by retrieval, ranking, and orchestration.

    The product payload is optional for lightweight fallbacks, but the primary
    retrieval/ranking path keeps it attached so rankers never need a second
    product model or an ID conversion layer.
    """

    parent_asin: str = ""
    product: Any | None = None
    score: float = 0.0
    source_scores: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    keyword_score: float = 0.0
    category_score: float = 0.0
    attribute_score: float = 0.0
    semantic_score: float = 0.0
    popularity_score: float = 0.0
    final_score: float = 0.0

    def __post_init__(self) -> None:
        value = str(self.parent_asin or "").strip()
        if not value and self.product is not None:
            value = str(getattr(self.product, "parent_asin", "")).strip()
        if not value:
            raise ValueError("parent_asin must not be empty")
        self.parent_asin = value
        try:
            self.score = float(self.score)
        except (TypeError, ValueError):
            self.score = 0.0
        for name in (
            "keyword_score",
            "category_score",
            "attribute_score",
            "semantic_score",
            "popularity_score",
            "final_score",
        ):
            try:
                setattr(self, name, float(getattr(self, name)))
            except (TypeError, ValueError):
                setattr(self, name, 0.0)
        if self.final_score and not self.score:
            self.score = self.final_score
        self.source_scores = dict(self.source_scores or {})
        self.reasons = tuple(str(item) for item in (self.reasons or ()))


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

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index):
        return self.candidates[index]


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
    override: bool = False
    override_reason: str = ""
    no_preference: bool = False
    no_preference_attributes: tuple[str, ...] = ()
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
        object.__setattr__(
            self,
            "no_preference_attributes",
            tuple(
                str(x)
                for x in self.no_preference_attributes
                if str(x) in ALLOWED_ATTRIBUTES
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "hard_constraints": dict(self.hard_constraints),
            "soft_preferences": dict(self.soft_preferences),
            "negative_constraints": dict(self.negative_constraints),
            "remove_fields": list(self.remove_fields),
            "replace_fields": dict(self.replace_fields),
            "clarification_attribute": self.clarification_attribute,
            "override": self.override,
            "override_reason": self.override_reason,
            "no_preference": self.no_preference,
            "no_preference_attributes": list(self.no_preference_attributes),
        }


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

    session_id: str = ""
    user_profile: dict[str, Any] = field(default_factory=dict)
    intent: str | None = None
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: dict[str, Any] = field(default_factory=dict)
    negative_constraints: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    recent_messages: list[str] = field(default_factory=list)
    summary: str = ""
    last_candidates: list[str] = field(default_factory=list)
    asked_attributes: set[str] = field(default_factory=set)
    no_preference_attributes: set[str] = field(default_factory=set)
    completed: bool = False
    termination_reason: str | None = None
    version: int = 0
    max_turns: int = 10
    terminated: bool = False

    def clone(self) -> "SessionState":
        """Return a deep-enough copy suitable for a transaction rollback."""

        return SessionState(
            session_id=self.session_id,
            user_profile=deepcopy(self.user_profile),
            intent=self.intent,
            hard_constraints=deepcopy(self.hard_constraints),
            soft_preferences=deepcopy(self.soft_preferences),
            negative_constraints=deepcopy(self.negative_constraints),
            turn_count=self.turn_count,
            recent_messages=list(self.recent_messages),
            summary=self.summary,
            last_candidates=list(self.last_candidates),
            asked_attributes=set(self.asked_attributes),
            no_preference_attributes=set(self.no_preference_attributes),
            completed=self.completed,
            termination_reason=self.termination_reason,
            version=self.version,
            max_turns=self.max_turns,
            terminated=self.terminated,
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
