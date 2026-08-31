"""Lifecycle orchestrator for the Track 4 Shopping Copilot.

The orchestrator owns the evaluator-facing lifecycle and deliberately knows
only the small protocols in :mod:`shopping_copilot.contracts`.  Intent routing
and retrieval/ranking implementations can therefore be developed by the other
team members and injected later.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    ALLOWED_ATTRIBUTES,
    AgentResponse,
    Candidate,
    IntentResult,
    RetrievalResult,
    SessionState,
)
from .memory import InMemoryContextMemory
from .policies import (
    DefaultIntentRouter,
    EmptyRetriever,
    RetryPolicy,
    ScoreRanker,
    build_query,
    call_with_retry,
    default_clarification,
)
from .trace import InMemoryTraceSink, NullTraceSink, message_digest, utc_now


class ShoppingOrchestrator:
    """Coordinate one Agent turn while enforcing the official protocol."""

    def __init__(
        self,
        router: Any | None = None,
        retriever: Any | None = None,
        ranker: Any | None = None,
        *,
        memory: InMemoryContextMemory | None = None,
        trace_sink: Any | None = None,
        valid_catalog_ids: set[str] | Callable[[str], bool] | None = None,
        max_turns: int = 10,
        broad_candidate_threshold: int = 250,
        retry_policy: RetryPolicy | None = None,
        top_k: int = 10,
        stage_timeout_seconds: float | None = None,
    ) -> None:
        self.router = router or DefaultIntentRouter()
        self.retriever = retriever or EmptyRetriever()
        self.ranker = ranker or ScoreRanker()
        self.memory = memory or InMemoryContextMemory()
        self.trace_sink = trace_sink or NullTraceSink()
        self.valid_catalog_ids = valid_catalog_ids
        self.max_turns = max(1, int(max_turns))
        self.broad_candidate_threshold = max(1, int(broad_candidate_threshold))
        self.retry_policy = retry_policy or RetryPolicy(max_retries=1, backoff_seconds=0.01)
        self.top_k = int(top_k)
        self.stage_timeout_seconds = (
            None
            if stage_timeout_seconds is None
            else max(0.001, float(stage_timeout_seconds))
        )
        if self.top_k != 10:
            raise ValueError("Track 4's official Agent contract requires top_k=10")

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Reset one evaluator session and discard all prior state."""

        self.memory.reset(session_id, user_profile)
        self._emit(
            {
                "event": "session_reset",
                "session_id": str(session_id),
                "profile_fields": sorted(str(key) for key in (user_profile or {})),
                "timestamp": utc_now(),
            }
        )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Run one complete turn and return the official response dictionary.

        ``request_id`` is an internal optional extension used for idempotency;
        the official harness does not pass it.  When omitted, a deterministic
        digest of the request is used.
        """

        session_id = str(session_id).strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string")
        if not isinstance(turn, int):
            raise TypeError("turn must be an integer")
        if int(top_k) != self.top_k:
            raise ValueError("top_k must be 10 for the official Track 4 contract")

        # A turn beyond the hard limit is never allowed to call A/B plugins.
        # This also gives a safe response if a caller violates the harness.
        if turn > self.max_turns:
            return self._termination_response(session_id, "turn_limit")
        if turn < 1:
            raise ValueError("turn must be between 1 and 10")

        request_id = request_id or self._request_id(session_id, turn, user_message)
        cached = self.memory.cached_response(session_id, request_id)
        if cached is not None:
            return cached

        started = time.perf_counter()
        with self.memory.session_guard(session_id) as record:
            state = record.state
            if state.completed or state.turn_count >= self.max_turns:
                response = self._termination_response(session_id, state.termination_reason or "turn_limit")
                self.memory.cache_response(session_id, request_id, response)
                return response
            expected_turn = state.turn_count + 1
            if turn != expected_turn:
                raise ValueError(
                    f"unexpected turn {turn}; expected {expected_turn} for session {session_id}"
                )
            before = state.clone()
            fallback_stages: list[str] = []
            retry_counts: dict[str, int] = {}
            stage_errors: list[dict[str, str]] = []
            state_diff = None
            query = ""
            retrieval_total: int | None = None
            ranked: list[Candidate] = []
            ask_attribute: str | None = None

            try:
                try:
                    intent_result, retries = self._run_router(user_message, state)
                    retry_counts["router"] = retries
                except Exception as exc:  # noqa: BLE001 - plugin boundary
                    stage_errors.append(self._error_info("router", exc))
                    fallback_stages.append("router")
                    intent_result = IntentResult(
                        intent=state.intent or "browsing",
                        clarification_attribute="other",
                    )

                state_diff = self.memory.apply_turn(
                    session_id,
                    user_message,
                    intent_result,
                    expected_turn=turn,
                    max_turns=self.max_turns,
                )
                state = record.state
                query = build_query(state, user_message)

                try:
                    retrieval, retries = self._run_retriever(query, state, self.top_k)
                    retry_counts["retriever"] = retries
                    retrieval_total = retrieval.total_count
                except Exception as exc:  # noqa: BLE001 - plugin boundary
                    stage_errors.append(self._error_info("retriever", exc))
                    fallback_stages.append("retriever")
                    retrieval = RetrievalResult()

                candidates = self._sanitize_candidates(retrieval.candidates)
                broad = bool(
                    retrieval_total is not None
                    and retrieval_total > self.broad_candidate_threshold
                )

                if candidates:
                    try:
                        ranked_values, retries = self._run_ranker(query, candidates, state)
                        retry_counts["ranker"] = retries
                        ranked = self._sanitize_candidates(ranked_values)
                    except Exception as exc:  # noqa: BLE001 - plugin boundary
                        stage_errors.append(self._error_info("ranker", exc))
                        fallback_stages.append("ranker")
                        # Candidate scores are supplied by the retriever and
                        # provide a deterministic, offline-safe fallback.
                        ranked = sorted(candidates, key=lambda item: (-item.score, item.parent_asin))
                else:
                    ranked = []

                state = record.state
                self.memory.set_candidates(session_id, [item.parent_asin for item in ranked])
                ask_attribute = self._select_ask_attribute(intent_result, state, broad, ranked)
                self.memory.mark_asked_attribute(session_id, ask_attribute)
                response = self._make_response(
                    ranked,
                    ask_attribute=ask_attribute,
                    broad=broad,
                    no_results=not bool(ranked),
                )

                if turn >= self.max_turns:
                    self.memory.mark_completed(session_id, "turn_limit")
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
                self._emit_turn(
                    session_id=session_id,
                    request_id=request_id,
                    turn=turn,
                    user_message=user_message,
                    before=before,
                    after=record.state,
                    state_diff=state_diff,
                    query=query,
                    candidates=ranked,
                    retrieval_total=retrieval_total,
                    ask_attribute=ask_attribute,
                    elapsed_ms=elapsed_ms,
                    fallback_stages=fallback_stages,
                    retry_counts=retry_counts,
                    errors=stage_errors,
                    done=record.state.completed,
                )
                self.memory.cache_response(session_id, request_id, response)
                return response
            except Exception as exc:  # noqa: BLE001 - final safety boundary
                # A catastrophic integration error must not leave a half-applied
                # state that makes the next evaluator turn inconsistent.
                self.memory.restore(session_id, before)
                error_info = self._error_info("orchestrator", exc)
                self._emit(
                    {
                        "event": "turn_error",
                        "session_id": session_id,
                        "request_id": request_id,
                        "turn": turn,
                        "error": error_info,
                        "fallback_stages": ["orchestrator"],
                        "timestamp": utc_now(),
                    }
                )
                # Returning a valid empty response is preferable to violating
                # the contract; the official evaluator will count it as a miss.
                response = AgentResponse(
                    message="I couldn't complete that search. Could you try a different requirement?",
                    ask_attribute="other",
                    recommendations=(),
                ).to_dict()
                self.memory.cache_response(session_id, request_id, response)
                return response

    def _run_router(self, user_message: str, state: SessionState) -> tuple[IntentResult, int]:
        value, retries = call_with_retry(
            lambda: self._invoke_with_timeout(
                lambda: self.router.classify(user_message, state)
            ),
            self.retry_policy,
        )
        return self._normalize_intent(self._coerce_intent(value), state), retries

    @staticmethod
    def _normalize_intent(result: IntentResult, state: SessionState) -> IntentResult:
        """Convert the shared A result into explicit C state operations."""

        if not result.override:
            return result
        incoming = set(result.hard_constraints) | set(result.soft_preferences) | set(result.negative_constraints)
        existing = set(state.hard_constraints) | set(state.soft_preferences) | set(state.negative_constraints)
        remove_fields = tuple(sorted(existing - incoming))
        replace_fields = {
            **dict(result.hard_constraints),
            **dict(result.soft_preferences),
            **dict(result.negative_constraints),
        }
        return IntentResult(
            intent=result.intent,
            confidence=result.confidence,
            hard_constraints=result.hard_constraints,
            soft_preferences=result.soft_preferences,
            negative_constraints=result.negative_constraints,
            remove_fields=remove_fields,
            replace_fields=replace_fields,
            clarification_attribute=result.clarification_attribute,
            override=result.override,
            override_reason=result.override_reason,
            no_preference=result.no_preference,
            no_preference_attributes=result.no_preference_attributes,
            raw=result.raw,
        )

    def _run_retriever(
        self, query: str, state: SessionState, top_k: int
    ) -> tuple[RetrievalResult, int]:
        value, retries = call_with_retry(
            lambda: self._invoke_with_timeout(
                lambda: self.retriever.retrieve(query, state, top_k)
            ),
            self.retry_policy,
        )
        return self._coerce_retrieval(value), retries

    def _run_ranker(
        self, query: str, candidates: Sequence[Candidate], state: SessionState
    ) -> tuple[Sequence[Candidate], int]:
        value, retries = call_with_retry(
            lambda: self._invoke_with_timeout(
                lambda: self.ranker.rank(query, candidates, state)
            ),
            self.retry_policy,
        )
        if isinstance(value, Mapping) and "candidates" in value:
            value = value["candidates"]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("ranker must return a sequence of candidates")
        return [self._coerce_candidate(item) for item in value], retries

    def _invoke_with_timeout(self, operation: Callable[[], Any]) -> Any:
        """Run a plugin call with an optional hard wait deadline.

        Timed-out work is abandoned and the normal retry/fallback path takes
        over.  The default is disabled because the official evaluator may run
        on slower machines; teams can enable a measured deadline explicitly.
        """

        if self.stage_timeout_seconds is None:
            return operation()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shopping-stage")
        future = executor.submit(operation)
        try:
            return future.result(timeout=self.stage_timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"plugin exceeded {self.stage_timeout_seconds:.3f}s deadline"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _coerce_intent(value: Any) -> IntentResult:
        if isinstance(value, IntentResult):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("router must return IntentResult or mapping")
        # Accept the common `slots` alias used by teammate A's implementation.
        slots = value.get("slots") if isinstance(value.get("slots"), Mapping) else {}
        hard = value.get("hard_constraints", value.get("hard", {}))
        soft = value.get("soft_preferences", value.get("soft", {}))
        if not hard and slots:
            hard = slots
        return IntentResult(
            intent=value.get("intent"),
            confidence=value.get("confidence", 0.0),
            hard_constraints=hard if isinstance(hard, Mapping) else {},
            soft_preferences=soft if isinstance(soft, Mapping) else {},
            negative_constraints=(
                value.get("negative_constraints", {})
                if isinstance(value.get("negative_constraints", {}), Mapping)
                else {}
            ),
            remove_fields=tuple(value.get("remove_fields", value.get("remove", ())) or ()),
            replace_fields=(
                value.get("replace_fields", value.get("replace", {}))
                if isinstance(value.get("replace_fields", value.get("replace", {})), Mapping)
                else {}
            ),
            clarification_attribute=value.get("clarification_attribute", value.get("ask_attribute")),
            raw=str(value.get("raw", "")),
        )

    @staticmethod
    def _coerce_candidate(value: Any) -> Candidate:
        if isinstance(value, Candidate):
            return value
        if isinstance(value, Mapping):
            parent_asin = value.get("parent_asin", value.get("product_id", value.get("id", "")))
            return Candidate(
                parent_asin=str(parent_asin),
                score=value.get("score", 0.0),
                source_scores=value.get("source_scores", {}),
                reasons=tuple(str(x) for x in value.get("reasons", ()) or ()),
            )
        return Candidate(parent_asin=str(value))

    def _coerce_retrieval(self, value: Any) -> RetrievalResult:
        if isinstance(value, RetrievalResult):
            return value
        if isinstance(value, Mapping):
            candidates = value.get("candidates", value.get("recommendations", ()))
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                candidates = ()
            total = value.get("total_count", value.get("candidate_count"))
            return RetrievalResult(
                candidates=tuple(self._coerce_candidate(item) for item in candidates),
                total_count=total if isinstance(total, int) else None,
                exhausted=bool(value.get("exhausted", False)),
            )
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("retriever must return RetrievalResult or sequence")
        return RetrievalResult(candidates=tuple(self._coerce_candidate(item) for item in value))

    def _sanitize_candidates(self, values: Sequence[Candidate]) -> list[Candidate]:
        result: list[Candidate] = []
        seen: set[str] = set()
        for value in values:
            try:
                candidate = self._coerce_candidate(value)
            except (TypeError, ValueError):
                continue
            parent_asin = candidate.parent_asin
            if not parent_asin or parent_asin in seen:
                continue
            if self.valid_catalog_ids is not None:
                valid = (
                    parent_asin in self.valid_catalog_ids
                    if isinstance(self.valid_catalog_ids, set)
                    else bool(self.valid_catalog_ids(parent_asin))
                )
                if not valid:
                    continue
            seen.add(parent_asin)
            result.append(candidate)
            if len(result) >= 100:
                break
        return result

    @staticmethod
    def _select_ask_attribute(
        intent_result: IntentResult,
        state: SessionState,
        broad: bool,
        ranked: Sequence[Candidate],
    ) -> str | None:
        if (
            intent_result.clarification_attribute in ALLOWED_ATTRIBUTES
            and intent_result.clarification_attribute not in state.no_preference_attributes
        ):
            return intent_result.clarification_attribute
        if not broad and ranked:
            return None
        # Prefer a missing field that is meaningful for the current intent.
        for field in ("category", "use_case", "budget", "size", "color", "material", "brand", "feature"):
            if field in state.asked_attributes or field in state.no_preference_attributes:
                continue
            if field not in state.hard_constraints and field not in state.soft_preferences:
                return field
        return "other" if broad or not ranked else None

    @staticmethod
    def _make_response(
        ranked: Sequence[Candidate],
        *,
        ask_attribute: str | None,
        broad: bool,
        no_results: bool,
    ) -> dict[str, Any]:
        recommendations = tuple(ranked[:10])
        if ask_attribute:
            message = default_clarification(ask_attribute, broad=broad)
        elif no_results:
            message = "I couldn't find a matching item. Could you relax one requirement?"
        else:
            message = "Here are the closest matches I found."
        return AgentResponse(
            message=message,
            ask_attribute=ask_attribute,
            recommendations=recommendations,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        ).to_dict()

    def _termination_response(self, session_id: str, reason: str) -> dict[str, Any]:
        try:
            if session_id in self.memory.sessions():
                self.memory.mark_completed(session_id, reason)
        except (KeyError, ValueError):
            pass
        response = AgentResponse(
            message="This shopping session has ended. Please start a new session to continue.",
            ask_attribute=None,
            recommendations=(),
        ).to_dict()
        self._emit(
            {
                "event": "session_terminated",
                "session_id": session_id,
                "reason": reason,
                "timestamp": utc_now(),
            }
        )
        return response

    @staticmethod
    def _request_id(session_id: str, turn: int, user_message: str) -> str:
        raw = f"{session_id}\0{turn}\0{user_message}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _error_info(stage: str, error: BaseException) -> dict[str, str]:
        return {"stage": stage, "type": type(error).__name__, "message": str(error)[:240]}

    def _emit_turn(
        self,
        *,
        session_id: str,
        request_id: str,
        turn: int,
        user_message: str,
        before: SessionState,
        after: SessionState,
        state_diff: Any,
        query: str,
        candidates: Sequence[Candidate],
        retrieval_total: int | None,
        ask_attribute: str | None,
        elapsed_ms: float,
        fallback_stages: Sequence[str],
        retry_counts: Mapping[str, int],
        errors: Sequence[Mapping[str, str]],
        done: bool,
    ) -> None:
        event = {
            "event": "turn",
            "timestamp": utc_now(),
            "session_id": session_id,
            "request_id": request_id,
            "message_digest": message_digest(user_message),
            "turn": turn,
            "intent_before": before.intent,
            "intent_after": after.intent,
            "state_diff": asdict(state_diff) if state_diff is not None else {},
            "query_digest": message_digest(query),
            "candidate_count": len(candidates),
            "retrieval_total": retrieval_total,
            "top_parent_asins": [candidate.parent_asin for candidate in candidates[:10]],
            "ask_attribute": ask_attribute,
            "latency_ms": elapsed_ms,
            "fallback_stages": list(fallback_stages),
            "retry_counts": dict(retry_counts),
            "errors": list(errors),
            "done": done,
            "state_version": after.version,
        }
        self._emit(event)

    def _emit(self, event: Mapping[str, Any]) -> None:
        try:
            self.trace_sink.emit(event)
        except Exception:
            # Observability must never break the scoring path.
            return None


__all__ = ["ShoppingOrchestrator"]
