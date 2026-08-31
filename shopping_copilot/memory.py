"""Session-scoped context memory for the Shopping Copilot.

The competition evaluates one isolated session at a time.  This implementation
keeps only session state in memory and never writes to or mutates the product
catalog.  A storage protocol can be added later without changing the
orchestrator.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterator, Mapping

from .contracts import IntentResult, SessionState, StateDiff
from .contracts import ALLOWED_ATTRIBUTES


def _clean_value(value: Any) -> Any:
    """Make slot values safe and compact for state/logging purposes."""

    if isinstance(value, Mapping):
        return {str(key): _clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_value(item) for item in value]
    if value is None:
        return None
    text = str(value).strip()
    return text[:500]


def _remove_field(state: SessionState, field: str) -> dict[str, Any]:
    removed: dict[str, Any] = {}
    for container_name in (
        "hard_constraints",
        "soft_preferences",
        "negative_constraints",
    ):
        container = getattr(state, container_name)
        if field in container:
            removed[f"{container_name}.{field}"] = container.pop(field)
    return removed


def _set_field(state: SessionState, field: str, value: Any, preferred: str) -> None:
    """Set a slot in one constraint bucket and remove stale copies."""

    for container_name in (
        "hard_constraints",
        "soft_preferences",
        "negative_constraints",
    ):
        if container_name != preferred:
            getattr(state, container_name).pop(field, None)
    getattr(state, preferred)[field] = _clean_value(value)


@dataclass
class SessionRecord:
    state: SessionState
    request_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_diff: StateDiff | None = None
    lock: RLock = field(default_factory=RLock, repr=False)


class InMemoryContextMemory:
    """Thread-safe session memory with atomic turn updates.

    ``apply_turn`` is intentionally the only method that advances the turn.
    This prevents a failed retrieval/ranking stage from accidentally advancing
    the session twice.  The orchestrator can snapshot and restore a state when
    an unexpected error occurs.
    """

    def __init__(self, history_window: int = 3, max_sessions: int | None = None) -> None:
        self.history_window = max(1, int(history_window))
        self.max_sessions = None if max_sessions is None else max(1, int(max_sessions))
        self._records: dict[str, SessionRecord] = {}
        self._lock = RLock()

    def create(self, session_id: str, user_profile: Mapping[str, Any] | None = None) -> SessionState:
        session_id = self._validate_session_id(session_id)
        with self._lock:
            if self.max_sessions is not None and session_id not in self._records:
                while len(self._records) >= self.max_sessions:
                    oldest = next(iter(self._records))
                    self._records.pop(oldest, None)
            state = SessionState(session_id=session_id, user_profile=dict(user_profile or {}))
            self._records[session_id] = SessionRecord(state=state)
            return state.clone()

    def reset(self, session_id: str, user_profile: Mapping[str, Any] | None = None) -> SessionState:
        """Create/recreate a session, as required by the official Agent API."""

        return self.create(session_id, user_profile)

    def get(self, session_id: str) -> SessionState:
        record = self._record(session_id)
        with record.lock:
            return record.state.clone()

    def snapshot(self, session_id: str) -> SessionState:
        return self.get(session_id)

    def restore(self, session_id: str, snapshot: SessionState) -> None:
        record = self._record(session_id)
        with record.lock:
            if snapshot.session_id != session_id:
                raise ValueError("snapshot belongs to another session")
            record.state = snapshot.clone()

    @contextmanager
    def session_guard(self, session_id: str) -> Iterator[SessionRecord]:
        """Serialize turns for one session while allowing re-entrant helpers."""

        record = self._record(session_id)
        with record.lock:
            yield record

    def apply_turn(
        self,
        session_id: str,
        user_message: str,
        intent_result: IntentResult,
        *,
        expected_turn: int | None = None,
        max_turns: int = 10,
        asked_attribute: str | None = None,
    ) -> StateDiff:
        """Apply an intent/slot update and atomically advance the turn."""

        record = self._record(session_id)
        with record.lock:
            state = record.state
            if state.completed:
                raise RuntimeError("session is already completed")
            if state.turn_count >= max_turns:
                raise RuntimeError("session turn limit reached")
            next_turn = state.turn_count + 1
            if expected_turn is not None and int(expected_turn) != next_turn:
                raise ValueError(
                    f"unexpected turn {expected_turn}; expected {next_turn} for session {session_id}"
                )

            before_intent = state.intent
            removed: dict[str, Any] = {}
            replaced: dict[str, Any] = {}
            added: dict[str, Any] = {}

            # Explicit removals happen first, so a same-turn replacement wins.
            for field in intent_result.remove_fields:
                removed.update(_remove_field(state, str(field)))

            for field, value in intent_result.replace_fields.items():
                field = str(field)
                preferred = "hard_constraints" if field in intent_result.hard_constraints else "soft_preferences"
                for candidate_bucket in (
                    "hard_constraints",
                    "soft_preferences",
                    "negative_constraints",
                ):
                    if field in getattr(state, candidate_bucket):
                        preferred = candidate_bucket
                        break
                previous = _remove_field(state, field)
                if previous:
                    replaced.update(previous)
                _set_field(state, field, value, preferred)
                replaced[f"{preferred}.{field}"] = _clean_value(value)

            for bucket_name, preferred in (
                ("hard_constraints", "hard_constraints"),
                ("soft_preferences", "soft_preferences"),
                ("negative_constraints", "negative_constraints"),
            ):
                updates = getattr(intent_result, bucket_name)
                for field, value in updates.items():
                    field = str(field)
                    normalized = _clean_value(value)
                    existing = getattr(state, preferred).get(field)
                    _set_field(state, field, normalized, preferred)
                    if existing is None:
                        added[f"{preferred}.{field}"] = normalized
                    else:
                        replaced[f"{preferred}.{field}"] = normalized

            if intent_result.intent is not None:
                state.intent = intent_result.intent

            if asked_attribute in ALLOWED_ATTRIBUTES:
                state.asked_attributes.add(str(asked_attribute))
            for attribute in intent_result.no_preference_attributes:
                if attribute in ALLOWED_ATTRIBUTES:
                    state.no_preference_attributes.add(attribute)
            if intent_result.no_preference and not intent_result.no_preference_attributes:
                missing = [
                    name
                    for name in ("category", "use_case", "budget", "size", "color", "material", "brand", "feature")
                    if name in state.asked_attributes
                    and name not in state.hard_constraints
                    and name not in state.soft_preferences
                ]
                if missing:
                    state.no_preference_attributes.add(missing[-1])

            if user_message.strip():
                state.recent_messages.append(user_message.strip()[:2000])
                if len(state.recent_messages) > self.history_window:
                    state.recent_messages = state.recent_messages[-self.history_window :]

            state.turn_count = next_turn
            state.version += 1
            state.summary = self._build_summary(state)
            diff = StateDiff(
                added=added,
                removed=removed,
                replaced=replaced,
                intent_before=before_intent,
                intent_after=state.intent,
            )
            record.last_diff = diff
            return diff

    def set_candidates(self, session_id: str, parent_asins: list[str]) -> None:
        record = self._record(session_id)
        with record.lock:
            # Keep only a bounded, normalized list.  The catalog itself is not touched.
            result: list[str] = []
            seen: set[str] = set()
            for value in parent_asins:
                item = str(value).strip()
                if item and item not in seen:
                    result.append(item)
                    seen.add(item)
                if len(result) >= 100:
                    break
            record.state.last_candidates = result
            record.state.version += 1

    def mark_asked_attribute(self, session_id: str, attribute: str | None) -> None:
        if attribute not in ALLOWED_ATTRIBUTES:
            return
        record = self._record(session_id)
        with record.lock:
            record.state.asked_attributes.add(str(attribute))
            record.state.version += 1

    def mark_completed(self, session_id: str, reason: str) -> None:
        record = self._record(session_id)
        with record.lock:
            record.state.completed = True
            record.state.termination_reason = str(reason)[:200]
            record.state.version += 1

    def cache_response(self, session_id: str, request_id: str, response: Mapping[str, Any]) -> None:
        record = self._record(session_id)
        with record.lock:
            record.request_cache[str(request_id)] = deepcopy(dict(response))
            # A session cannot need an unbounded cache in the 10-turn protocol.
            if len(record.request_cache) > 16:
                oldest = next(iter(record.request_cache))
                record.request_cache.pop(oldest, None)

    def cached_response(self, session_id: str, request_id: str) -> dict[str, Any] | None:
        record = self._record(session_id)
        with record.lock:
            response = record.request_cache.get(str(request_id))
            return None if response is None else deepcopy(response)

    def last_diff(self, session_id: str) -> StateDiff | None:
        record = self._record(session_id)
        with record.lock:
            return record.last_diff

    def close(self, session_id: str) -> None:
        with self._lock:
            self._records.pop(str(session_id), None)

    def sessions(self) -> list[str]:
        with self._lock:
            return list(self._records)

    def _record(self, session_id: str) -> SessionRecord:
        session_id = self._validate_session_id(session_id)
        with self._lock:
            try:
                return self._records[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        value = str(session_id).strip()
        if not value:
            raise ValueError("session_id must not be empty")
        return value

    @staticmethod
    def _build_summary(state: SessionState) -> str:
        pieces: list[str] = []
        if state.intent:
            pieces.append(f"intent={state.intent}")
        for label, values in (
            ("hard", state.hard_constraints),
            ("soft", state.soft_preferences),
            ("negative", state.negative_constraints),
        ):
            if values:
                compact = ", ".join(f"{key}:{value}" for key, value in values.items())
                pieces.append(f"{label}=[{compact}]")
        return " | ".join(pieces)[:2000]
