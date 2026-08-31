"""Small, dependency-free trace sinks for local development and judging.

Trace events intentionally contain metadata and hashes rather than raw secrets
or unbounded conversation text.  The JSONL sink is suitable for attaching to a
reproducible run; the in-memory sink is convenient for tests.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def message_digest(message: str) -> str:
    """Return a stable, non-reversible identifier for a user message."""

    return hashlib.sha256(str(message).encode("utf-8")).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def redact(value: Any, *, max_length: int = 500) -> Any:
    """Recursively remove obvious secret fields and bound trace payload size."""

    secret_names = {
        "api_key",
        "access_token",
        "refresh_token",
        "secret",
        "client_secret",
        "password",
        "authorization",
        "credential",
        "credentials",
    }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in secret_names:
                result[key_text] = "[REDACTED]"
            else:
                result[key_text] = redact(item, max_length=max_length)
        return result
    if isinstance(value, (list, tuple, set)):
        return [redact(item, max_length=max_length) for item in list(value)[:100]]
    if isinstance(value, str):
        return value[:max_length]
    return value


class NullTraceSink:
    """A no-op sink for production-like runs where trace output is disabled."""

    def emit(self, event: Mapping[str, Any]) -> None:
        return None


class InMemoryTraceSink:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            self._events.append(redact(dict(event)))

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._events]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class JSONLTraceSink:
    """Append one bounded JSON object per event.

    The parent directory is created lazily.  File writes are serialized so the
    sink remains safe when different sessions are evaluated concurrently.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def emit(self, event: Mapping[str, Any]) -> None:
        payload = redact(dict(event))
        payload.setdefault("timestamp", utc_now())
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
