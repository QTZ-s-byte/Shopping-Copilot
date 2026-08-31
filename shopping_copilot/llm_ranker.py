"""Optional DeepSeek semantic reranker.

The adapter is disabled by default. It uses the official OpenAI-compatible
Chat Completions endpoint and keeps the offline evaluator independent of any
external model service.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import defaultdict
from threading import RLock
from typing import Any, Sequence

from .contracts import Candidate, SessionState


class LLMRanker:
    """Reorder candidates with the DeepSeek Chat Completions API."""

    def __init__(self, fallback: Any) -> None:
        self.fallback = fallback
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "SHOPPING_COPILOT_ENABLE_LLM=1 requires DEEPSEEK_API_KEY"
            )
        self.base_url = os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/chat/completions",
        ).strip()
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
        self.timeout_seconds = max(
            1.0, float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "30"))
        )
        self.max_calls_per_session = max(
            1, int(os.getenv("DEEPSEEK_MAX_CALLS_PER_SESSION", "1"))
        )
        self._session_calls: dict[str, int] = defaultdict(int)
        self._pending_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._lock = RLock()

    def rank(
        self, query: str, candidates: Sequence[Candidate], state: SessionState
    ) -> list[Candidate]:
        baseline = list(self.fallback.rank(query, candidates, state))
        if not baseline:
            return baseline
        session_id = state.session_id or "unknown-session"
        with self._lock:
            if self._session_calls[session_id] >= self.max_calls_per_session:
                return baseline
            self._session_calls[session_id] += 1

        payload = {
            "model": self.model,
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": 512,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rank the candidate products for the shopping request. Return only "
                        "a JSON object with one key named ordered_ids whose value is an "
                        "array of product IDs. Use only IDs present in the candidate list."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "candidates": [self._candidate_view(item) for item in baseline[:20]],
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage") or {}
            with self._lock:
                self._pending_usage["prompt_tokens"] += max(
                    0, int(usage.get("prompt_tokens", 0))
                )
                self._pending_usage["completion_tokens"] += max(
                    0, int(usage.get("completion_tokens", 0))
                )
            ordered_ids = self._parse_ids(content)
        except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return baseline

        by_id = {item.parent_asin: item for item in baseline}
        reordered = [by_id[parent_asin] for parent_asin in ordered_ids if parent_asin in by_id]
        reordered_ids = {item.parent_asin for item in reordered}
        reordered.extend(item for item in baseline if item.parent_asin not in reordered_ids)
        return reordered

    def consume_usage(self) -> dict[str, int]:
        """Return and clear token usage accumulated since the previous turn."""

        with self._lock:
            usage = dict(self._pending_usage)
            self._pending_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        return usage

    @staticmethod
    def _candidate_view(candidate: Candidate) -> dict[str, Any]:
        product = candidate.product
        return {
            "parent_asin": candidate.parent_asin,
            "title": getattr(product, "title", ""),
            "categories": list(getattr(product, "categories", ()) or ()),
            "features": list(getattr(product, "features", ()) or ())[:4],
            "price": getattr(product, "price", None),
        }

    @staticmethod
    def _parse_ids(content: Any) -> list[str]:
        if isinstance(content, dict):
            values = content.get("ordered_ids", [])
        elif isinstance(content, list):
            values = content
        else:
            text = str(content)
            object_match = re.search(r"\{[\s\S]*\}", text)
            array_match = re.search(r"\[[\s\S]*\]", text)
            if object_match:
                parsed = json.loads(object_match.group(0))
                values = parsed.get("ordered_ids", []) if isinstance(parsed, dict) else []
            elif array_match:
                values = json.loads(array_match.group(0))
            else:
                return []
        if not isinstance(values, list):
            return []
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


__all__ = ["LLMRanker"]
