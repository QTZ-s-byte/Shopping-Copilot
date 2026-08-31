"""Optional OpenAI-compatible semantic reranker.

The adapter is disabled by default. It is intentionally small and dependency
free so the offline evaluator never requires an external model service.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Sequence

from .contracts import Candidate, SessionState


class LLMRanker:
    """Reorder candidates with an OpenAI-compatible chat-completions API."""

    def __init__(self, fallback: Any, *, timeout_seconds: float = 8.0) -> None:
        self.fallback = fallback
        self.api_key = os.getenv("SHOPPING_COPILOT_LLM_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "SHOPPING_COPILOT_ENABLE_LLM=1 requires SHOPPING_COPILOT_LLM_API_KEY"
            )
        self.base_url = os.getenv(
            "SHOPPING_COPILOT_LLM_BASE_URL",
            "https://api.openai.com/v1/chat/completions",
        ).strip()
        self.model = os.getenv("SHOPPING_COPILOT_LLM_MODEL", "gpt-4o-mini").strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def rank(
        self, query: str, candidates: Sequence[Candidate], state: SessionState
    ) -> list[Candidate]:
        baseline = list(self.fallback.rank(query, candidates, state))
        if not baseline:
            return baseline
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only a JSON array of product IDs ordered by relevance. "
                        "Use only IDs present in the candidate list."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "candidates": [self._candidate_view(item) for item in baseline[:40]],
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
            ordered_ids = self._parse_ids(content)
        except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return baseline

        by_id = {item.parent_asin: item for item in baseline}
        reordered = [by_id[parent_asin] for parent_asin in ordered_ids if parent_asin in by_id]
        reordered.extend(item for item in baseline if item.parent_asin not in {x.parent_asin for x in reordered})
        return reordered

    @staticmethod
    def _candidate_view(candidate: Candidate) -> dict[str, Any]:
        product = candidate.product
        return {
            "parent_asin": candidate.parent_asin,
            "title": getattr(product, "title", ""),
            "categories": list(getattr(product, "categories", ()) or ()),
            "features": list(getattr(product, "features", ()) or ())[:6],
            "price": getattr(product, "price", None),
        }

    @staticmethod
    def _parse_ids(content: Any) -> list[str]:
        if isinstance(content, list):
            values = content
        else:
            text = str(content)
            match = re.search(r"\[[\s\S]*\]", text)
            if not match:
                return []
            values = json.loads(match.group(0))
        if not isinstance(values, list):
            return []
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


__all__ = ["LLMRanker"]
