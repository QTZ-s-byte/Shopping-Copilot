"""Official Track 4 entry point.

This wrapper intentionally keeps the evaluator-facing API tiny.  Teammate A or
B can inject richer router/retriever/ranker implementations into
``ShoppingOrchestrator`` without changing the public method signatures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shopping_copilot.catalog import SQLiteCatalogRetriever
from shopping_copilot.orchestrator import ShoppingOrchestrator
from shopping_copilot.policies import DefaultIntentRouter, ScoreRanker
from shopping_copilot.trace import NullTraceSink


class Agent:
    """Offline-safe Agent implementation for the official harness."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog = SQLiteCatalogRetriever(catalog_path)
        self.orchestrator = ShoppingOrchestrator(
            router=DefaultIntentRouter(),
            retriever=self.catalog,
            ranker=ScoreRanker(),
            valid_catalog_ids=self.catalog.valid_ids,
            trace_sink=NullTraceSink(),
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.orchestrator.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict[str, Any]:
        return self.orchestrator.respond(session_id, user_message, turn, top_k)

