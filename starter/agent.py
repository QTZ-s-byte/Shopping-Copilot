"""Official Track 4 entry point for the integrated Shopping Copilot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent.intent_router import IntentRouter
from ranking.rule_ranker import RuleRanker
from retrieval.hybrid_retriever import HybridRetriever
from shopping_copilot.catalog import ProductCatalog, SQLiteCatalogRetriever
from shopping_copilot.orchestrator import ShoppingOrchestrator
from shopping_copilot.policies import ScoreRanker
from shopping_copilot.trace import NullTraceSink


class Agent:
    """Wire Member A, Member B, and the C lifecycle into one Agent."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.fallback_catalog = None
        self.catalog = None
        self.using_fallback = False

        try:
            if os.getenv("SHOPPING_COPILOT_FORCE_FALLBACK", "0").lower() in {
                "1",
                "true",
                "yes",
            }:
                raise RuntimeError("fallback requested by environment")
            from retrieval.bm25_retriever import BM25Okapi
            from retrieval.semantic_retriever import TfidfVectorizer

            if BM25Okapi is None:
                raise RuntimeError("the BM25 dependency is unavailable")
            self.catalog = ProductCatalog(self.catalog_path)
            router = IntentRouter()
            use_semantic = os.getenv("SHOPPING_COPILOT_USE_SEMANTIC", "0").lower() in {
                "1",
                "true",
                "yes",
            }
            if use_semantic and TfidfVectorizer is None:
                raise RuntimeError("the semantic retrieval dependency is unavailable")
            retriever = HybridRetriever(self.catalog, use_semantic=use_semantic)
            ranker = RuleRanker()
        except Exception:
            # The official environment may not install optional scientific
            # packages. Keep the same Agent contract with the SQLite baseline.
            self.catalog = None
            self.using_fallback = True
            self.fallback_catalog = SQLiteCatalogRetriever(self.catalog_path)
            router = IntentRouter()
            retriever = self.fallback_catalog
            ranker = ScoreRanker()

        valid_ids = (
            self.catalog.valid_ids
            if self.catalog is not None
            else self.fallback_catalog.valid_ids
        )
        self.orchestrator = ShoppingOrchestrator(
            router=router,
            retriever=retriever,
            ranker=ranker,
            valid_catalog_ids=valid_ids,
            trace_sink=NullTraceSink(),
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.orchestrator.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict[str, Any]:
        return self.orchestrator.respond(session_id, user_message, turn, top_k)


__all__ = ["Agent"]
