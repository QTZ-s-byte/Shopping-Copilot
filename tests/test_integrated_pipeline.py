from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.intent_router import IntentRouter
from ranking.rule_ranker import RuleRanker
from retrieval.hybrid_retriever import HybridRetriever
from shopping_copilot.catalog import ProductCatalog
from shopping_copilot.orchestrator import ShoppingOrchestrator
from shopping_copilot.policies import EmptyRetriever, ScoreRanker


class IntegratedPipelineTests(unittest.TestCase):
    def test_member_a_and_b_run_through_c_orchestrator(self) -> None:
        rows = [
            {
                "parent_asin": "A",
                "title": "Black running shoes",
                "features": ["lightweight", "breathable"],
                "description": ["running shoes for outdoor training"],
                "price": 50,
                "categories": ["Clothing, Shoes & Jewelry", "Shoes"],
                "details": {"Manufacturer": "Nike"},
                "average_rating": 4.5,
                "rating_number": 100,
                "store": "Nike",
            },
            {
                "parent_asin": "B",
                "title": "Red leather bag",
                "features": ["durable"],
                "description": ["travel bag"],
                "price": 80,
                "categories": ["Clothing, Shoes & Jewelry", "Bags"],
                "details": {"Manufacturer": "Coach"},
                "average_rating": 4.3,
                "rating_number": 80,
                "store": "Coach",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            catalog = ProductCatalog(path)
            orchestrator = ShoppingOrchestrator(
                router=IntentRouter(),
                retriever=HybridRetriever(catalog),
                ranker=RuleRanker(),
                valid_catalog_ids=catalog.valid_ids,
            )
            orchestrator.reset("integration", {})
            response = orchestrator.respond(
                "integration", "I need black running shoes", 1, 10
            )
            self.assertEqual(response["recommendations"][0]["parent_asin"], "A")
            self.assertEqual(orchestrator.memory.get("integration").intent, "buying")

    def test_override_replaces_the_previous_goal_in_shared_memory(self) -> None:
        orchestrator = ShoppingOrchestrator(
            router=IntentRouter(),
            retriever=EmptyRetriever(),
            ranker=ScoreRanker(),
        )
        orchestrator.reset("override", {})
        orchestrator.respond("override", "I need black running shoes", 1, 10)
        orchestrator.respond("override", "Actually, I want a leather bag", 2, 10)
        state = orchestrator.memory.get("override")
        self.assertEqual(state.hard_constraints["category"], ["bags"])
        self.assertEqual(state.hard_constraints["material"], ["leather"])
        self.assertNotIn("running shoes", state.hard_constraints.get("category", []))


if __name__ == "__main__":
    unittest.main()
