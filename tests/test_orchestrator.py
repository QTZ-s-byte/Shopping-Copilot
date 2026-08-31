from __future__ import annotations

import unittest
import time

from shopping_copilot.contracts import Candidate, IntentResult, RetrievalResult
from shopping_copilot.orchestrator import ShoppingOrchestrator
from shopping_copilot.policies import RetryPolicy
from shopping_copilot.trace import InMemoryTraceSink


class FakeRouter:
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, message, state):
        self.calls += 1
        if "broad" in message:
            return IntentResult(intent="browsing", clarification_attribute="category")
        return IntentResult(intent="buying", hard_constraints={"category": "shoes"})


class FakeRetriever:
    def __init__(self, *, broad: bool = False) -> None:
        self.calls = 0
        self.broad = broad

    def retrieve(self, query, state, top_k):
        self.calls += 1
        return RetrievalResult(
            candidates=(
                Candidate("A-1", score=0.8),
                Candidate("A-2", score=0.4),
                Candidate("INVALID", score=99),
            ),
            total_count=500 if self.broad else 2,
        )


class FakeRanker:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def rank(self, query, candidates, state):
        self.calls += 1
        if self.fail:
            raise RuntimeError("ranker unavailable")
        return list(reversed(candidates))


class FailingRouter:
    def classify(self, message, state):
        raise RuntimeError("router unavailable")


class SlowRetriever:
    def retrieve(self, query, state, top_k):
        time.sleep(0.05)
        return [Candidate("A-1", score=1.0)]


class OrchestratorTests(unittest.TestCase):
    def make(self, *, broad=False, rank_fail=False):
        router = FakeRouter()
        retriever = FakeRetriever(broad=broad)
        ranker = FakeRanker(fail=rank_fail)
        trace = InMemoryTraceSink()
        orchestrator = ShoppingOrchestrator(
            router=router,
            retriever=retriever,
            ranker=ranker,
            trace_sink=trace,
            valid_catalog_ids={"A-1", "A-2"},
            retry_policy=RetryPolicy(max_retries=0),
        )
        orchestrator.reset("session", {})
        return orchestrator, router, retriever, ranker, trace

    def test_end_to_end_filters_invalid_and_returns_contract(self) -> None:
        orchestrator, router, retriever, ranker, trace = self.make()
        response = orchestrator.respond("session", "I need shoes", 1, 10)
        self.assertEqual(response["ask_attribute"], None)
        self.assertEqual([x["parent_asin"] for x in response["recommendations"]], ["A-2", "A-1"])
        self.assertEqual(set(response), {"message", "ask_attribute", "recommendations", "usage"})
        self.assertEqual(router.calls, 1)
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(ranker.calls, 1)
        self.assertEqual(len(trace.events()), 2)  # reset + turn
        self.assertEqual(trace.events()[-1]["turn"], 1)

    def test_idempotent_duplicate_request_does_not_run_plugins_twice(self) -> None:
        orchestrator, router, retriever, ranker, _ = self.make()
        first = orchestrator.respond("session", "I need shoes", 1, 10)
        second = orchestrator.respond("session", "I need shoes", 1, 10)
        self.assertEqual(first, second)
        self.assertEqual(router.calls, 1)
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(ranker.calls, 1)

    def test_broad_query_asks_for_clarification_and_keeps_candidates(self) -> None:
        orchestrator, _, _, _, _ = self.make(broad=True)
        response = orchestrator.respond("session", "broad shopping ideas", 1, 10)
        self.assertEqual(response["ask_attribute"], "category")
        self.assertTrue(response["recommendations"])
        self.assertIn("many possible matches", response["message"])

    def test_ranker_failure_uses_deterministic_fallback(self) -> None:
        orchestrator, _, _, ranker, trace = self.make(rank_fail=True)
        response = orchestrator.respond("session", "I need shoes", 1, 10)
        self.assertEqual([x["parent_asin"] for x in response["recommendations"]], ["A-1", "A-2"])
        self.assertEqual(ranker.calls, 1)
        self.assertEqual(trace.events()[-1]["fallback_stages"], ["ranker"])

    def test_no_plugins_after_tenth_turn(self) -> None:
        orchestrator, router, retriever, ranker, _ = self.make()
        for turn in range(1, 11):
            response = orchestrator.respond("session", f"message {turn}", turn, 10)
            self.assertIsInstance(response["message"], str)
        calls = (router.calls, retriever.calls, ranker.calls)
        response = orchestrator.respond("session", "late message", 11, 10)
        self.assertEqual(response["recommendations"], [])
        self.assertEqual(calls, (router.calls, retriever.calls, ranker.calls))

    def test_invalid_top_k_is_rejected(self) -> None:
        orchestrator, *_ = self.make()
        with self.assertRaises(ValueError):
            orchestrator.respond("session", "query", 1, 5)

    def test_router_failure_advances_with_safe_fallback(self) -> None:
        trace = InMemoryTraceSink()
        orchestrator = ShoppingOrchestrator(
            router=FailingRouter(),
            retriever=FakeRetriever(),
            ranker=FakeRanker(),
            trace_sink=trace,
            valid_catalog_ids={"A-1", "A-2"},
            retry_policy=RetryPolicy(max_retries=0),
        )
        orchestrator.reset("session", {})
        response = orchestrator.respond("session", "ideas", 1, 10)
        self.assertEqual(response["ask_attribute"], "other")
        self.assertEqual(orchestrator.memory.get("session").turn_count, 1)
        self.assertEqual(trace.events()[-1]["fallback_stages"], ["router"])

    def test_retriever_timeout_returns_contract_safe_response(self) -> None:
        trace = InMemoryTraceSink()
        orchestrator = ShoppingOrchestrator(
            router=FakeRouter(),
            retriever=SlowRetriever(),
            ranker=FakeRanker(),
            trace_sink=trace,
            valid_catalog_ids={"A-1"},
            retry_policy=RetryPolicy(max_retries=0),
            stage_timeout_seconds=0.005,
        )
        orchestrator.reset("session", {})
        response = orchestrator.respond("session", "query", 1, 10)
        self.assertEqual(response["recommendations"], [])
        self.assertIn("retriever", trace.events()[-1]["fallback_stages"])


if __name__ == "__main__":
    unittest.main()
