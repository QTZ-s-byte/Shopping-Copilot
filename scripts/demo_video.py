"""Deterministic, non-interactive demo for the submission video."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("SHOPPING_COPILOT_RETRIEVAL_MODE", "hybrid")
os.environ.setdefault("SHOPPING_COPILOT_ENABLE_BM25", "1")
os.environ.setdefault("SHOPPING_COPILOT_ENABLE_TFIDF", "0")

from agent.intent_router import IntentRouter  # noqa: E402
from agent.types import SessionState  # noqa: E402
from shopping_copilot.catalog import ProductCatalog  # noqa: E402
from starter.agent import Agent  # noqa: E402


PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.7,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort", "durability"],
    "summary": "Prior purchases emphasize fit, comfort, durability.",
}


def _title(catalog: ProductCatalog, asin: str) -> str:
    product = catalog.get(asin)
    text = product.title if product is not None else asin
    return str(text).encode("ascii", "ignore").decode("ascii")


def main() -> None:
    catalog_path = ROOT / "data" / "catalog.jsonl"
    catalog = ProductCatalog(catalog_path)
    agent = Agent(catalog_path)
    router = IntentRouter()

    print("Shopping Copilot - end-to-end demo")
    print(f"Catalog products: {len(catalog.valid_ids):,}")
    print(
        "Retrieval: "
        + ("hybrid (BM25)" if not agent.using_fallback else "SQLite fallback")
    )
    print()

    session_id = "demo-video"
    agent.reset(session_id, PROFILE)

    turns = [
        "I need running shoes",
        "I want it from Nike",
        "Actually, I want a leather bag instead",
        "Show me some summer outfit ideas",
    ]

    for turn, message in enumerate(turns, start=1):
        classification = router.classify(message, SessionState())
        response = agent.respond(session_id, message, turn, 10)

        print(f"Turn {turn}: {message}")
        print(f"  intent          : {classification.intent}")
        print(f"  hard constraints: {classification.hard_constraints}")
        print(f"  assistant       : {response.get('message', '')}")
        ask = response.get("ask_attribute")
        if ask:
            print(f"  asks about      : {ask}")
        recommendations = response.get("recommendations") or []
        print(f"  recommendations ({len(recommendations)}):")
        for index, item in enumerate(recommendations[:5], start=1):
            asin = item.get("parent_asin", "")
            score = item.get("score")
            score_text = f"  score={score:.4f}" if isinstance(score, (int, float)) else ""
            print(f"    {index}. {asin}  {_title(catalog, asin)[:76]}{score_text}")
        print()


if __name__ == "__main__":
    main()
