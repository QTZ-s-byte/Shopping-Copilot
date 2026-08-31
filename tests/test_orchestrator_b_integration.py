from data.catalog_loader import ProductCatalog
from retrieval.hybrid_retriever import HybridRetriever
from ranking.rule_ranker import RuleRanker
from shopping_copilot.orchestrator import ShoppingOrchestrator


def main():

    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(catalog)
    ranker = RuleRanker()

    orchestrator = ShoppingOrchestrator(
        retriever=retriever,
        ranker=ranker,
        valid_catalog_ids=catalog.valid_ids,
    )

    session_id = "integration_001"

    orchestrator.reset(
        session_id,
        {}
    )

    response = orchestrator.respond(
        session_id=session_id,
        user_message="I need running shoes",
        turn=1,
        top_k=10,
    )

    print("=== Orchestrator + B Integration ===")
    print(response)


if __name__ == "__main__":
    main()