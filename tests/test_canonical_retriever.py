from data.catalog_loader import ProductCatalog
from retrieval.hybrid_retriever import HybridRetriever
from shopping_copilot.contracts import (
    RetrievalResult,
    SessionState,
)


def main():

    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(
        catalog
    )

    state = SessionState(
        session_id="test_001",
        intent="buying",
        hard_constraints={
            "category": "running shoes",
            "price_max": 100,
        },
    )

    result = retriever.retrieve(
        query="running shoes",
        state=state,
        top_k=100,
    )

    print(
        "Result type:",
        type(result).__name__
    )

    print(
        "Total count:",
        result.total_count
    )

    print(
        "Candidate count:",
        len(result.candidates)
    )

    print(
        "Exhausted:",
        result.exhausted
    )

    print("\n=== Top Candidates ===")

    for rank, candidate in enumerate(
        result.candidates[:10],
        start=1,
    ):

        product = candidate.product

        print(
            f"{rank}. "
            f"{candidate.parent_asin} | "
            f"score={candidate.score:.4f} | "
            f"BM25={candidate.keyword_score:.4f} | "
            f"TFIDF={candidate.semantic_score:.4f} | "
            f"{product.store} | "
            f"{product.title}"
        )


if __name__ == "__main__":
    main()