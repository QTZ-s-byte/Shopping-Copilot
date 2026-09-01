from shopping_copilot.catalog import ProductCatalog
from shopping_copilot.contracts import SessionState
from retrieval.hybrid_retriever import HybridRetriever


def main():

    # =================================================
    # Load catalog
    # =================================================

    catalog = ProductCatalog(
        "data/catalog.jsonl"
    )

    # =================================================
    # Create retriever
    # =================================================

    retriever = HybridRetriever(
        catalog,
        use_semantic=False,
    )

    query = "running shoes"

    # =================================================
    # Canonical hard constraints
    # =================================================

    constraints = {
        "category": ["running shoes"],
        "budget": {
            "min": None,
            "max": 100,
        },
    }

    # =================================================
    # Build canonical session state
    # =================================================

    state = SessionState(
        session_id="test",
        intent="buying",
        hard_constraints=constraints,
    )

    # =================================================
    # Hard-filter diagnostic
    # =================================================

    filtered_products = retriever.hard_filter.filter(
        catalog.products.values(),
        state,
    )

    print(
        "Total products:",
        len(catalog.products),
    )

    print(
        "After hard filter:",
        len(filtered_products),
    )

    # =================================================
    # Hybrid retrieval
    # =================================================

    results = retriever.retrieve(
        query=query,
        state=state,
        top_k=10,
    )

    print(
        "Result type:",
        type(results).__name__,
    )

    print(
        "Retrieved candidates:",
        len(results.candidates),
    )

    print(
        "Total count:",
        results.total_count,
    )

    print(
        "Exhausted:",
        results.exhausted,
    )

    print("\n=== Hybrid Results ===")

    for rank, candidate in enumerate(
        results.candidates,
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