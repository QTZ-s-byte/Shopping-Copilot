from data.catalog_loader import ProductCatalog
from retrieval.hybrid_retriever import HybridRetriever


def main():

    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(catalog)

    query = "running shoes"

    constraints = {
        "category": "running shoes",
        "price_max": 100,
    }

    filtered_products = retriever.hard_filter.filter(
        catalog.products.values(),
        constraints
    )

    print(
        "Total products:",
        len(catalog.products)
    )

    print(
        "After hard filter:",
        len(filtered_products)
    )

    results = retriever.retrieve(
        query=query,
        constraints=constraints,
        top_k=10
    )

    print(
        "Retrieved candidates:",
        len(results)
    )

    print("\n=== Hybrid Results ===")

    for rank, candidate in enumerate(
        results,
        start=1
    ):

        product = candidate.product

        print(
            f"{rank}. "
            f"{product.parent_asin} | "
            f"final={candidate.final_score:.4f} | "
            f"BM25={candidate.keyword_score:.4f} | "
            f"TFIDF={candidate.semantic_score:.4f} | "
            f"{product.store} | "
            f"{product.title}"
        )


if __name__ == "__main__":
    main()