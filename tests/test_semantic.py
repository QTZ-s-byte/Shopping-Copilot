from data.catalog_loader import ProductCatalog
from retrieval.semantic_retriever import TFIDFSemanticRetriever


def main():

    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    semantic = TFIDFSemanticRetriever(
        catalog
    )

    print(
        "Building TF-IDF index..."
    )

    semantic.build_index()

    print(
        "TF-IDF index built successfully."
    )

    query = "running shoes"

    results = semantic.search(
        query,
        top_k=10
    )

    print(
        f"\nQuery: {query}"
    )

    print(
        f"Results: {len(results)}"
    )

    for rank, (product, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{score:.4f} | "
            f"{product.parent_asin} | "
            f"{product.store} | "
            f"{product.title}"
        )


if __name__ == "__main__":
    main()