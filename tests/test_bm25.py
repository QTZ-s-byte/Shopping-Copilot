from data.catalog_loader import ProductCatalog
from retrieval.bm25_retriever import BM25Retriever


def main():
    # 1. Load official catalog
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    print("Catalog size:", len(catalog.products))

    # 2. Build BM25
    retriever = BM25Retriever(catalog)

    print("Building BM25 index...")
    retriever.build_index()

    print("BM25 index built successfully.")

    # 3. Search
    query = "running shoes"

    results = retriever.search(
        query,
        top_k=10
    )

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")

    # 4. Print results
    for rank, (product, score) in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"{product.parent_asin} | "
            f"{score:.4f} | "
            f"{product.title}"
        )


if __name__ == "__main__":
    main()