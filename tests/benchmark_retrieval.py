import time

from shopping_copilot.catalog import (
    ProductCatalog,
    SQLiteCatalogRetriever,
)
from shopping_copilot.contracts import SessionState
from retrieval.hard_filter import HardConstraintFilter
from retrieval.hybrid_retriever import HybridRetriever


CATALOG_PATH = "data/catalog.jsonl"


# 每个 query 使用与它匹配的 session state。
# 这样 benchmark 才有意义。
BENCHMARKS = [
    (
        "running shoes",
        SessionState(
            session_id="benchmark_1",
            intent="buying",
            hard_constraints={
                "category": ["running shoes"],
            },
        ),
    ),
    (
        "black Nike running shoes",
        SessionState(
            session_id="benchmark_2",
            intent="buying",
            hard_constraints={
                "category": ["running shoes"],
                "brand": ["nike"],
                "color": ["black"],
            },
        ),
    ),
    (
        "waterproof hiking shoes",
        SessionState(
            session_id="benchmark_3",
            intent="buying",
            hard_constraints={
                "category": ["shoes"],
                "feature": ["waterproof"],
                "use_case": ["hiking"],
            },
        ),
    ),
    (
        "women running shoes under $100",
        SessionState(
            session_id="benchmark_4",
            intent="buying",
            hard_constraints={
                "category": ["running shoes"],
                "budget": {
                    "min": None,
                    "max": 100,
                },
            },
        ),
    ),
    (
        "leather handbag",
        SessionState(
            session_id="benchmark_5",
            intent="buying",
            hard_constraints={
                "category": ["bags"],
                "material": ["leather"],
            },
        ),
    ),
]


def measure_query(
    name,
    retriever,
    query,
    state,
    top_k,
):
    start = time.perf_counter()

    result = retriever.retrieve(
        query=query,
        state=state,
        top_k=top_k,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    if hasattr(result, "candidates"):
        count = len(result.candidates)
        total_count = result.total_count
    else:
        count = len(result)
        total_count = None

    print(
        f"{name:<25} "
        f"{query:<35} "
        f"{elapsed_ms:>9.2f} ms "
        f"candidates={count}"
        + (
            f" total={total_count}"
            if total_count is not None
            else ""
        )
    )


def main():
    print("=== Retrieval Benchmark ===\n")

    # ==================================================
    # Catalog
    # ==================================================

    catalog = ProductCatalog(
        CATALOG_PATH
    )

    print(
        f"Catalog size: {len(catalog.products)}"
    )

    # ==================================================
    # Retrievers
    # ==================================================

    sqlite = SQLiteCatalogRetriever(
        CATALOG_PATH
    )

    hybrid_bm25 = HybridRetriever(
        catalog,
        use_semantic=False,
    )

    hybrid_full = HybridRetriever(
        catalog,
        use_semantic=True,
    )

    # ==================================================
    # Hard filter
    # ==================================================

    filter_engine = HardConstraintFilter()

    print("\n=== Hard Filter Diagnostics ===")

    for query, state in BENCHMARKS:
        filtered = filter_engine.filter(
            catalog.products.values(),
            state,
        )

        print(
            f"{query:<35} -> "
            f"{len(filtered)} products"
        )

    # ==================================================
    # Direct state check
    # ==================================================

    print("\n=== Direct Hybrid State Check ===")

    query, state = BENCHMARKS[1]

    filtered = filter_engine.filter(
        catalog.products.values(),
        state,
    )

    print(
        "Query:",
        query
    )

    print(
        "Hard-filter count:",
        len(filtered)
    )

    direct_result = hybrid_bm25.retrieve(
        query=query,
        state=state,
        top_k=100,
    )

    print(
        "Hybrid result type:",
        type(direct_result).__name__,
    )

    print(
        "Hybrid candidate count:",
        len(direct_result.candidates),
    )

    print(
        "Hybrid total count:",
        direct_result.total_count,
    )

    print("\nFirst Hybrid candidates:")

    for candidate in direct_result.candidates[:10]:
        product = candidate.product

        print(
            candidate.parent_asin,
            "|",
            getattr(product, "price", None),
            "|",
            getattr(product, "store", None),
            "|",
            getattr(product, "title", ""),
        )

    # ==================================================
    # Query latency benchmark
    # ==================================================

    print("\n=== Query Latency Benchmark ===\n")

    print(
        f"{'Path':<25} "
        f"{'Query':<35} "
        f"{'Latency':>12} "
        f"Candidates"
    )

    print("-" * 100)

    for query, state in BENCHMARKS:

        measure_query(
            "SQLite FTS5",
            sqlite,
            query,
            state,
            top_k=10,
        )

        measure_query(
            "Hybrid BM25",
            hybrid_bm25,
            query,
            state,
            top_k=100,
        )

        measure_query(
            "Hybrid BM25 + TF-IDF",
            hybrid_full,
            query,
            state,
            top_k=100,
        )

        print()


if __name__ == "__main__":
    main()