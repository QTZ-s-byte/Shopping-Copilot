from data.catalog_loader import ProductCatalog
from retrieval.hybrid_retriever import HybridRetriever
from ranking.rule_ranker import RuleRanker


def print_results(title, ranked):

    print(f"\n=== {title} ===")

    for rank, candidate in enumerate(
        ranked[:10],
        start=1
    ):
        product = candidate.product

        print(
            f"{rank}. "
            f"{product.parent_asin} | "
            f"final={candidate.final_score:.4f} | "
            f"BM25={candidate.keyword_score:.4f} | "
            f"category={candidate.category_score:.2f} | "
            f"attribute={candidate.attribute_score:.2f} | "
            f"TFIDF={candidate.semantic_score:.4f} | "
            f"popularity={candidate.popularity_score:.2f} | "
            f"{product.store} | "
            f"{product.title}"
        )


def main():

    # -------------------------------------------------
    # Load catalog
    # -------------------------------------------------

    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    print(
        "Total products:",
        len(catalog.products)
    )

    # -------------------------------------------------
    # Retriever and Ranker
    # -------------------------------------------------

    retriever = HybridRetriever(catalog)
    ranker = RuleRanker()

    query = "black Nike running shoes"

    constraints = {
        "category": "running shoes",
        "brand": "Nike",
        "color": "black",
    }

    # -------------------------------------------------
    # Hard filter information
    # -------------------------------------------------

    filtered_products = retriever.hard_filter.filter(
        catalog.products.values(),
        constraints
    )

    print(
        "After hard filter:",
        len(filtered_products)
    )

    # -------------------------------------------------
    # Buying candidates
    # -------------------------------------------------

    buying_candidates = retriever.retrieve(
        query=query,
        constraints=constraints,
        top_k=100
    )

    print(
        "Buying candidates:",
        len(buying_candidates)
    )

    buying_ranked = ranker.rank(
        query=query,
        candidates=buying_candidates,
        constraints=constraints,
        intent="buying"
    )

    # -------------------------------------------------
    # Browsing candidates
    # -------------------------------------------------

    browsing_candidates = retriever.retrieve(
        query=query,
        constraints=constraints,
        top_k=100
    )

    print(
        "Browsing candidates:",
        len(browsing_candidates)
    )

    browsing_ranked = ranker.rank(
        query=query,
        candidates=browsing_candidates,
        constraints=constraints,
        intent="browsing"
    )

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print_results(
        "Buying Results",
        buying_ranked
    )

    print_results(
        "Browsing Results",
        browsing_ranked
    )


if __name__ == "__main__":
    main()