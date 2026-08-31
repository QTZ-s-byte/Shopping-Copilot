from data.catalog_loader import ProductCatalog
from retrieval.hybrid_retriever import HybridRetriever
from ranking.rule_ranker import RuleRanker


def main():

    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(catalog)

    ranker = RuleRanker()

    query = "running shoes"

    constraints = {
        "category": "running shoes",
        "price_max": 100,
    }

    # ---------------------------------------------
    # Retrieval
    # ---------------------------------------------

    candidates = retriever.retrieve(
        query=query,
        constraints=constraints,
        top_k=100
    )

    print(
        "Retrieved candidates:",
        len(candidates)
    )

    # ---------------------------------------------
    # Ranking
    # ---------------------------------------------

    ranked = ranker.rank(
        query=query,
        candidates=candidates,
        constraints=constraints
    )

    print("\n=== Ranked Results ===")

    for rank, candidate in enumerate(
        ranked[:10],
        start=1
    ):

        product = candidate.product

        print(
            f"{rank}. "
            f"{product.parent_asin} | "
            f"final={candidate.final_score:.4f} | "
            f"keyword={candidate.keyword_score:.4f} | "
            f"category={candidate.category_score:.2f} | "
            f"attribute={candidate.attribute_score:.2f} | "
            f"semantic={candidate.semantic_score:.2f} | "
            f"popularity={candidate.popularity_score:.2f} | "
            f"{product.title}"
        )


if __name__ == "__main__":
    main()