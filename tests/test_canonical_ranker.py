from data.catalog_loader import ProductCatalog
from retrieval.hybrid_retriever import HybridRetriever
from ranking.rule_ranker import RuleRanker
from shopping_copilot.contracts import SessionState


def main():

    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(catalog)
    ranker = RuleRanker()

    state = SessionState(
        session_id="test_rank_001",
        intent="buying",
        hard_constraints={
            "category": "running shoes",
            "brand": "Nike",
            "color": "black",
        },
    )

    retrieval_result = retriever.retrieve(
        query="black Nike running shoes",
        state=state,
        top_k=100,
    )

    print(
        "Retrieved:",
        len(retrieval_result.candidates)
    )

    ranked = ranker.rank(
        query="black Nike running shoes",
        candidates=retrieval_result.candidates,
        state=state,
    )

    print(
        "Ranked:",
        len(ranked)
    )

    print("\n=== Canonical Ranker Results ===")

    for rank, candidate in enumerate(
        ranked[:10],
        start=1,
    ):

        product = candidate.product

        print(
            f"{rank}. "
            f"{candidate.parent_asin} | "
            f"score={candidate.score:.4f} | "
            f"BM25={candidate.keyword_score:.4f} | "
            f"category={candidate.category_score:.2f} | "
            f"attribute={candidate.attribute_score:.2f} | "
            f"TFIDF={candidate.semantic_score:.4f} | "
            f"popularity={candidate.popularity_score:.2f} | "
            f"{product.store} | "
            f"{product.title}"
        )


if __name__ == "__main__":
    main()