from data.catalog_loader import ProductCatalog
from ranking.candidate import Candidate
from ranking.rule_ranker import RuleRanker
from retrieval.hybrid_retriever import HybridRetriever


class ProductSearchService:

    def __init__(self, catalog: ProductCatalog):

        self.retriever = HybridRetriever(catalog)
        self.ranker = RuleRanker()

    def search(
        self,
        query: str,
        constraints: dict,
        intent: str,
        top_k: int = 10
    ) -> list[Candidate]:

        if top_k <= 0:
            return []

        if top_k > 100:
            top_k = 100

        candidates = self.retriever.retrieve(
            query=query,
            constraints=constraints,
            top_k=100
        )

        ranked_candidates = self.ranker.rank(
            query=query,
            candidates=candidates,
            constraints=constraints,
            intent=intent
        )

        return ranked_candidates[:top_k]