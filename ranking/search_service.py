from data.catalog_loader import ProductCatalog
from ranking.candidate import Candidate
from ranking.rule_ranker import RuleRanker
from retrieval.hybrid_retriever import HybridRetriever
from shopping_copilot.contracts import SessionState


class ProductSearchService:

    def __init__(self, catalog: ProductCatalog):

        self.retriever = HybridRetriever(catalog)
        self.ranker = RuleRanker()

    def search(
        self,
        query: str,
        state: SessionState | None = None,
        top_k: int = 10,
        constraints: dict | None = None,
        intent: str | None = None,
    ) -> list[Candidate]:
        if state is None:
            state = SessionState(
                hard_constraints=dict(constraints or {}),
                intent=intent or "buying",
            )

        if top_k <= 0:
            return []

        if top_k > 100:
            top_k = 100

        candidates = self.retriever.retrieve(
            query=query,
            state=state,
            top_k=100
        )

        ranked_candidates = self.ranker.rank(
            query=query,
            candidates=list(candidates),
            state=state,
        )

        return ranked_candidates[:top_k]
