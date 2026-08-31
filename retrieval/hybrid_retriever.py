from data.catalog_loader import ProductCatalog
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hard_filter import HardConstraintFilter
from retrieval.semantic_retriever import TFIDFSemanticRetriever
from ranking.candidate import Candidate
from shopping_copilot.contracts import RetrievalResult, SessionState

# Preliminary ordering for candidate selection only.
# Final ranking is handled by RuleRanker.

class HybridRetriever:

    def __init__(self, catalog: ProductCatalog, *, use_semantic: bool = False):
        self.use_semantic = bool(use_semantic)

        self.catalog = catalog

        # ---------------------------------------------
        # BM25
        # ---------------------------------------------

        self.bm25 = BM25Retriever(catalog)
        self.bm25.build_index()

        # ---------------------------------------------
        # TF-IDF semantic retrieval
        # ---------------------------------------------

        self.semantic = TFIDFSemanticRetriever(catalog) if self.use_semantic else None
        if self.semantic is not None:
            self.semantic.build_index()

        # ---------------------------------------------
        # Hard constraints
        # ---------------------------------------------

        self.hard_filter = HardConstraintFilter()

    def retrieve(
        self,
        query: str,
        state: SessionState | None = None,
        top_k: int = 100,
        constraints: dict | None = None,
    ) -> RetrievalResult:
        if state is None:
            state = SessionState(
                hard_constraints=dict(constraints or {}),
                intent="buying",
            )

        # ---------------------------------------------
        # Step 1: Hard constraint filtering
        # ---------------------------------------------

        filtered_products = self.hard_filter.filter(
            self.catalog.products.values(),
            state,
        )

        total_count = len(filtered_products)

        allowed_ids = {
            product.parent_asin
            for product in filtered_products
        }

        if not allowed_ids:
            return RetrievalResult(candidates=(), total_count=0, exhausted=True)

        # ---------------------------------------------
        # Step 2: BM25
        # ---------------------------------------------

        retrieval_pool = min(len(allowed_ids), max(100, int(top_k) * 10))
        bm25_results = self.bm25.search(
            query=query,
            top_k=retrieval_pool,
            allowed_ids=allowed_ids
        )

        # ---------------------------------------------
        # Step 3: TF-IDF
        # ---------------------------------------------

        bm25_scores = {
            product.parent_asin: score
            for product, score in bm25_results
        }

        # Re-rank the BM25 pool instead of materializing a full-catalog
        # semantic ranking for every turn. This keeps latency bounded.
        tfidf_results = (
            self.semantic.search(
                query=query,
                top_k=retrieval_pool,
                allowed_ids=set(bm25_scores) if bm25_scores else allowed_ids,
            )
            if self.semantic is not None
            else []
        )

        # ---------------------------------------------
        # Step 4: Convert scores to dictionaries
        # ---------------------------------------------

        tfidf_scores = {
            product.parent_asin: score
            for product, score in tfidf_results
        }

        # ---------------------------------------------
        # Step 5: Normalize scores
        # ---------------------------------------------

        normalized_bm25 = self._normalize(
            bm25_scores
        )

        normalized_tfidf = self._normalize(
            tfidf_scores
        )

        # ---------------------------------------------
        # Step 6: Build Candidates
        # ---------------------------------------------

        candidate_map = {}

        candidate_ids = set(bm25_scores) | set(tfidf_scores)
        for product in filtered_products:

            product_id = product.parent_asin
            if product_id not in candidate_ids:
                continue

            keyword_score = normalized_bm25.get(
                product_id,
                0.0
            )

            semantic_score = normalized_tfidf.get(
                product_id,
                0.0
            )

            candidate = Candidate(
                product=product,
                keyword_score=keyword_score,
                semantic_score=semantic_score,
                score=0.6 * keyword_score + 0.4 * semantic_score,
                source_scores={"keyword": keyword_score, "semantic": semantic_score},
                reasons=("bm25", "tfidf"),
            )

            candidate_map[product_id] = candidate


        # ---------------------------------------------
        # Step 7: Temporary candidate ordering
        # ---------------------------------------------

        ranked_candidates = sorted(
            candidate_map.values(),
            key=lambda c: (
                0.6 * c.keyword_score
                + 0.4 * c.semantic_score
            ),
            reverse=True
        )

        return RetrievalResult(
            candidates=tuple(ranked_candidates[:top_k]),
            total_count=total_count,
            exhausted=len(ranked_candidates) <= top_k,
        )

    @staticmethod
    def _normalize(
        scores: dict[str, float]
    ) -> dict[str, float]:

        if not scores:
            return {}

        values = list(scores.values())

        min_score = min(values)
        max_score = max(values)

        if max_score == min_score:

            return {
                product_id: 1.0
                for product_id in scores
            }

        return {
            product_id: (
                (score - min_score)
                / (max_score - min_score)
            )
            for product_id, score in scores.items()
        }
