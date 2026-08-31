from data.catalog_loader import ProductCatalog

from retrieval.bm25_retriever import BM25Retriever
from retrieval.hard_filter import HardConstraintFilter
from retrieval.semantic_retriever import TFIDFSemanticRetriever

from shopping_copilot.contracts import (
    Candidate,
    RetrievalResult,
    SessionState,
)


class HybridRetriever:
    """Canonical hybrid retriever used by the integrated Shopping Copilot."""

    def __init__(
        self,
        catalog: ProductCatalog,
        *,
        use_semantic: bool = False,
    ):
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

        self.semantic = None

        if self.use_semantic:
            self.semantic = TFIDFSemanticRetriever(catalog)
            self.semantic.build_index()

        # ---------------------------------------------
        # Hard constraints
        # ---------------------------------------------

        self.hard_filter = HardConstraintFilter()

    # =================================================
    # Canonical Retriever interface
    # =================================================

    def retrieve(
        self,
        query: str,
        state: SessionState,
        top_k: int,
    ) -> RetrievalResult:

        # ---------------------------------------------
        # Step 1: Build constraints from canonical state
        # ---------------------------------------------

        constraints = self._build_constraints(state)

        # ---------------------------------------------
        # Step 2: Hard filtering
        # ---------------------------------------------

        filtered_products = self.hard_filter.filter(
            self.catalog.products.values(),
            constraints,
        )

        total_count = len(filtered_products)

        if total_count == 0:
            return RetrievalResult(
                candidates=(),
                total_count=0,
                exhausted=True,
            )

        allowed_ids = {
            product.parent_asin
            for product in filtered_products
        }

        # ---------------------------------------------
        # Step 3: Retrieve a bounded candidate pool
        # ---------------------------------------------

        # The official response limit is 10, but retrieval may use
        # a larger internal pool before the final ranker.
        retrieval_pool = min(
            total_count,
            max(100, int(top_k) * 10),
        )

        # ---------------------------------------------
        # Step 4: BM25
        # ---------------------------------------------

        bm25_results = self.bm25.search(
            query=query,
            top_k=retrieval_pool,
            allowed_ids=allowed_ids,
        )

        bm25_scores = {
            product.parent_asin: score
            for product, score in bm25_results
        }

        # ---------------------------------------------
        # Step 5: Optional TF-IDF
        # ---------------------------------------------

        if self.use_semantic and self.semantic is not None:
            semantic_allowed_ids = (
                set(bm25_scores)
                if bm25_scores
                else allowed_ids
            )

            tfidf_results = self.semantic.search(
                query=query,
                top_k=retrieval_pool,
                allowed_ids=semantic_allowed_ids,
            )
        else:
            tfidf_results = []

        tfidf_scores = {
            product.parent_asin: score
            for product, score in tfidf_results
        }

        # ---------------------------------------------
        # Step 6: Normalize source scores
        # ---------------------------------------------

        normalized_bm25 = self._normalize(
            bm25_scores
        )

        normalized_tfidf = self._normalize(
            tfidf_scores
        )

        # ---------------------------------------------
        # Step 7: Build canonical Candidates
        # ---------------------------------------------

        candidate_ids = (
            set(bm25_scores)
            | set(tfidf_scores)
        )

        product_by_id = {
            product.parent_asin: product
            for product in filtered_products
            if product.parent_asin in candidate_ids
        }

        candidates: list[Candidate] = []

        for product_id, product in product_by_id.items():

            keyword_score = normalized_bm25.get(
                product_id,
                0.0,
            )

            semantic_score = normalized_tfidf.get(
                product_id,
                0.0,
            )

            preliminary_score = (
                0.6 * keyword_score
                + 0.4 * semantic_score
            )

            candidates.append(
                Candidate(
                    parent_asin=product_id,
                    product=product,
                    score=preliminary_score,
                    keyword_score=keyword_score,
                    semantic_score=semantic_score,
                    source_scores={
                        "bm25": keyword_score,
                        "tfidf": semantic_score,
                    },
                    reasons=(
                        "bm25",
                        "tfidf",
                    ),
                )
            )

        # ---------------------------------------------
        # Step 8: Preliminary ordering
        # ---------------------------------------------
        #
        # Final ranking is handled by RuleRanker.
        # This ordering only determines the retrieval pool.
        # ---------------------------------------------

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.parent_asin,
            )
        )

        limit = max(
            1,
            min(int(top_k), 100),
        )

        selected = candidates[:limit]

        return RetrievalResult(
            candidates=tuple(selected),
            total_count=total_count,
            exhausted=(total_count <= limit),
        )

    # =================================================
    # Build constraints from SessionState
    # =================================================

    @staticmethod
    def _build_constraints(
        state: SessionState,
    ) -> dict:

        # Only hard constraints enter the hard filter.
        constraints = dict(
            state.hard_constraints
        )

        # Negative constraints remain separate so that the hard
        # filter can exclude matching products explicitly.
        constraints["negative_constraints"] = dict(
            state.negative_constraints
        )

        constraints["negative_keywords"] = list(
            state.negative_constraints.get(
                "negative_keywords",
                [],
            ) or []
        )

        return constraints

    # =================================================
    # Score normalization
    # =================================================

    @staticmethod
    def _normalize(
        scores: dict[str, float],
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


__all__ = ["HybridRetriever"]