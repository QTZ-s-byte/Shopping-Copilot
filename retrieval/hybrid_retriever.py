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

    def __init__(
        self,
        catalog: ProductCatalog,
    ):
        self.catalog = catalog

        # ---------------------------------------------
        # BM25
        # ---------------------------------------------

        self.bm25 = BM25Retriever(catalog)
        self.bm25.build_index()

        # ---------------------------------------------
        # TF-IDF semantic retrieval
        # ---------------------------------------------

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
        # Step 3: BM25
        # ---------------------------------------------

        bm25_results = self.bm25.search(
            query=query,
            top_k=total_count,
            allowed_ids=allowed_ids,
        )

        # ---------------------------------------------
        # Step 4: TF-IDF
        # ---------------------------------------------

        tfidf_results = self.semantic.search(
            query=query,
            top_k=total_count,
            allowed_ids=allowed_ids,
        )

        # ---------------------------------------------
        # Step 5: Convert scores
        # ---------------------------------------------

        bm25_scores = {
            product.parent_asin: score
            for product, score in bm25_results
        }

        tfidf_scores = {
            product.parent_asin: score
            for product, score in tfidf_results
        }

        # ---------------------------------------------
        # Step 6: Normalize
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

        candidates = []

        for product in filtered_products:

            product_id = product.parent_asin

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

            candidate = Candidate(
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
                    "keyword_match",
                    "semantic_match",
                ),
            )

            candidates.append(candidate)

        # ---------------------------------------------
        # Step 8: Preliminary candidate ordering
        # ---------------------------------------------
        #
        # This is NOT final ranking.
        # RuleRanker performs final ranking.
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

        constraints = dict(
            state.hard_constraints
        )

        # Soft preferences are available to retrieval
        # when they are not already hard constraints.
        for key, value in state.soft_preferences.items():

            if key not in constraints:
                constraints[key] = value

        # Negative constraints
        negative_keywords = (
            state.negative_constraints.get(
                "keywords",
                [],
            )
        )

        constraints["negative_keywords"] = list(
            negative_keywords
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