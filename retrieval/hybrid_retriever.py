from data.catalog_loader import ProductCatalog
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hard_filter import HardConstraintFilter
from retrieval.semantic_retriever import TFIDFSemanticRetriever
from ranking.candidate import Candidate

# Preliminary ordering for candidate selection only.
# Final ranking is handled by RuleRanker.

class HybridRetriever:

    def __init__(self, catalog: ProductCatalog):

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

    def retrieve(
        self,
        query: str,
        constraints: dict,
        top_k: int = 100
    ) -> list[Candidate]:

        # ---------------------------------------------
        # Step 1: Hard constraint filtering
        # ---------------------------------------------

        filtered_products = self.hard_filter.filter(
            self.catalog.products.values(),
            constraints
        )

        allowed_ids = {
            product.parent_asin
            for product in filtered_products
        }

        if not allowed_ids:
            return []

        # ---------------------------------------------
        # Step 2: BM25
        # ---------------------------------------------

        bm25_results = self.bm25.search(
            query=query,
            top_k=len(allowed_ids),
            allowed_ids=allowed_ids
        )

        # ---------------------------------------------
        # Step 3: TF-IDF
        # ---------------------------------------------

        tfidf_results = self.semantic.search(
            query=query,
            top_k=len(allowed_ids),
            allowed_ids=allowed_ids
        )

        # ---------------------------------------------
        # Step 4: Convert scores to dictionaries
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

        for product in filtered_products:

            product_id = product.parent_asin

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
                semantic_score=semantic_score
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

        return ranked_candidates[:top_k]

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