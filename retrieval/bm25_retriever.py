import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from data.catalog_loader import Product, ProductCatalog


class BM25Retriever:
    def __init__(self, catalog: ProductCatalog):
        self.catalog = catalog

        self.products: List[Product] = []
        self.tokenized_corpus = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Convert text into lowercase tokens.
        """
        text = text.lower()

        tokens = re.findall(
            r"[a-z0-9]+(?:\.[0-9]+)?",
            text
        )

        return tokens

    def _product_to_text(self, product: Product) -> str:
        """
        Convert a product into searchable text.
        """

        parts = [
            product.title,
            " ".join(product.features),
            " ".join(product.description),
            " ".join(product.categories),
            product.store or "",
        ]

        return " ".join(parts)

    def build_index(self) -> None:
        """
        Build an in-memory BM25 index from the official catalog.
        """

        self.products = list(
            self.catalog.products.values()
        )

        self.tokenized_corpus = []

        for product in self.products:
            text = self._product_to_text(product)
            tokens = self._tokenize(text)

            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    def search(
        self,
        query: str,
        top_k: int = 100,
        allowed_ids: set[str] | None = None
    ) -> List[Tuple[Product, float]]:
        """
        Search the BM25 index.

        If allowed_ids is provided, only products whose
        parent_asin is in allowed_ids will be returned.
        """

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built. "
                "Call build_index() first."
            )

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            product = self.products[index]

            # Apply optional hard-constraint ID filter
            if allowed_ids is not None:
                if product.parent_asin not in allowed_ids:
                    continue

            results.append(
                (
                    product,
                    float(scores[index])
                )
            )

            if len(results) >= top_k:
                break

        return results