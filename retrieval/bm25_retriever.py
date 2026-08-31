import re
import math
import heapq
from typing import List, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised in dependency-free runs
    BM25Okapi = None

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

        self.bm25 = BM25Okapi(self.tokenized_corpus) if BM25Okapi else None

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

        if not self.products:
            raise RuntimeError(
                "BM25 index has not been built or the catalog is empty. "
                "Call build_index() first."
            )

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        if self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
        else:
            query_set = set(query_tokens)
            document_frequency = {
                token: sum(token in document for document in self.tokenized_corpus)
                for token in query_set
            }
            scores = []
            for document in self.tokenized_corpus:
                overlap = set(document) & query_set
                score = sum(
                    math.log((1 + len(self.tokenized_corpus)) / (1 + document_frequency[token]))
                    for token in overlap
                )
                scores.append(float(score))

        eligible = (
            range(len(scores))
            if allowed_ids is None
            else (
                index
                for index, product in enumerate(self.products)
                if product.parent_asin in allowed_ids
            )
        )
        ranked_indices = heapq.nlargest(
            max(0, int(top_k)), eligible, key=lambda index: float(scores[index])
        )

        results = []

        for index in ranked_indices:

            product = self.products[index]

            results.append(
                (
                    product,
                    float(scores[index])
                )
            )

        return results
