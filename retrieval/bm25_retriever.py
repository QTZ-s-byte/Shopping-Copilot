import re
import math
import heapq
from collections import Counter, defaultdict
from typing import List, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional compatibility export
    BM25Okapi = None

from data.catalog_loader import Product, ProductCatalog


class BM25Retriever:
    def __init__(self, catalog: ProductCatalog):
        self.catalog = catalog

        self.products: List[Product] = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._term_counts: list[Counter[str]] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._idf: dict[str, float] = {}
        self._document_lengths: list[int] = []
        self._average_length = 0.0
        self._id_to_index: dict[str, int] = {}

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

        self._id_to_index = {
            product.parent_asin: index
            for index, product in enumerate(self.products)
        }

        self.tokenized_corpus = []

        for product in self.products:
            text = self._product_to_text(product)
            tokens = self._tokenize(text)

            self.tokenized_corpus.append(tokens)
        self._term_counts = []
        self._postings = defaultdict(list)
        self._document_lengths = []
        for index, tokens in enumerate(self.tokenized_corpus):
            counts = Counter(tokens)
            self._term_counts.append(counts)
            self._document_lengths.append(len(tokens))
            for token in counts:
                self._postings[token].append(index)
        self._average_length = sum(self._document_lengths) / max(1, len(self._document_lengths))
        document_count = len(self.tokenized_corpus)
        self._idf = {
            token: math.log(1.0 + (document_count - len(indices) + 0.5) / (len(indices) + 0.5))
            for token, indices in self._postings.items()
        }
        # The custom scorer avoids allocating a 50,000-element score array on
        # every turn while retaining the standard BM25 ranking formula.
        self.bm25 = None

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

        query_terms = set(query_tokens)

        if allowed_ids is not None:
            allowed_indices = {
                self._id_to_index[parent_asin]
                for parent_asin in allowed_ids
                if parent_asin in self._id_to_index
            }

            candidate_indices = {
                index
                for token in query_terms
                for index in self._postings.get(token, ())
                if index in allowed_indices
            }
        else:
            candidate_indices = {
                index
                for token in query_terms
                for index in self._postings.get(token, ())
            }
        k1, b = 1.5, 0.75
        scored: list[tuple[int, float]] = []
        for index in candidate_indices:
            counts = self._term_counts[index]
            length = self._document_lengths[index]
            score = 0.0
            for token in query_terms:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + k1 * (
                    1.0 - b + b * length / max(1.0, self._average_length)
                )
                score += self._idf.get(token, 0.0) * frequency * (k1 + 1.0) / denominator
            scored.append((index, score))
        ranked_indices = [index for index, _ in heapq.nlargest(max(0, int(top_k)), scored, key=lambda item: item[1])]
        score_by_index = dict(scored)

        results = []

        for index in ranked_indices:

            product = self.products[index]

            results.append(
                (
                    product,
                    float(score_by_index[index])
                )
            )

        return results
