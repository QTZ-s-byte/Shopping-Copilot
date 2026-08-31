import math
import re
from collections import Counter
from typing import List, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:  # pragma: no cover - exercised in dependency-free runs
    TfidfVectorizer = None
    cosine_similarity = None
    np = None

from data.catalog_loader import Product, ProductCatalog


class TFIDFSemanticRetriever:

    def __init__(self, catalog: ProductCatalog):

        self.catalog = catalog

        self.products: List[Product] = []

        self.documents: List[str] = []

        self.vectorizer = (
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 1),
                min_df=3,
                max_df=0.9,
                max_features=20000,
                dtype=np.float32 if np is not None else float,
            )
            if TfidfVectorizer
            else None
        )

        self.document_matrix = None
        self._fallback_documents: list[Counter[str]] = []
        self._fallback_idf: dict[str, float] = {}

    def _product_to_text(
        self,
        product: Product
    ) -> str:

        parts = [
            product.title,
            " ".join(product.features),
            " ".join(product.description),
            " ".join(product.categories),
            product.store or "",
        ]

        return " ".join(
            part for part in parts
            if part
        )

    def build_index(self) -> None:

        self.products = list(
            self.catalog.products.values()
        )

        self.documents = [
            self._product_to_text(product)
            for product in self.products
        ]

        if self.vectorizer is not None:
            try:
                self.document_matrix = self.vectorizer.fit_transform(self.documents)
            except ValueError as exc:
                if "max_df corresponds" not in str(exc):
                    raise
                self.vectorizer = TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=1.0,
                    max_features=20000,
                    dtype=np.float32 if np is not None else float,
                )
                self.document_matrix = self.vectorizer.fit_transform(self.documents)
            return
        self._fallback_documents = [Counter(self._tokens(document)) for document in self.documents]
        document_frequency: Counter[str] = Counter()
        for document in self._fallback_documents:
            document_frequency.update(document.keys())
        count = max(1, len(self._fallback_documents))
        self._fallback_idf = {
            token: math.log((1 + count) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }

    def score(
        self,
        query: str
    ) -> List[float]:

        if self.document_matrix is None and not self._fallback_documents:
            raise RuntimeError(
                "TF-IDF index has not been built. "
                "Call build_index() first."
            )

        query = query.strip()

        if not query:
            return [0.0] * len(self.products)

        if self.vectorizer is not None:
            query_vector = self.vectorizer.transform([query])
            return cosine_similarity(query_vector, self.document_matrix)[0].tolist()
        query_counts = Counter(self._tokens(query))
        query_weighted = {
            token: count * self._fallback_idf.get(token, 1.0)
            for token, count in query_counts.items()
        }
        query_norm = math.sqrt(sum(value * value for value in query_weighted.values()))
        similarities: list[float] = []
        for document in self._fallback_documents:
            weighted = {
                token: count * self._fallback_idf.get(token, 1.0)
                for token, count in document.items()
            }
            norm = math.sqrt(sum(value * value for value in weighted.values()))
            dot = sum(query_weighted.get(token, 0.0) * value for token, value in weighted.items())
            similarities.append(dot / (query_norm * norm) if query_norm and norm else 0.0)
        return similarities

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(
        self,
        query: str,
        top_k: int = 100,
        allowed_ids: set[str] | None = None
    ) -> List[Tuple[Product, float]]:

        similarities = self.score(query)

        ranked_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            product = self.products[index]

            if allowed_ids is not None:

                if product.parent_asin not in allowed_ids:
                    continue

            results.append(
                (
                    product,
                    float(similarities[index])
                )
            )

            if len(results) >= top_k:
                break

        return results
