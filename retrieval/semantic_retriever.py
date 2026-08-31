from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data.catalog_loader import Product, ProductCatalog


class TFIDFSemanticRetriever:

    def __init__(self, catalog: ProductCatalog):

        self.catalog = catalog

        self.products: List[Product] = []

        self.documents: List[str] = []

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        self.document_matrix = None

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

        self.document_matrix = (
            self.vectorizer.fit_transform(
                self.documents
            )
        )

    def score(
        self,
        query: str
    ) -> List[float]:

        if self.document_matrix is None:
            raise RuntimeError(
                "TF-IDF index has not been built. "
                "Call build_index() first."
            )

        query = query.strip()

        if not query:
            return [0.0] * len(self.products)

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        similarities = cosine_similarity(
            query_vector,
            self.document_matrix
        )[0]

        return similarities.tolist()

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