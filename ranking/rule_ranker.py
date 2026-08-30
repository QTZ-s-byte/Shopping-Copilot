import math

from data.catalog_loader import Product
from ranking.candidate import Candidate


class RuleRanker:
    BUYING_WEIGHTS = {
    "keyword": 0.35,
    "category": 0.25,
    "attribute": 0.20,
    "semantic": 0.15,
    "popularity": 0.05,
}

    BROWSING_WEIGHTS = {
    "keyword": 0.25,
    "category": 0.20,
    "attribute": 0.15,
    "semantic": 0.30,
    "popularity": 0.10,
}

    def rank(
        self,
        query: str,
        candidates: list[Candidate],
        constraints: dict | None = None,
        intent: str = "buying"
    ) -> list[Candidate]:

        if not candidates:
            return []

        constraints = constraints or {}

        # -------------------------------------------------
        # Calculate all ranking features
        # -------------------------------------------------

        for candidate in candidates:

            product = candidate.product

            candidate.category_score = (
                self._category_score(
                    product,
                    constraints
                )
            )

            candidate.attribute_score = (
                self._attribute_score(
                    product,
                    constraints
                )
            )

            candidate.popularity_score = (
                self._popularity_score(
                    product
                )
            )

            if intent == "browsing":
                weights = self.BROWSING_WEIGHTS
            elif intent == "buying":
                    weights = self.BUYING_WEIGHTS
            else:
                raise ValueError(
                    f"Unsupported intent: {intent}"
                )

            # keyword_score:
            # already produced by BM25
            #
            # semantic_score:
            # already produced by TF-IDF

            candidate.final_score = (
                weights["keyword"] * candidate.keyword_score
                + weights["category"] * candidate.category_score
                + weights["attribute"] * candidate.attribute_score
                + weights["semantic"] * candidate.semantic_score
                + weights["popularity"] * candidate.popularity_score
            )

        # -------------------------------------------------
        # Sort by final score
        # -------------------------------------------------

        candidates.sort(
            key=lambda c: (
                c.final_score,
                c.keyword_score,
                c.semantic_score
            ),
            reverse=True
        )

        return candidates

    # =====================================================
    # Category score
    # =====================================================

    def _category_score(
        self,
        product: Product,
        constraints: dict
    ) -> float:

        category = constraints.get("category")

        if not category:
            return 0.0

        category = category.lower().strip()

        # Exact/substring match in official category path
        category_text = " ".join(
            product.categories
        ).lower()

        if category in category_text:
            return 1.0

        # Exact phrase in title
        title = product.title.lower()

        if category in title:
            return 0.9

        # Partial word overlap
        query_words = set(
            category.split()
        )

        if not query_words:
            return 0.0

        matched = sum(
            1
            for word in query_words
            if word in title
        )

        return 0.6 * (
            matched / len(query_words)
        )

    # =====================================================
    # Attribute score
    # =====================================================

    def _attribute_score(
        self,
        product: Product,
        constraints: dict
    ) -> float:

        scores = []

        # -------------------------------------------------
        # Brand
        # -------------------------------------------------

        brand = constraints.get("brand")

        if brand:
            evidence = " ".join([
                product.store or "",
                product.details.get(
                    "Manufacturer",
                    ""
                ),
                product.title
            ]).lower()

            if brand.lower() in evidence:
                scores.append(1.0)
            else:
                scores.append(0.0)

        # -------------------------------------------------
        # Color
        # -------------------------------------------------

        color = constraints.get("color")

        if color:

            text = " ".join([
                product.title,
                *product.features,
                *product.description
            ]).lower()

            if color.lower() in text:
                scores.append(1.0)
            else:
                scores.append(0.0)

        # -------------------------------------------------
        # Material
        # -------------------------------------------------

        material = constraints.get("material")

        if material:

            text = " ".join([
                product.title,
                *product.features,
                *product.description
            ]).lower()

            if material.lower() in text:
                scores.append(1.0)
            else:
                scores.append(0.0)

        # -------------------------------------------------
        # Size
        # -------------------------------------------------

        size = constraints.get("size")

        if size:

            text = " ".join([
                product.title,
                *product.features,
                *product.description
            ]).lower()

            if size.lower() in text:
                scores.append(1.0)
            else:
                scores.append(0.0)

        # No attributes requested
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    # =====================================================
    # Popularity score
    # =====================================================

    def _popularity_score(
        self,
        product: Product
    ) -> float:

        rating_score = 0.0

        if product.average_rating is not None:
            rating_score = min(
                max(
                    product.average_rating / 5.0,
                    0.0
                ),
                1.0
            )

        review_score = 0.0

        if product.rating_number > 0:
            review_score = min(
                math.log1p(
                    product.rating_number
                ) / 12.0,
                1.0
            )

        # Give rating slightly more importance
        # than review volume.
        return (
            0.6 * rating_score
            + 0.4 * review_score
        )