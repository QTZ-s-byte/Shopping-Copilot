import math

from data.catalog_loader import Product
from ranking.candidate import Candidate
from shopping_copilot.contracts import SessionState


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
        state: SessionState | None = None,
        constraints: dict | None = None,
        intent: str | None = None,
    ) -> list[Candidate]:

        if not candidates:
            return []

        if state is not None:
            constraints = dict(state.hard_constraints)
            constraints.update(state.soft_preferences)
            intent = state.intent or "browsing"
        constraints = constraints or {}
        intent = intent or "buying"

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
            candidate.score = candidate.final_score
            candidate.source_scores = {
                "keyword": candidate.keyword_score,
                "category": candidate.category_score,
                "attribute": candidate.attribute_score,
                "semantic": candidate.semantic_score,
                "popularity": candidate.popularity_score,
            }

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

        category = str(category[-1] if isinstance(category, list) and category else category).lower().strip()

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

            brands = brand if isinstance(brand, (list, tuple, set)) else [brand]
            if any(str(item).lower() in evidence for item in brands):
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

            colors = color if isinstance(color, (list, tuple, set)) else [color]
            if any(str(item).lower() in text for item in colors):
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

            materials = material if isinstance(material, (list, tuple, set)) else [material]
            if any(str(item).lower() in text for item in materials):
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

            sizes = size if isinstance(size, (list, tuple, set)) else [size]
            if any(str(item).lower() in text for item in sizes):
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
