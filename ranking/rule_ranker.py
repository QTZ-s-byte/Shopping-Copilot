from dataclasses import replace
import math
from typing import Sequence

from shopping_copilot.contracts import (
    Candidate,
    Product,
    SessionState,
)


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
        candidates: Sequence[Candidate],
        state: SessionState,
    ) -> Sequence[Candidate]:

        if not candidates:
            return []

        constraints = dict(state.hard_constraints)

        # Soft preferences are available for ranking,
        # but do not overwrite explicit hard constraints.
        for key, value in state.soft_preferences.items():
            if key not in constraints:
                constraints[key] = value

        if state.intent == "buying":
            weights = self.BUYING_WEIGHTS

        elif state.intent == "browsing":
            weights = self.BROWSING_WEIGHTS

        else:
            raise ValueError(
                f"Unsupported intent: {state.intent}"
            )

        ranked_candidates: list[Candidate] = []

        for candidate in candidates:

            product = candidate.product

            # Candidate should normally contain Product.
            if product is None:
                continue

            category_score = self._category_score(
                product,
                constraints,
            )

            attribute_score = self._attribute_score(
                product,
                constraints,
            )

            popularity_score = self._popularity_score(
                product,
            )

            final_score = (
                weights["keyword"] * candidate.keyword_score
                + weights["category"] * category_score
                + weights["attribute"] * attribute_score
                + weights["semantic"] * candidate.semantic_score
                + weights["popularity"] * popularity_score
            )

            new_candidate = replace(
                candidate,
                score=final_score,
                category_score=category_score,
                attribute_score=attribute_score,
                popularity_score=popularity_score,
                final_score=final_score,
                source_scores={
                    **dict(candidate.source_scores),
                    "bm25": candidate.keyword_score,
                    "tfidf": candidate.semantic_score,
                    "category": category_score,
                    "attribute": attribute_score,
                    "popularity": popularity_score,
                    "final": final_score,
                },
                reasons=tuple(
                    dict.fromkeys(
                        (
                            *candidate.reasons,
                            "rule_ranked",
                        )
                    )
                ),
            )

            ranked_candidates.append(new_candidate)

        ranked_candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.parent_asin,
            )
        )

        return ranked_candidates

    # =====================================================
    # Category score
    # =====================================================

    def _category_score(
        self,
        product: Product,
        constraints: dict,
    ) -> float:

        category = constraints.get("category")

        if not category:
            return 0.0

        category = str(category).lower().strip()

        category_text = " ".join(
            product.categories
        ).lower()

        if category in category_text:
            return 1.0

        title = product.title.lower()

        if category in title:
            return 0.9

        query_words = set(category.split())

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
        constraints: dict,
    ) -> float:

        scores: list[float] = []

        # -------------------------------------------------
        # Brand
        # -------------------------------------------------

        brand = constraints.get("brand")

        if brand:
            evidence = " ".join(
                (
                    product.store or "",
                    str(
                        product.details.get(
                            "Manufacturer",
                            "",
                        )
                    ),
                    product.title,
                )
            ).lower()

            if str(brand).lower() in evidence:
                scores.append(1.0)
            else:
                scores.append(0.0)

        # -------------------------------------------------
        # Color
        # -------------------------------------------------

        color = constraints.get("color")

        if color:
            text = " ".join(
                (
                    product.title,
                    *product.features,
                    *product.description,
                )
            ).lower()

            if str(color).lower() in text:
                scores.append(1.0)
            else:
                scores.append(0.0)

        # -------------------------------------------------
        # Material
        # -------------------------------------------------

        material = constraints.get("material")

        if material:
            text = " ".join(
                (
                    product.title,
                    *product.features,
                    *product.description,
                )
            ).lower()

            if str(material).lower() in text:
                scores.append(1.0)
            else:
                scores.append(0.0)

        # -------------------------------------------------
        # Size
        # -------------------------------------------------

        size = constraints.get("size")

        if size:
            text = " ".join(
                (
                    product.title,
                    *product.features,
                    *product.description,
                )
            ).lower()

            if str(size).lower() in text:
                scores.append(1.0)
            else:
                scores.append(0.0)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    # =====================================================
    # Popularity score
    # =====================================================

    def _popularity_score(
        self,
        product: Product,
    ) -> float:

        rating_score = 0.0

        if product.average_rating is not None:
            rating_score = min(
                max(
                    product.average_rating / 5.0,
                    0.0,
                ),
                1.0,
            )

        review_score = 0.0

        if product.rating_number > 0:
            review_score = min(
                math.log1p(
                    product.rating_number
                ) / 12.0,
                1.0,
            )

        return (
            0.6 * rating_score
            + 0.4 * review_score
        )


__all__ = ["RuleRanker"]