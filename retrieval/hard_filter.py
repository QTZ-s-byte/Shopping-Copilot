from typing import Iterable

from data.catalog_loader import Product


class HardConstraintFilter:

    def filter(
        self,
        products: Iterable[Product],
        constraints: dict
    ) -> list[Product]:

        results = []

        for product in products:

            # -------------------------
            # Brand
            # -------------------------
            if constraints.get("brand"):
                if not self._match_brand(
                    product,
                    constraints["brand"]
                ):
                    continue

            # -------------------------
            # Category
            # -------------------------
            if constraints.get("category"):
                if not self._match_category(
                    product,
                    constraints["category"]
                ):
                    continue

            # -------------------------
            # Color
            # -------------------------
            if constraints.get("color"):
                if not self._match_color(
                    product,
                    constraints["color"]
                ):
                    continue

            # -------------------------
            # Price minimum
            # -------------------------
            if constraints.get("price_min") is not None:

                if product.price is not None:

                    if product.price < constraints["price_min"]:
                        continue

            # -------------------------
            # Price maximum
            # -------------------------
            if constraints.get("price_max") is not None:

                if product.price is not None:

                    if product.price > constraints["price_max"]:
                        continue

            # -------------------------
            # Negative keywords
            # -------------------------
            negative_keywords = constraints.get(
                "negative_keywords",
                []
            )

            if self._contains_negative_keyword(
                product,
                negative_keywords
            ):
                continue

            results.append(product)

        return results

    # =====================================================
    # Brand
    # =====================================================

    def _match_brand(
        self,
        product: Product,
        brand: str
    ) -> bool:

        brand = brand.lower().strip()

        evidence = []

        # Store
        if product.store:
            evidence.append(product.store)

        # Manufacturer
        manufacturer = product.details.get(
            "Manufacturer"
        )

        if manufacturer:
            evidence.append(manufacturer)

        # Title
        evidence.append(product.title)

        for text in evidence:

            if brand in text.lower():
                return True

        return False

    # =====================================================
    # Category
    # =====================================================

    def _match_category(
        self,
        product: Product,
        category: str
    ) -> bool:

        category = category.lower().strip()

        # First check official category hierarchy
        for item in product.categories:

            if category in item.lower():
                return True

        # Then check title
        if category in product.title.lower():
            return True

        return False

    # =====================================================
    # Color
    # =====================================================

    def _match_color(
        self,
        product: Product,
        color: str
    ) -> bool:

        color = color.lower().strip()

        text_parts = [
            product.title,
            *product.features,
            *product.description,
        ]

        text = " ".join(text_parts).lower()

        return color in text

    # =====================================================
    # Negative keywords
    # =====================================================

    def _contains_negative_keyword(
        self,
        product: Product,
        negative_keywords: list[str]
    ) -> bool:

        if not negative_keywords:
            return False

        text_parts = [
            product.title,
            *product.features,
            *product.description,
            *product.categories,
        ]

        text = " ".join(text_parts).lower()

        for keyword in negative_keywords:

            if keyword.lower() in text:
                return True

        return False