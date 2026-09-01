"""Deterministic hard-constraint filtering for the canonical session state."""

from typing import Any, Iterable, Mapping

from shopping_copilot.contracts import Product, SessionState


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [str(value).strip().lower()]


def _text(product: Product) -> str:
    return " ".join(
        [
            product.title,
            *product.features,
            *product.description,
            *product.categories,
            product.store or "",
            " ".join(f"{key} {value}" for key, value in product.details.items()),
        ]
    ).lower()


class HardConstraintFilter:
    """Filter products using canonical SessionState or a constraint mapping."""

    def __init__(self) -> None:
        self._text_cache: dict[str, str] = {}

    def filter(
        self,
        products: Iterable[Product],
        state_or_constraints: SessionState | Mapping[str, Any],
    ) -> list[Product]:
        constraints = self._constraints(state_or_constraints)
        results: list[Product] = []
        for product in products:
            if not self._matches_positive(product, constraints):
                continue
            if self._matches_negative(product, constraints):
                continue
            results.append(product)
        return results

    @staticmethod
    def _constraints(
        value: SessionState | Mapping[str, Any],
    ) -> dict[str, Any]:

        if isinstance(value, SessionState):
            constraints = dict(value.hard_constraints)

            constraints["negative_constraints"] = dict(
                value.negative_constraints
            )

            constraints["negative_keywords"] = list(
                value.negative_constraints.get(
                    "negative_keywords",
                    [],
                ) or []
            )

        else:
            constraints = dict(value)

        # Canonical budget format:
        # {
        #     "budget": {
        #         "min": ...,
        #         "max": ...
        #     }
        # }
        #
        # Internally normalize it to price_min / price_max.
        budget = constraints.get("budget") or {}

        if isinstance(budget, Mapping):
            if budget.get("min") is not None:
                constraints["price_min"] = budget["min"]

            if budget.get("max") is not None:
                constraints["price_max"] = budget["max"]

        return constraints

    def _matches_positive(self, product: Product, constraints: Mapping[str, Any]) -> bool:
        searchable = self._searchable(product)
        for field in ("brand", "category", "color", "size", "material", "style", "feature", "use_case"):
            values = _values(constraints.get(field))
            if values and not any(value in searchable for value in values):
                return False

        minimum = constraints.get("price_min")
        maximum = constraints.get("price_max")
        if minimum is not None or maximum is not None:
            if product.price is None:
                return False
            if minimum is not None and product.price < float(minimum):
                return False
            if maximum is not None and product.price > float(maximum):
                return False
        return True

    def _matches_negative(self, product: Product, constraints: Mapping[str, Any]) -> bool:
        searchable = self._searchable(product)
        negative = constraints.get("negative_constraints")
        if isinstance(negative, Mapping):
            for field, value in negative.items():
                values = _values(value)
                if field == "negative_keywords":
                    if any(item in searchable for item in values):
                        return True
                elif any(item in searchable for item in values):
                    return True
        return any(item in searchable for item in _values(constraints.get("negative_keywords")))

    def _searchable(self, product: Product) -> str:
        cached = self._text_cache.get(product.parent_asin)
        if cached is None:
            cached = _text(product)
            self._text_cache[product.parent_asin] = cached
        return cached


__all__ = ["HardConstraintFilter"]
