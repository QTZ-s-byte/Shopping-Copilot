import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Product:
    parent_asin: str
    title: str
    features: list[str]
    description: list[str]
    price: Optional[float]
    categories: list[str]
    details: dict
    average_rating: Optional[float]
    rating_number: int
    store: Optional[str]


def normalize_price(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)

        if match:
            return float(match.group())

    return None


class ProductCatalog:
    def __init__(self):
        self.products: dict[str, Product] = {}

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                data = json.loads(line)

                product = Product(
                    parent_asin=data["parent_asin"],
                    title=data.get("title", ""),
                    features=data.get("features", []),
                    description=data.get("description", []),
                    price=normalize_price(data.get("price")),
                    categories=data.get("categories", []),
                    details=data.get("details", {}),
                    average_rating=data.get("average_rating"),
                    rating_number=data.get("rating_number", 0),
                    store=data.get("store"),
                )

                self.products[product.parent_asin] = product

    def get(self, product_id: str) -> Optional[Product]:
        return self.products.get(product_id)

    def search(self, query: str) -> list[Product]:
        query = query.lower().strip()

        if not query:
            return []

        results = []

        for product in self.products.values():
            text = " ".join([
                product.title,
                *product.features,
                *product.description,
                *product.categories,
                product.store or "",
            ]).lower()

            if query in text:
                results.append(product)

        return results
