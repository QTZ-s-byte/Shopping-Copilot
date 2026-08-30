from dataclasses import dataclass

from data.catalog_loader import Product


@dataclass
class Candidate:
    product: Product

    keyword_score: float = 0.0
    category_score: float = 0.0
    attribute_score: float = 0.0
    semantic_score: float = 0.0
    popularity_score: float = 0.0

    final_score: float = 0.0