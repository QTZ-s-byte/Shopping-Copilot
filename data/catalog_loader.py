"""Canonical catalog import surface for retrieval and ranking components.

Product and ProductCatalog are defined in ``shopping_copilot.catalog``. This
module re-exports them so member-owned retrieval files share one domain model.
"""

from shopping_copilot.catalog import Product, ProductCatalog, _normalize_price as normalize_price

__all__ = ["Product", "ProductCatalog", "normalize_price"]
