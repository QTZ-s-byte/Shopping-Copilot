from data.catalog_loader import ProductCatalog
from retrieval.hard_filter import HardConstraintFilter


def print_results(name, results):

    print(f"\n=== {name} ===")
    print("Count:", len(results))

    for product in results[:10]:

        print(
            product.parent_asin,
            "|",
            product.price,
            "|",
            product.store,
            "|",
            product.title
        )


def main():

    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    print("Total products:", len(catalog.products))

    filter_engine = HardConstraintFilter()

    # --------------------------------------------------
    # Test 1: Category + Price
    # --------------------------------------------------

    constraints = {
        "category": "running shoes",
        "price_max": 100,
    }

    results = filter_engine.filter(
        catalog.products.values(),
        constraints
    )

    print_results(
        "Running shoes under $100",
        results
    )

    # --------------------------------------------------
    # Test 2: Brand + Category + Price
    # --------------------------------------------------

    constraints = {
        "brand": "Nike",
        "category": "running shoes",
        "price_max": 100,
    }

    results = filter_engine.filter(
        catalog.products.values(),
        constraints
    )

    print_results(
        "Nike running shoes under $100",
        results
    )

    # --------------------------------------------------
    # Test 3: Color + Category
    # --------------------------------------------------

    constraints = {
        "category": "running shoes",
        "color": "black",
    }

    results = filter_engine.filter(
        catalog.products.values(),
        constraints
    )

    print_results(
        "Black running shoes",
        results
    )

    # --------------------------------------------------
    # Test 4: Negative keyword
    # --------------------------------------------------

    constraints = {
        "category": "running shoes",
        "negative_keywords": [
            "trail"
        ]
    }

    results = filter_engine.filter(
        catalog.products.values(),
        constraints
    )

    print_results(
        "Running shoes excluding trail",
        results
    )


if __name__ == "__main__":
    main()