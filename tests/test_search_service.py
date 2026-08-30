from data.catalog_loader import ProductCatalog
from ranking.search_service import ProductSearchService


def main():

    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    service = ProductSearchService(catalog)

    results = service.search(
        query="black Nike running shoes",
        constraints={
            "category": "running shoes",
            "brand": "Nike",
            "color": "black",
        },
        intent="buying",
        top_k=10
    )

    print(
        "Results:",
        len(results)
    )

    for rank, candidate in enumerate(
        results,
        start=1
    ):

        product = candidate.product

        print(
            f"{rank}. "
            f"{product.parent_asin} | "
            f"{candidate.final_score:.4f} | "
            f"{product.store} | "
            f"{product.title}"
        )


if __name__ == "__main__":
    main()