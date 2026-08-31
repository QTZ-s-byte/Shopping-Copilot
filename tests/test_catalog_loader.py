from data.catalog_loader import ProductCatalog


def main():
    catalog = ProductCatalog()

    # Load the official product catalog.
    catalog.load("data/catalog.jsonl")

    # 1. Check the catalog size.
    print("Product count:", len(catalog.products))

    # 2. Inspect the first product.
    first_product = next(iter(catalog.products.values()))
    print("\nFirst product:")
    print(first_product)

    # 3. Test ASIN lookup.
    product_id = first_product.parent_asin
    found = catalog.get(product_id)

    print("\nLookup result:")
    print(found)

    # 4. Test keyword search.
    results = catalog.search("shoes")

    print("\nSearch results for 'shoes':", len(results))

    for product in results[:5]:
        print("-", product.parent_asin, product.title)


if __name__ == "__main__":
    main()
