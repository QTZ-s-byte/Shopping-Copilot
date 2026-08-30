from data.catalog_loader import ProductCatalog


def main():
    catalog = ProductCatalog()

    # 加载官方商品目录
    catalog.load("data/catalog.jsonl")

    # 1. 检查商品数量
    print("Product count:", len(catalog.products))

    # 2. 看第一个商品
    first_product = next(iter(catalog.products.values()))
    print("\nFirst product:")
    print(first_product)

    # 3. 测试 ASIN 查询
    product_id = first_product.parent_asin
    found = catalog.get(product_id)

    print("\nLookup result:")
    print(found)

    # 4. 测试关键词搜索
    results = catalog.search("shoes")

    print("\nSearch results for 'shoes':", len(results))

    for product in results[:5]:
        print("-", product.parent_asin, product.title)


if __name__ == "__main__":
    main()