from data.catalog_loader import ProductCatalog
from retrieval.hard_filter import HardConstraintFilter
from retrieval.hybrid_retriever import HybridRetriever


def test_catalog_size():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    assert len(catalog.products) == 50000


def test_price_filter():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    filter_engine = HardConstraintFilter()

    results = filter_engine.filter(
        catalog.products.values(),
        {
            "category": "running shoes",
            "price_max": 100,
        }
    )

    assert len(results) > 0

    for product in results:
        if product.price is not None:
            assert product.price <= 100


def test_brand_filter():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    filter_engine = HardConstraintFilter()

    results = filter_engine.filter(
        catalog.products.values(),
        {
            "brand": "Nike",
            "category": "running shoes",
        }
    )

    assert len(results) > 0

    for product in results:
        evidence = " ".join([
            product.store or "",
            product.details.get("Manufacturer", ""),
            product.title,
        ]).lower()

        assert "nike" in evidence


def test_category_filter():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    filter_engine = HardConstraintFilter()

    results = filter_engine.filter(
        catalog.products.values(),
        {
            "category": "running shoes",
        }
    )

    assert len(results) > 0


def test_negative_keyword():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    filter_engine = HardConstraintFilter()

    results = filter_engine.filter(
        catalog.products.values(),
        {
            "category": "running shoes",
            "negative_keywords": ["trail"],
        }
    )

    assert len(results) > 0

    for product in results:
        text = " ".join([
            product.title,
            *product.features,
            *product.description,
            *product.categories,
        ]).lower()

        assert "trail" not in text


def test_hybrid_retrieval_returns_candidates():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(catalog)

    results = retriever.retrieve(
        query="running shoes",
        constraints={
            "category": "running shoes",
            "price_max": 100,
        },
        top_k=100,
    )

    assert len(results) > 0
    assert len(results) <= 100


def test_candidates_exist_in_catalog():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(catalog)

    results = retriever.retrieve(
        query="running shoes",
        constraints={
            "category": "running shoes",
            "price_max": 100,
        },
        top_k=100,
    )

    catalog_ids = set(catalog.products.keys())

    for candidate in results:
        assert candidate.product.parent_asin in catalog_ids


def test_deterministic_retrieval():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(catalog)

    kwargs = {
        "query": "running shoes",
        "constraints": {
            "category": "running shoes",
            "price_max": 100,
        },
        "top_k": 20,
    }

    results1 = retriever.retrieve(**kwargs)
    results2 = retriever.retrieve(**kwargs)

    ids1 = [
        candidate.product.parent_asin
        for candidate in results1
    ]

    ids2 = [
        candidate.product.parent_asin
        for candidate in results2
    ]

    assert ids1 == ids2