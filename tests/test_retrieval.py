from data.catalog_loader import ProductCatalog

from retrieval.hard_filter import HardConstraintFilter
from retrieval.hybrid_retriever import HybridRetriever

from shopping_copilot.contracts import SessionState


def test_catalog_size():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    assert len(catalog.products) == 50000


def test_price_filter():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    filter_engine = HardConstraintFilter()

    state = SessionState(
        session_id="test-price",
        intent="buying",
        hard_constraints={
            "category": ["running shoes"],
            "budget": {
                "min": None,
                "max": 100,
            },
        },
    )

    results = filter_engine.filter(
        catalog.products.values(),
        state,
    )

    assert len(results) > 0

    for product in results:
        assert product.price is not None
        assert product.price <= 100


def test_brand_filter():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    filter_engine = HardConstraintFilter()

    state = SessionState(
        session_id="test-brand",
        intent="buying",
        hard_constraints={
            "brand": ["nike"],
            "category": ["running shoes"],
        },
    )

    results = filter_engine.filter(
        catalog.products.values(),
        state,
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

    state = SessionState(
        session_id="test-category",
        intent="buying",
        hard_constraints={
            "category": ["running shoes"],
        },
    )

    results = filter_engine.filter(
        catalog.products.values(),
        state,
    )

    assert len(results) > 0


def test_negative_keyword():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    filter_engine = HardConstraintFilter()

    state = SessionState(
        session_id="test-negative",
        intent="buying",
        hard_constraints={
            "category": ["running shoes"],
        },
        negative_constraints={
            "negative_keywords": ["trail"],
        },
    )

    results = filter_engine.filter(
        catalog.products.values(),
        state,
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

    retriever = HybridRetriever(
        catalog,
        use_semantic=False,
    )

    state = SessionState(
        session_id="test-hybrid",
        intent="buying",
        hard_constraints={
            "category": ["running shoes"],
            "budget": {
                "min": None,
                "max": 100,
            },
        },
    )

    results = retriever.retrieve(
        query="running shoes",
        state=state,
        top_k=100,
    )

    assert len(results.candidates) > 0
    assert len(results.candidates) <= 100


def test_candidates_exist_in_catalog():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(
        catalog,
        use_semantic=False,
    )

    state = SessionState(
        session_id="test-exists",
        intent="buying",
        hard_constraints={
            "category": ["running shoes"],
            "budget": {
                "min": None,
                "max": 100,
            },
        },
    )

    results = retriever.retrieve(
        query="running shoes",
        state=state,
        top_k=100,
    )

    catalog_ids = set(
        catalog.products.keys()
    )

    for candidate in results.candidates:
        assert candidate.parent_asin in catalog_ids


def test_deterministic_retrieval():
    catalog = ProductCatalog()
    catalog.load("data/catalog.jsonl")

    retriever = HybridRetriever(
        catalog,
        use_semantic=False,
    )

    state = SessionState(
        session_id="test-deterministic",
        intent="buying",
        hard_constraints={
            "category": ["running shoes"],
            "budget": {
                "min": None,
                "max": 100,
            },
        },
    )

    results1 = retriever.retrieve(
        query="running shoes",
        state=state,
        top_k=20,
    )

    results2 = retriever.retrieve(
        query="running shoes",
        state=state,
        top_k=20,
    )

    ids1 = [
        candidate.parent_asin
        for candidate in results1.candidates
    ]

    ids2 = [
        candidate.parent_asin
        for candidate in results2.candidates
    ]

    assert ids1 == ids2