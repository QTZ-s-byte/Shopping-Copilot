"""Offline catalog retriever used as a safe baseline/fallback.

The public Agent contract remains unchanged because the orchestrator consumes
only the ``retrieve`` protocol.
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .contracts import Candidate, Product, RetrievalResult, SessionState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {flatten_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class SQLiteCatalogRetriever:
    """SQLite FTS5 baseline with a deterministic empty-catalog fallback."""

    fields = ("title", "categories", "features", "details", "store", "description")

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.catalog_ids: set[str] = set()
        self.products: dict[str, Product] = {}
        self._available = False
        self._build_index()

    @property
    def valid_ids(self) -> set[str]:
        return set(self.catalog_ids)

    def _open_catalog(self):
        if self.catalog_path.suffix == ".gz":
            return gzip.open(self.catalog_path, mode="rt", encoding="utf-8")
        return self.catalog_path.open(encoding="utf-8")

    def _build_index(self) -> None:
        if not self.catalog_path.exists():
            return
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "CREATE VIRTUAL TABLE products USING fts5("
                "parent_asin UNINDEXED, title, categories, features, details, store, description, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
            batch: list[tuple[str, str, str, str, str, str, str]] = []
            with self._open_catalog() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    product = json.loads(line)
                    parent_asin = str(product.get("parent_asin", "")).strip()
                    if not parent_asin:
                        continue
                    product_record = Product(
                        parent_asin=parent_asin,
                        title=flatten_text(product.get("title")),
                        features=tuple(_as_text_list(product.get("features"))),
                        description=tuple(_as_text_list(product.get("description"))),
                        price=_normalize_price(product.get("price")),
                        categories=tuple(_as_text_list(product.get("categories"))),
                        details=product.get("details") if isinstance(product.get("details"), dict) else {},
                        average_rating=product.get("average_rating"),
                        rating_number=product.get("rating_number", 0),
                        store=str(product.get("store")) if product.get("store") is not None else None,
                    )
                    self.catalog_ids.add(parent_asin)
                    self.products[parent_asin] = product_record
                    batch.append(
                        (
                            parent_asin,
                            flatten_text(product.get("title")),
                            flatten_text(product.get("categories")),
                            flatten_text(product.get("features")),
                            flatten_text(product.get("details")),
                            flatten_text(product.get("store")),
                            flatten_text(product.get("description")),
                        )
                    )
                    if len(batch) >= 1000:
                        cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                        batch.clear()
            if batch:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            self.connection.commit()
            self._available = True
        except (OSError, json.JSONDecodeError, sqlite3.Error):
            self.connection.rollback()
            self.catalog_ids.clear()
            self.products.clear()
            self._available = False

    def retrieve(self, query: str, state: SessionState, top_k: int) -> RetrievalResult:
        if not self._available:
            return RetrievalResult()
        unique_terms = list(dict.fromkeys(terms(query)))[:40]
        if not unique_terms:
            return RetrievalResult()
        expression = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in unique_terms)
        limit = max(1, min(100, int(top_k)))
        try:
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5) "
                "FROM products WHERE products MATCH ? ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.Error:
            return RetrievalResult()
        candidates = []
        for parent_asin, raw_score in rows:
            # FTS5's bm25 is lower-is-better; map it to a stable higher-is-better score.
            score = 1.0 / (1.0 + max(0.0, float(raw_score)))
            candidates.append(
                Candidate(
                    parent_asin=str(parent_asin),
                    product=self.products.get(str(parent_asin)),
                    score=score,
                    source_scores={"bm25": score},
                    reasons=("keyword_match",),
                )
            )
        # A cheap count lets the orchestrator trigger a clarification for broad queries.
        try:
            total = int(
                self.connection.execute(
                    "SELECT count(*) FROM products WHERE products MATCH ?", (expression,)
                ).fetchone()[0]
            )
        except sqlite3.Error:
            total = len(candidates)
        return RetrievalResult(candidates=tuple(candidates), total_count=total)


class ProductCatalog:
    """Canonical in-memory catalog used by the hybrid retrieval pipeline."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.products: dict[str, Product] = {}
        if path is not None:
            self.load(path)

    @property
    def valid_ids(self) -> set[str]:
        return set(self.products)

    def load(self, path: str | Path) -> None:
        source = Path(path)
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, mode="rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                parent_asin = str(raw.get("parent_asin", "")).strip()
                if not parent_asin:
                    continue
                self.products[parent_asin] = Product(
                    parent_asin=parent_asin,
                    title=flatten_text(raw.get("title")),
                    features=tuple(_as_text_list(raw.get("features"))),
                    description=tuple(_as_text_list(raw.get("description"))),
                    price=_normalize_price(raw.get("price")),
                    categories=tuple(_as_text_list(raw.get("categories"))),
                    details=raw.get("details") if isinstance(raw.get("details"), dict) else {},
                    average_rating=raw.get("average_rating"),
                    rating_number=raw.get("rating_number", 0),
                    store=str(raw.get("store")) if raw.get("store") is not None else None,
                )

    def get(self, product_id: str) -> Product | None:
        return self.products.get(str(product_id))


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _normalize_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


__all__ = ["Product", "ProductCatalog", "SQLiteCatalogRetriever", "flatten_text", "terms"]
