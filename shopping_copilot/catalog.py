"""Offline catalog retriever used as a safe baseline/fallback.

Member B can replace this class with a stronger hybrid retriever.  The public
Agent contract remains unchanged because the orchestrator consumes only the
``retrieve`` protocol.
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .contracts import Candidate, RetrievalResult, SessionState


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
                    self.catalog_ids.add(parent_asin)
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


__all__ = ["SQLiteCatalogRetriever", "flatten_text", "terms"]

