"""Build ``data/vocab.json`` by reverse-deriving vocabulary from the catalog.

Usage:
    python scripts/build_vocab.py
    python scripts/build_vocab.py --catalog path/to/catalog.jsonl --out data/vocab.json

The script scans the frozen catalog, extracts brand/category/color/material/
style/feature/use_case vocabulary from structured fields, filters it by
frequency and character shape, and merges it with the hand-written defaults in
``agent.slot_extractor``.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent.slot_extractor import (  # noqa: E402
    DEFAULT_BRANDS,
    DEFAULT_CATEGORY_ALIASES,
    DEFAULT_COLORS,
    DEFAULT_FEATURES,
    DEFAULT_MATERIALS,
    DEFAULT_STYLES,
    DEFAULT_USE_CASE,
)


CATEGORY_BLOCKLIST = {
    "",
    "clothing, shoes & jewelry",
    "women",
    "men",
    "girls",
    "boys",
    "baby",
    "kids",
    "unisex",
    "novelty",
    "novelty & more",
}

BRAND_BLOCKLIST = {
    "",
    "generic",
    "unknown",
    "no brand",
    "unbranded",
    "n/a",
    "na",
    "none",
    "imported",
    "made in usa",
    "amazon",
}

MIN_FREQUENCY = 2


def _clean(value: str) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"[{}()\[\]]", " ", text)
    text = re.sub(r"\d+(?:\.\d+)?%", " ", text)
    text = re.sub(r"^[\d\s%.*\-]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .-")


def _split_values(value: object) -> set:
    if not isinstance(value, str):
        return set()
    out = set()
    for piece in re.split(r"[/;|]", value):
        for sub in piece.split(","):
            cleaned = _clean(sub)
            if cleaned:
                out.add(cleaned)
    return out


def _detail(value: object) -> object:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return value[0] if value else None
    return None


def _starts_alpha(value: str) -> bool:
    return bool(re.match(r"^[a-z]", value))


def _no_brackets(value: str) -> bool:
    return not re.search(r"[\[\]{}()]", value)


def _no_digits(value: str) -> bool:
    return not re.search(r"\d", value)


def _keep(value: str, count: int, max_len: int = 40, allow_digits: bool = False) -> bool:
    if count < MIN_FREQUENCY:
        return False
    if not (2 <= len(value) <= max_len):
        return False
    if not _starts_alpha(value):
        return False
    if not _no_brackets(value):
        return False
    if not allow_digits and not _no_digits(value):
        return False
    return True


def build(catalog_path: Path):
    brands: Counter = Counter()
    categories: Counter = Counter()
    colors: Counter = Counter()
    materials: Counter = Counter()
    styles: Counter = Counter()
    features: Counter = Counter()
    use_case: Counter = Counter()

    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            details = product.get("details") or {}
            details = details if isinstance(details, dict) else {}

            store = product.get("store")
            if isinstance(store, str):
                brands[_clean(store)] += 1

            cats = product.get("categories") or []
            if isinstance(cats, list):
                for cat in cats:
                    if isinstance(cat, str):
                        categories[_clean(cat)] += 1

            for value in _split_values(_detail(details.get("Color"))):
                colors[value] += 1
            for field in ("Material", "Fabric Type"):
                for value in _split_values(_detail(details.get(field))):
                    materials[value] += 1
            for field in ("Style", "Fit Type", "Pattern", "Shape"):
                for value in _split_values(_detail(details.get(field))):
                    styles[value] += 1
            for field in (
                "Sport Type",
                "Sport",
                "Occasion",
                "Theme",
                "Target Audience",
                "Suggested Users",
            ):
                for value in _split_values(_detail(details.get(field))):
                    use_case[value] += 1
            for value in _split_values(_detail(details.get("Special Feature"))):
                features[value] += 1

    catalog_brands = {
        v for v, n in brands.items()
        if v not in BRAND_BLOCKLIST and _keep(v, n, max_len=30, allow_digits=True)
    }
    catalog_categories = {
        v for v, n in categories.items()
        if v not in CATEGORY_BLOCKLIST and _keep(v, n, max_len=40)
    }
    catalog_colors = {v for v, n in colors.items() if _keep(v, n, max_len=20)}
    catalog_materials = {
        v for v, n in materials.items() if _keep(v, n, max_len=40, allow_digits=True)
    }
    catalog_styles = {v for v, n in styles.items() if _keep(v, n, max_len=40)}
    catalog_features = {v for v, n in features.items() if _keep(v, n, max_len=40)}
    catalog_use_case = {v for v, n in use_case.items() if _keep(v, n, max_len=40)}

    category_aliases = {k: list(v) for k, v in DEFAULT_CATEGORY_ALIASES.items()}
    for category in catalog_categories:
        category_aliases.setdefault(category, []).append(category)

    vocab = {
        "brands": sorted(catalog_brands | DEFAULT_BRANDS),
        "category_aliases": {k: sorted(set(v)) for k, v in category_aliases.items()},
        "colors": sorted(catalog_colors | DEFAULT_COLORS),
        "materials": sorted(catalog_materials | DEFAULT_MATERIALS),
        "styles": sorted(catalog_styles | DEFAULT_STYLES),
        "features": sorted(catalog_features | DEFAULT_FEATURES),
        "use_case": sorted(catalog_use_case | DEFAULT_USE_CASE),
    }
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        default=str(REPO / "techjam-conversational-search" / "data" / "catalog.jsonl"),
    )
    parser.add_argument("--out", default=str(REPO / "data" / "vocab.json"))
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise SystemExit(f"catalog not found: {catalog_path}")

    vocab = build(catalog_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"wrote {out_path}")
    for key in (
        "brands",
        "category_aliases",
        "colors",
        "materials",
        "styles",
        "features",
        "use_case",
    ):
        print(f"{key:18s} {len(vocab[key])}")


if __name__ == "__main__":
    main()
