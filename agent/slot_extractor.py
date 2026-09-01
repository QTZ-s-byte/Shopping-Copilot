"""Deterministic, offline product-constraint extraction for a user message.

The extractor targets the official ``ask_attribute`` taxonomy:
``category, material, color, size, style, brand, budget, feature, use_case``.
Every lexicon can be overridden so the team can inject catalog-derived
vocabulary for higher coverage.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_COLORS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "grey",
    "gray",
    "navy",
    "beige",
    "silver",
    "maroon",
    "teal",
    "tan",
    "ivory",
    "cream",
    "charcoal",
    "coral",
    "turquoise",
    "burgundy",
    "khaki",
    "olive",
    "mustard",
    "lavender",
    "mint",
    "rose gold",
    "multicolor",
    "clear",
}

DEFAULT_MATERIALS = {
    "leather",
    "faux leather",
    "cotton",
    "silk",
    "wool",
    "denim",
    "polyester",
    "nylon",
    "suede",
    "linen",
    "velvet",
    "canvas",
    "rubber",
    "metal",
    "plastic",
    "wood",
    "steel",
    "stainless steel",
    "sterling silver",
    "gold",
    "platinum",
    "14k gold",
    "18k gold",
    "ceramic",
    "glass",
    "acrylic",
    "resin",
    "satin",
    "chiffon",
    "lace",
    "cashmere",
}

DEFAULT_STYLES = {
    "casual",
    "formal",
    "dressy",
    "bohemian",
    "boho",
    "vintage",
    "minimalist",
    "minimal",
    "streetwear",
    "athletic",
    "sporty",
    "elegant",
    "classic",
    "modern",
    "chic",
    "edgy",
    "western",
    "retro",
    "trendy",
    "preppy",
    "gothic",
    "punk",
    "business",
    "office",
    "oversized",
    "slim fit",
}

DEFAULT_FEATURES = {
    "waterproof",
    "breathable",
    "lightweight",
    "durable",
    "stretchy",
    "stretch",
    "adjustable",
    "washable",
    "padded",
    "insulated",
    "non-slip",
    "slip-resistant",
    "quick-dry",
    "quick drying",
    "moisture-wicking",
    "hypoallergenic",
    "comfortable",
    "comfort",
    "fit",
    "durability",
    "supportive",
    "flexible",
    "stain-resistant",
    "wrinkle-free",
    "machine washable",
    "anti-odor",
    "shockproof",
    "scratch-resistant",
}

DEFAULT_USE_CASE = {
    "running",
    "work",
    "travel",
    "gym",
    "hiking",
    "outdoor",
    "camping",
    "party",
    "wedding",
    "summer",
    "winter",
    "school",
    "everyday",
    "sports",
    "yoga",
    "swimming",
    "beach",
    "date night",
    "gift",
    "bridal",
    "bridesmaid",
}

# A small default brand set; inject the catalog's store/manufacturer values for
# real coverage.
DEFAULT_BRANDS = {
    "nike",
    "adidas",
    "puma",
    "reebok",
    "vans",
    "converse",
    "new balance",
    "under armour",
    "levi's",
    "levis",
    "calvin klein",
    "tommy hilfiger",
    "ralph lauren",
    "gucci",
    "coach",
    "michael kors",
    "pandora",
    "swarovski",
    "fossil",
    "casio",
    "timex",
    "seiko",
    "zara",
    "h&m",
    "hm",
    "uniqlo",
    "gap",
    "old navy",
    "champion",
    "hanes",
    "carhartt",
    "columbia",
    "the north face",
    "patagonia",
    "crocs",
    "skechers",
    "dr. martens",
    "dr martens",
    "timberland",
    "ugg",
    "steve madden",
    "nine west",
    "anne klein",
    "kate spade",
    "tory burch",
    "ray-ban",
    "ray ban",
    "oakley",
}

# category -> keyword aliases. Longest aliases are matched first so multi-word
# phrases win over single words.
DEFAULT_CATEGORY_ALIASES = {
    "running shoes": ["running shoes", "running shoe", "trainers", "trainer"],
    "casual shoes": ["casual shoes", "casual shoe", "everyday shoes"],
    "formal shoes": ["formal shoes", "formal shoe", "dress shoes", "dress shoe", "oxfords", "oxford"],
    "boots": ["boots", "boot"],
    "sandals": ["sandals", "sandal"],
    "heels": ["heels", "heel", "high heels"],
    "flats": ["flats", "flat", "loafers", "loafer"],
    "shoes": ["shoes", "shoe", "sneakers", "sneaker", "slip-on", "slip ons"],
    "shirts": ["shirt", "shirts", "t-shirt", "tshirt", "tee", "blouse", "blouses"],
    "pants": ["pants", "trousers", "jeans", "denim", "shorts", "skirt", "skirts", "leggings"],
    "dresses": ["dress", "dresses"],
    "outerwear": ["jacket", "jackets", "coat", "coats", "sweater", "sweaters", "hoodie", "hoodies", "cardigan", "cardigans"],
    "activewear": ["activewear", "athletic wear", "workout clothes", "tank top", "tank tops"],
    "tops": ["top", "tops"],
    "jewelry": ["jewelry", "jewellery", "earrings", "earring", "necklace", "necklaces", "pendant", "pendants", "charm", "charms", "anklet", "anklets"],
    "rings": ["ring", "rings"],
    "bracelets": ["bracelet", "bracelets", "bangle", "bangles"],
    "watches": ["watch", "watches"],
    "bags": ["bag", "bags", "backpack", "backpacks", "handbag", "handbags", "purse", "purses", "tote", "totes", "wallet", "wallets"],
    "accessories": ["belt", "belts", "hat", "hats", "scarf", "scarves", "sunglasses", "socks", "gloves", "hair accessory", "hair accessories"],
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "i",
    "im",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "we",
    "our",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "with",
    "and",
    "or",
    "but",
    "is",
    "are",
    "be",
    "was",
    "were",
    "am",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "will",
    "would",
    "should",
    "need",
    "want",
    "wants",
    "wanted",
    "looking",
    "find",
    "get",
    "buy",
    "some",
    "any",
    "something",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "just",
    "please",
    "really",
    "very",
    "about",
    "under",
    "below",
    "over",
    "above",
    "than",
    "less",
    "more",
    "around",
    "between",
    "from",
    "into",
    "up",
    "out",
    "there",
    "here",
    "show",
    "ideas",
    "idea",
    "recommend",
    "recommendation",
    "suggest",
    "suggestion",
    "explore",
    "browse",
    "instead",
    "actually",
    "forget",
    "ignore",
    "earlier",
    "what",
    "whats",
    "what's",
    "give",
    "like",
    "not",
    "without",
    "avoid",
    "except",
    "excluding",
    "never",
    "dislike",
    "hate",
    "dont",
    "don't",
    "anything",
    "preference",
    "doesn't",
    "doesnt",
    "matter",
}


@dataclass
class ExtractedSlots:
    """Structured slots extracted from one user message."""

    category: List[str] = field(default_factory=list)
    brand: List[str] = field(default_factory=list)
    color: List[str] = field(default_factory=list)
    size: List[str] = field(default_factory=list)
    material: List[str] = field(default_factory=list)
    style: List[str] = field(default_factory=list)
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    feature: List[str] = field(default_factory=list)
    use_case: List[str] = field(default_factory=list)
    positive_keywords: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    negative: Dict[str, List[str]] = field(default_factory=dict)

    def to_hard_constraints(self) -> Dict[str, Any]:
        hc: Dict[str, Any] = {}
        if self.category:
            hc["category"] = list(self.category)
        if self.brand:
            hc["brand"] = list(self.brand)
        if self.color:
            hc["color"] = list(self.color)
        if self.size:
            hc["size"] = list(self.size)
        if self.material:
            hc["material"] = list(self.material)
        if self.style:
            hc["style"] = list(self.style)
        if self.budget_min is not None or self.budget_max is not None:
            hc["budget"] = {"min": self.budget_min, "max": self.budget_max}
        return hc

    def to_soft_preferences(self) -> Dict[str, Any]:
        sp: Dict[str, Any] = {}
        if self.feature:
            sp["feature"] = list(self.feature)
        if self.use_case:
            sp["use_case"] = list(self.use_case)
        if self.positive_keywords:
            sp["positive_keywords"] = list(self.positive_keywords)
        return sp

    def to_negative_constraints(self) -> Dict[str, Any]:
        nc: Dict[str, Any] = {}
        for field_name, values in self.negative.items():
            if values:
                nc[field_name] = list(values)
        if self.negative_keywords:
            nc["negative_keywords"] = list(self.negative_keywords)
        return nc


class SlotExtractor:
    def __init__(
        self,
        colors: Optional[Iterable[str]] = None,
        materials: Optional[Iterable[str]] = None,
        styles: Optional[Iterable[str]] = None,
        features: Optional[Iterable[str]] = None,
        brands: Optional[Iterable[str]] = None,
        use_case: Optional[Iterable[str]] = None,
        category_aliases: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        catalog = _load_catalog_vocab()
        self.colors = _normalize_set(colors or catalog.get("colors") or DEFAULT_COLORS)
        self.materials = _normalize_set(materials or catalog.get("materials") or DEFAULT_MATERIALS)
        self.styles = _normalize_set(styles or catalog.get("styles") or DEFAULT_STYLES)
        self.features = _normalize_set(features or catalog.get("features") or DEFAULT_FEATURES)
        self.brands = _normalize_set(brands or catalog.get("brands") or DEFAULT_BRANDS)
        self.use_case = _normalize_set(use_case or catalog.get("use_case") or DEFAULT_USE_CASE)

        aliases = category_aliases or catalog.get("category_aliases") or DEFAULT_CATEGORY_ALIASES
        self.category_aliases = {
            category: [a.lower() for a in alias_list]
            for category, alias_list in aliases.items()
        }

        self._brand_regex = _compile_terms(self.brands)
        self._color_regex = _compile_terms(self.colors)
        self._material_regex = _compile_terms(self.materials)
        self._style_regex = _compile_terms(self.styles)
        self._feature_regex = _compile_terms(self.features)
        self._use_case_regex = _compile_terms(self.use_case)
        self._category_regex, self._alias_to_category = _compile_alias_map(self.category_aliases)

    def extract(self, message: str) -> ExtractedSlots:
        text = _normalize_text(message)
        slots = ExtractedSlots()
        excluded = self._excluded_spans(text)

        slots.budget_min, slots.budget_max = self._extract_budget(text)
        slots.category = self._extract_category(text, excluded)
        slots.brand = self._extract_vocab(text, self._brand_regex, excluded)
        slots.color = self._extract_vocab(text, self._color_regex, excluded)
        slots.material = self._extract_vocab(text, self._material_regex, excluded)

        # Avoid treating explicit material terms such as "cotton" or
        # "polyester" as product categories when the same surface form
        # exists in both vocabularies.
        material_values = {value.lower() for value in slots.material}

        if material_values and slots.category:
            slots.category = [
                category
                for category in slots.category
                if category.lower() not in material_values
            ]

        slots.style = self._extract_vocab(text, self._style_regex, excluded)
        slots.feature = self._extract_vocab(text, self._feature_regex, excluded)
        slots.use_case = self._extract_vocab(text, self._use_case_regex, excluded)
        slots.size = self._extract_size(text)

        consumed = _consumed_terms(slots)
        slots.positive_keywords = _content_keywords(text, consumed, excluded)

        self._extract_negation(text, slots)
        return slots

    def _extract_budget(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        if "$" not in text:
            return None, None

        budget_min: Optional[float] = None
        budget_max: Optional[float] = None

        between = re.search(
            r"(?:between\s+)?(?:\$|usd\s*)?(\d+(?:\.\d+)?)\s*(?:-|to|and)\s*(?:\$|usd\s*)?(\d+(?:\.\d+)?)",
            text,
        )
        if between:
            return float(between.group(1)), float(between.group(2))

        max_match = re.search(
            r"(?:under|below|less\s+than|at\s+most|up\s+to|no\s+more\s+than|within|maximum|max|cheaper\s+than)"
            r"\s*(?:\$|usd\s*)?(\d+(?:\.\d+)?)",
            text,
        )
        if max_match:
            budget_max = float(max_match.group(1))

        min_match = re.search(
            r"(?:above|over|more\s+than|at\s+least|minimum|min|starting\s+from|from)\s*"
            r"(?:\$|usd\s*)?(\d+(?:\.\d+)?)",
            text,
        )
        if min_match:
            budget_min = float(min_match.group(1))

        if budget_min is None and budget_max is None:
            bare = re.search(r"\$\s*(\d+(?:\.\d+)?)", text)
            if bare:
                budget_max = float(bare.group(1))

        return budget_min, budget_max

    def _extract_category(
        self, text: str, excluded: List[Tuple[int, int]]
    ) -> List[str]:
        if self._category_regex is None:
            return []
        found: List[str] = []
        for match in self._category_regex.finditer(text):
            if _overlaps((match.start(), match.end()), excluded):
                continue
            category = self._alias_to_category.get(match.group(0))
            if category:
                found.append(category)
        return _dedupe(found)

    def _extract_vocab(
        self, text: str, regex, excluded: List[Tuple[int, int]]
    ) -> List[str]:
        if regex is None:
            return []
        found: List[str] = []
        for match in regex.finditer(text):
            if _overlaps((match.start(), match.end()), excluded):
                continue
            found.append(match.group(0))
        return _dedupe(found)

    def _extract_size(self, text: str) -> List[str]:
        found: List[str] = []
        for match in re.finditer(r"\bsize\s+(\d+(?:\.\d+)?|[a-zA-Z]+)", text):
            found.append(match.group(1).lower())
        for label in (
            "extra small",
            "extra large",
            "one size",
            "xxl",
            "xl",
            "xs",
            "medium",
            "large",
            "small",
        ):
            if re.search(r"\b" + re.escape(label) + r"\b", text):
                found.append(label)
        return _dedupe(found)

    def _extract_negation(self, text: str, slots: ExtractedSlots) -> None:
        negation_markers = (
            r"(?:not|no|without|avoid|except|excluding|anything\s+but|don'?t\s+want|"
            r"don'?t\s+like|dislike|hate)"
        )
        vocab_by_field = [
            ("color", self._color_regex),
            ("material", self._material_regex),
            ("brand", self._brand_regex),
            ("style", self._style_regex),
            ("feature", self._feature_regex),
        ]
        for field_name, regex in vocab_by_field:
            if regex is None:
                continue
            for match in regex.finditer(text):
                if _is_negated(text, match.start()):
                    slots.negative.setdefault(field_name, []).append(match.group(0))

        for match in re.finditer(
            r"\b" + negation_markers + r"\s+((?:\w+\s*){1,4}?)(?=\s+(?:but|and|or|,|\.)|\s*$)",
            text,
        ):
            phrase = match.group(1).strip()
            if phrase and phrase not in _FLAT_VOCAB:
                slots.negative_keywords.append(phrase)
        slots.negative_keywords = _dedupe(slots.negative_keywords)

    def _excluded_spans(self, text: str) -> List[Tuple[int, int]]:
        discard_markers = [
            r"\bforget(?:\s+about)?\b",
            r"\bnever\s+mind\b",
            r"\bnevermind\b",
            r"\binstead\s+of\b",
            r"\bscratch\b",
            r"\bno\s+longer\s+want\b",
        ]
        negation_markers = [
            r"\bnot\b",
            r"\bno\b",
            r"\bwithout\b",
            r"\bavoid\b",
            r"\bexcept\b",
            r"\bexcluding\b",
            r"\banything\s+but\b",
            r"\bdon'?t\s+want\b",
            r"\bdon'?t\s+like\b",
            r"\bdislike\b",
            r"\bhate\b",
        ]
        boundary = r"(?=\.|,|!|\?|\band\b|\bbut\b|\blet'?s\b|\bi\s+want\b|\binstead\b|$)"
        spans: List[Tuple[int, int]] = []
        for marker in discard_markers + negation_markers:
            pattern = marker + r"\s+((?:\w[\w'-]*\s*){1,4}?)" + boundary
            for match in re.finditer(pattern, text):
                spans.append((match.start(1), match.end(1)))
        return spans


def _normalize_text(message: str) -> str:
    text = message.lower().strip()
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:dollars?|bucks|usd)\b", r"$\1", text)
    return text


def _normalize_set(values: Iterable[str]) -> Set[str]:
    return {str(v).lower().strip() for v in values if str(v).strip()}


def _overlaps(span: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
    start, end = span
    for other_start, other_end in spans:
        if start < other_end and other_start < end:
            return True
    return False


def _dedupe(values: List[str]) -> List[str]:
    seen: List[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _consumed_terms(slots: ExtractedSlots) -> Set[str]:
    terms: Set[str] = set()
    for value in (
        slots.category
        + slots.brand
        + slots.color
        + slots.material
        + slots.style
        + slots.feature
        + slots.use_case
    ):
        terms.update(value.split())
    for value in slots.size:
        terms.add(value)
    return terms


def _content_keywords(
    text: str, consumed: Set[str], excluded: List[Tuple[int, int]]
) -> List[str]:
    keywords: List[str] = []
    for match in re.finditer(r"[a-z][a-z0-9']*", text):
        word = match.group(0)
        if word in STOPWORDS or word in consumed or len(word) < 3:
            continue
        if _overlaps((match.start(), match.end()), excluded):
            continue
        keywords.append(word)
    return _dedupe(keywords)


_COMPILE_CACHE = {}


def _compile_terms(terms: Set[str]):
    key = frozenset(terms)
    cached = _COMPILE_CACHE.get(("terms", key))
    if cached is not None:
        return cached
    ordered = sorted(terms, key=lambda t: (len(t), t), reverse=True)
    regex = None
    if ordered:
        pattern = r"\b(?:" + "|".join(re.escape(t) for t in ordered) + r")\b"
        regex = re.compile(pattern)
    _COMPILE_CACHE[("terms", key)] = regex
    return regex


def _compile_alias_map(category_aliases: Dict[str, List[str]]):
    pairs = []
    for category, aliases in category_aliases.items():
        for alias in aliases:
            pairs.append((alias.lower(), category))
    key = frozenset(pairs)
    cached = _COMPILE_CACHE.get(("alias", key))
    if cached is not None:
        return cached
    ordered = sorted(pairs, key=lambda p: (len(p[0]), p[0]), reverse=True)
    regex = None
    alias_to_category = {}
    if ordered:
        pattern = r"\b(?:" + "|".join(re.escape(a) for a, _ in ordered) + r")\b"
        regex = re.compile(pattern)
        alias_to_category = {a: c for a, c in ordered}
    result = (regex, alias_to_category)
    _COMPILE_CACHE[("alias", key)] = result
    return result


_NEGATION_MARKERS = (
    "anything but",
    "don't want",
    "dont want",
    "don't like",
    "dont like",
    "without",
    "avoid",
    "except",
    "excluding",
    "dislike",
    "hate",
    "not",
    "no",
)


def _is_negated(text: str, start: int, max_words: int = 3) -> bool:
    prefix = text[:start]
    for marker in _NEGATION_MARKERS:
        idx = prefix.rfind(marker)
        if idx < 0:
            continue
        before_ok = idx == 0 or not prefix[idx - 1].isalnum()
        after_idx = idx + len(marker)
        after_ok = after_idx >= len(text) or not text[after_idx].isalnum()
        if not (before_ok and after_ok):
            continue
        between = prefix[after_idx:start]
        if len(re.findall(r"\w+", between)) <= max_words:
            return True
    return False


_FLAT_VOCAB = set()
for _vocab in (
    DEFAULT_COLORS,
    DEFAULT_MATERIALS,
    DEFAULT_STYLES,
    DEFAULT_FEATURES,
    DEFAULT_BRANDS,
    DEFAULT_USE_CASE,
):
    _FLAT_VOCAB.update(_vocab)


_CATALOG_VOCAB_CACHE = None


def _load_catalog_vocab():
    global _CATALOG_VOCAB_CACHE
    if _CATALOG_VOCAB_CACHE is not None:
        return _CATALOG_VOCAB_CACHE

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "data", "vocab.json"),
        "data/vocab.json",
    ]
    data = {}
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                break
            except (OSError, ValueError):
                continue
    _CATALOG_VOCAB_CACHE = data
    return data
