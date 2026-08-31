"""Convert Member A's structured state into Member B's flat constraint dict.

Member A stores slots as lists (``brand: ["nike"]``) and budget as a
``{"min": ..., "max": ...}`` dict. Member B's ``HardConstraintFilter`` and
``RuleRanker`` expect flat string values plus ``price_min`` / ``price_max``.
This adapter is the single translation point.
"""

from typing import Any, Dict, List, Optional

from agent.types import SessionState


def _last(values: Optional[List[str]]) -> Optional[str]:
    if not values:
        return None
    return str(values[-1])


def to_constraints(state: SessionState) -> Dict[str, Any]:
    hard = state.hard_constraints
    budget = hard.get("budget") or {}

    constraints: Dict[str, Any] = {}

    for field in ("category", "brand", "color", "size", "material", "style"):
        value = _last(hard.get(field))
        if value:
            constraints[field] = value

    if isinstance(budget, dict):
        if budget.get("min") is not None:
            constraints["price_min"] = float(budget["min"])
        if budget.get("max") is not None:
            constraints["price_max"] = float(budget["max"])

    negative_keywords = state.negative_constraints.get("negative_keywords") or []
    constraints["negative_keywords"] = list(negative_keywords)

    return constraints


def to_query(state: SessionState, user_message: str) -> str:
    """Build a retrieval query from the latest message plus a compact state summary."""
    summary = state.summary.strip()
    message = user_message.strip()
    if summary:
        return f"{message} {summary}"
    return message
