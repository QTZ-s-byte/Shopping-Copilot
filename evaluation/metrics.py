"""Pure metric functions matching the official public evaluator.

Keeping these functions independent of the Agent makes it possible to unit
test scoring logic without reading private data or changing organizer files.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


def reciprocal_rank(ranked_ids: Sequence[str], target_id: str, top_k: int = 10) -> float:
    target = str(target_id)
    for index, value in enumerate(ranked_ids[:top_k], start=1):
        if str(value) == target:
            return 1.0 / index
    return 0.0


def hit_rate_at_k(ranked_lists: Iterable[Sequence[str]], targets: Iterable[str], k: int = 10) -> float:
    pairs = list(zip(ranked_lists, targets))
    if not pairs:
        return 0.0
    return sum(str(target) in {str(value) for value in ranked[:k]} for ranked, target in pairs) / len(pairs)


def mean_reciprocal_rank(
    ranked_lists: Iterable[Sequence[str]], targets: Iterable[str], k: int = 10
) -> float:
    values = [reciprocal_rank(ranked, target, k) for ranked, target in zip(ranked_lists, targets)]
    return fmean(values) if values else 0.0


def mean_turn_to_conversion(
    first_hit_turns: Iterable[int | None], *, max_turns: int = 10, miss_turn: int | None = None
) -> float:
    miss_value = max_turns + 1 if miss_turn is None else int(miss_turn)
    values = [miss_value if turn is None else int(turn) for turn in first_hit_turns]
    return fmean(values) if values else 0.0


def efficiency(mttc: float, *, max_turns: int = 10, miss_turn: int | None = None) -> float:
    miss_value = max_turns + 1 if miss_turn is None else int(miss_turn)
    value = (float(miss_value) - float(mttc)) / float(max_turns)
    return max(0.0, min(1.0, value))


def technical_score(hit_rate: float, mrr: float, efficiency_value: float) -> float:
    return 0.50 * float(hit_rate) + 0.30 * float(mrr) + 0.20 * float(efficiency_value)


def aggregate_metrics(
    sessions: Sequence[Mapping[str, Any]], *, max_turns: int = 10, top_k: int = 10,
    _include_scenarios: bool = True,
) -> dict[str, Any]:
    """Aggregate official-style per-session records.

    Each record may contain either ``ranked_ids``/``target_id`` or the already
    normalized fields ``hit``, ``reciprocal_rank`` and ``first_hit_turn``.
    """

    normalized: list[dict[str, Any]] = []
    for item in sessions:
        record = dict(item)
        if "ranked_ids" in record and "target_id" in record:
            ranked = [str(value) for value in record.get("ranked_ids", ())]
            target = str(record["target_id"])
            rank = next((i + 1 for i, value in enumerate(ranked[:top_k]) if value == target), None)
            record["hit"] = rank is not None
            record["best_rank"] = rank
            record["reciprocal_rank"] = 0.0 if rank is None else 1.0 / rank
        record.setdefault("hit", bool(record.get("best_rank")))
        record.setdefault("reciprocal_rank", 0.0)
        record.setdefault("first_hit_turn", None)
        normalized.append(record)

    if not normalized:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": 0.0,
            "efficiency": 0.0,
            "recommended_technical_score": 0.0,
            "scenario_metrics": {},
        }

    hit = sum(bool(item["hit"]) for item in normalized) / len(normalized)
    mrr = fmean(float(item["reciprocal_rank"]) for item in normalized)
    mttc = mean_turn_to_conversion(
        [item.get("first_hit_turn") for item in normalized], max_turns=max_turns
    )
    eff = efficiency(mttc, max_turns=max_turns)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if _include_scenarios:
        for item in normalized:
            scenario = item.get("scenario_type")
            if scenario:
                grouped[str(scenario)].append(item)
    return {
        "sample_count": len(normalized),
        "hit_rate_at_10": round(hit, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(eff, 6),
        "recommended_technical_score": round(technical_score(hit, mrr, eff), 6),
        "scenario_metrics": {
            name: aggregate_metrics(
                records, max_turns=max_turns, top_k=top_k, _include_scenarios=False
            )
            for name, records in sorted(grouped.items())
        },
    }
