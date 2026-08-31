"""Thin adapters around the organizer's public evaluator.

The adapter never edits the evaluator or labels.  It only launches the
published module when it is present, then reads its JSON output for a compact
report.  This keeps the submission compatible with the official
``from starter.agent import Agent`` entry point.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import aggregate_metrics


@dataclass(frozen=True)
class EvaluationReport:
    aggregate: Mapping[str, Any]
    raw: Mapping[str, Any]
    command: tuple[str, ...] = ()


def load_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("evaluation result must be a JSON object")
    return value


def summarize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    sessions = result.get("sessions")
    if isinstance(sessions, Sequence) and not isinstance(sessions, (str, bytes)):
        aggregate = aggregate_metrics(list(sessions))
        if "reported_token_usage" in result:
            aggregate["reported_token_usage"] = result["reported_token_usage"]
        return aggregate
    aggregate_keys = (
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
        "reported_token_usage",
        "scenario_metrics",
    )
    return {key: result[key] for key in aggregate_keys if key in result}


def run_official_evaluator(
    *,
    project_root: str | Path,
    catalog_path: str | Path = "data/catalog.jsonl",
    dataset_path: str | Path = "data/public_set.jsonl",
    output_path: str | Path = "results.json",
    python_executable: str | None = None,
) -> EvaluationReport:
    """Run ``evaluator.local_evaluator`` without modifying its source."""

    root = Path(project_root)
    evaluator_file = root / "evaluator" / "local_evaluator.py"
    if not evaluator_file.exists():
        raise FileNotFoundError(
            "official evaluator is not present; copy the participant-kit evaluator "
            "or point project_root at the participant-kit checkout"
        )
    python = python_executable or sys.executable
    command = (
        python,
        "-m",
        "evaluator.local_evaluator",
        "--catalog",
        str(catalog_path),
        "--dataset",
        str(dataset_path),
        "--output",
        str(output_path),
    )
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"official evaluator failed with exit code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    result = load_result(root / output_path if not Path(output_path).is_absolute() else output_path)
    return EvaluationReport(aggregate=summarize_result(result), raw=result, command=command)
