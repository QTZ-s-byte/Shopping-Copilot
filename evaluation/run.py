"""Internal compatibility wrapper around the official evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import run_official_evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public TechJam evaluator")
    parser.add_argument("--root", default=".", help="participant-kit/project root")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/results.json")
    args = parser.parse_args()
    report = run_official_evaluator(
        project_root=Path(args.root),
        catalog_path=args.catalog,
        dataset_path=args.dataset,
        output_path=args.output,
    )
    print(json.dumps(dict(report.aggregate), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

