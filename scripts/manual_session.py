"""Run a local user-style Shopping Copilot session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from starter.agent import Agent  # noqa: E402


def main() -> int:
    """Read up to ten user turns and print the public Agent response."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="data/catalog.jsonl",
        help="Path to the official catalog.jsonl file.",
    )
    parser.add_argument(
        "--session-id",
        default="manual-session",
        help="Session identifier used for this interactive run.",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = REPOSITORY_ROOT / catalog_path
    if not catalog_path.exists():
        parser.error(f"catalog file does not exist: {catalog_path}")

    agent = Agent(catalog_path)
    agent.reset(
        args.session_id,
        {
            "summary": "Local manual shopping session",
            "purchase_frequency": "occasional",
        },
    )

    print("Shopping Copilot manual session (press Enter on an empty line to exit).")
    print(f"Catalog: {catalog_path}")
    print(f"Retrieval mode: {'Member B' if not agent.using_fallback else 'SQLite fallback'}")

    for turn in range(1, 11):
        try:
            user_message = input(f"\nTurn {turn}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            break

        response = agent.respond(args.session_id, user_message, turn, 10)
        print(f"Assistant: {response.get('message', '')}")
        ask_attribute = response.get("ask_attribute")
        if ask_attribute:
            print(f"Clarification attribute: {ask_attribute}")
        recommendations = response.get("recommendations") or []
        print("Recommendations:")
        for index, item in enumerate(recommendations, start=1):
            if isinstance(item, dict):
                print(f"  {index}. {item.get('parent_asin', '')}")
            else:
                print(f"  {index}. {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
