"""Quick CLI to ask the RAG chain a single question."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chain import ask  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the RAG teaching assistant.")
    parser.add_argument("question", nargs="+", help="Question text")
    args = parser.parse_args()
    question = " ".join(args.question)

    result = ask(question)

    print("\n=== Answer ===")
    print(result["answer"])

    print("\n=== Sources ===")
    seen: set[str] = set()
    for doc in result.get("context", []):
        url = doc.metadata.get("source_url") or doc.metadata.get("source", "")
        title = doc.metadata.get("title", "")
        key = url or title
        if key in seen:
            continue
        seen.add(key)
        print(f"- {title} -- {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
