"""Print the chunks retrieved for a question -- useful for debugging retrieval."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import get_retriever  # noqa: E402


def show(question: str) -> None:
    print(f"\n--- {question} ---")
    docs = get_retriever().invoke(question)
    for d in docs:
        src = d.metadata.get("source", "?")
        url = d.metadata.get("source_url", "")
        print(f"  [{src:<8}] {url}")


def main() -> int:
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return 0
    for q in [
        "What does StratifiedKFold do that KFold does not?",
        "What does numpy.asarray do that numpy.array does not?",
        "What is the difference between Pipeline and make_pipeline in scikit-learn?",
    ]:
        show(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
