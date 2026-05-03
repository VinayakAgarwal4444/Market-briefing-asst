"""Compare what BM25, FAISS, and the ensemble retrieve for a hard query."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.retrievers import BM25Retriever  # noqa: E402

from src.retriever import _dense_retriever, get_retriever, load_vectorstore  # noqa: E402


QUESTIONS = [
    "What does StratifiedKFold do that KFold does not?",
    "What does numpy.asarray do that numpy.array does not?",
]


def show(label: str, docs) -> None:
    print(f"  {label}:")
    for d in docs:
        url = d.metadata.get("source_url", "?").split("/")[-1]
        print(f"    - {url}")


def main() -> int:
    vs = load_vectorstore()
    all_docs = list(vs.docstore._dict.values())

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = 8
    dense = _dense_retriever(8)
    ensemble = get_retriever()

    for q in QUESTIONS:
        print(f"\n=== {q} ===")
        show("BM25 alone (k=8)", bm25.invoke(q))
        show("FAISS+MMR alone (k=8)", dense.invoke(q))
        show("Hybrid ensemble", ensemble.invoke(q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
