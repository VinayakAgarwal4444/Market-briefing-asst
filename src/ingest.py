"""Load processed docs, split into chunks, embed, and persist a FAISS index."""
from __future__ import annotations

import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PROCESSED_DIR,
    VECTORSTORE_DIR,
    ensure_dirs,
)
from src.llm import get_embeddings


_FRONT_MATTER_RE = re.compile(
    r"\A---\s*\n(?P<fm>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group("fm").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, match.group("body")


def _load_processed_docs() -> list[Document]:
    docs: list[Document] = []
    for md_path in sorted(PROCESSED_DIR.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(text)
        source = md_path.parent.name
        docs.append(
            Document(
                page_content=body,
                metadata={
                    "source": source,
                    "source_url": meta.get("source_url", ""),
                    "title": meta.get("title", md_path.stem),
                    "path": str(md_path.relative_to(PROCESSED_DIR)),
                },
            )
        )
    return docs


def run() -> int:
    """Build the FAISS index from processed docs; return chunk count."""
    ensure_dirs()
    docs = _load_processed_docs()
    if not docs:
        raise RuntimeError(
            f"No processed docs found in {PROCESSED_DIR}. "
            "Run the scraper first (python scripts/build_index.py)."
        )

    print(f"[ingest] loaded {len(docs)} source documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] split into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embeddings = get_embeddings()
    print("[ingest] embedding chunks with Ollama (this may take a minute)...")
    store = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTORSTORE_DIR))
    print(f"[ingest] FAISS index saved to {VECTORSTORE_DIR}")
    return len(chunks)


if __name__ == "__main__":
    run()
