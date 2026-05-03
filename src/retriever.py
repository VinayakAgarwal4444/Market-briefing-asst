"""Load the persisted FAISS index and expose a configured retriever.

Default is a hybrid retriever that ensembles a lexical BM25 retriever with
the dense FAISS retriever. This fixes a real failure mode we observed in
evaluation: dense embeddings sometimes miss queries that contain very
specific symbols (e.g. "StratifiedKFold") because the embedding distance
between the query and the right passage is large despite the verbatim
overlap. BM25 handles those queries well, while FAISS handles paraphrased
or conceptual queries -- ensembling gives us the best of both.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from config import (
    RETRIEVAL_BM25_WEIGHT,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_HYBRID,
    RETRIEVAL_K,
    RETRIEVAL_SEARCH_TYPE,
    VECTORSTORE_DIR,
)
from src.llm import get_embeddings


class VectorstoreMissingError(RuntimeError):
    """Raised when no FAISS index exists on disk."""


@lru_cache(maxsize=1)
def load_vectorstore() -> FAISS:
    if not (VECTORSTORE_DIR / "index.faiss").exists():
        raise VectorstoreMissingError(
            f"No FAISS index found at {VECTORSTORE_DIR}. "
            "Build it first: python scripts/build_index.py"
        )
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _dense_retriever(k: int) -> BaseRetriever:
    if RETRIEVAL_SEARCH_TYPE == "mmr":
        return load_vectorstore().as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": RETRIEVAL_FETCH_K},
        )
    return load_vectorstore().as_retriever(search_kwargs={"k": k})


# A small English stopword list. We deliberately avoid pulling in NLTK
# for this -- the list below catches the question-shaped noise ("what",
# "does", "between", ...) that otherwise drowns out rare informative
# tokens like "StratifiedKFold".
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing
    for from had has have having how i if in into is it its of on or so
    such than that the their them then there these they this those to
    too was we were what when where which who why will with would you
    your not no yes any all more most some between also however whether
    """.split()
)

# Word-token regex that PRESERVES identifiers like StratifiedKFold and
# numpy_asarray; punctuation including dots becomes a separator so that
# "numpy.asarray" tokenises to ["numpy", "asarray"].
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*", re.ASCII)


def _bm25_preprocess(text: str) -> list[str]:
    return [
        m.group(0).lower()
        for m in _TOKEN_RE.finditer(text)
        if m.group(0).lower() not in _STOPWORDS
    ]


def _bm25_retriever(k: int) -> BM25Retriever:
    """Build a BM25 retriever over the same chunks indexed in FAISS.

    BM25Retriever needs the chunk texts directly; we read them out of the
    FAISS docstore so dense and lexical retrievers operate on identical
    chunks (no risk of drift from re-chunking).

    A custom preprocessor strips stopwords and lowercases tokens so rare
    informative tokens (e.g. ``StratifiedKFold``) dominate scoring instead
    of being diluted by question-shaped filler words.
    """
    vs = load_vectorstore()
    docs = list(vs.docstore._dict.values())  # noqa: SLF001 -- docstore has no public iterator
    retriever = BM25Retriever.from_documents(
        docs,
        preprocess_func=_bm25_preprocess,
    )
    retriever.k = k
    return retriever


def get_retriever(k: int | None = None) -> BaseRetriever:
    """Return the configured retriever (hybrid by default)."""
    k = k or RETRIEVAL_K
    dense = _dense_retriever(k)
    if not RETRIEVAL_HYBRID:
        return dense

    bm25 = _bm25_retriever(k)
    bm25_w = max(0.0, min(1.0, RETRIEVAL_BM25_WEIGHT))
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_w, 1.0 - bm25_w],
    )
