"""Hybrid search over the stories in the news window.

Ranking = reciprocal-rank fusion of two retrievers:

* **Dense** (bge-small cosine) catches paraphrases: "chip stocks" finds
  "semiconductor shares".
* **BM25** catches exact names and tickers that embeddings blur:
  "Okta", "Series I bond", "INTC".

An optional recency boost multiplies the fused score by
``(1 - w) + w * 0.5^(age / half_life)``. It is off by default
(``RECENCY_WEIGHT=0``): on the evaluation set even w = 0.05 pushed correct
but day-old stories down the ranking (hit@1 27/28 -> 25/28; w = 0.4 gave
20/28), because fused scores sit so close together that any multiplier
reorders them. The 48-hour window already keeps everything fresh.

A few hundred headlines fit in one matrix, so dense search is a single
exact matrix-vector product -- an approximate index (FAISS, HNSW) would
add a dependency and buy nothing at this size.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from config import (
    RECENCY_HALF_LIFE_HOURS,
    RECENCY_WEIGHT,
    RETRIEVAL_K,
    RRF_K,
)
from src.embed import embed_query
from src.stories import Story

# A story reaches the LLM only if it is close to the question in absolute
# terms AND not far behind the best match. Without the relative cut, every
# question drags in a tail of loosely related market stories (cosine ~0.6)
# that the model then feels obliged to mention. A strong keyword match (at
# least half the best BM25 score) lifts the relative cut but never the
# absolute one: in the evaluation every correct story scored >= 0.66, while
# keyword-only noise on uncovered questions scored 0.45-0.64. Dropping weak
# matches is what lets the assistant say "nothing was reported" instead of
# stretching a loose one.
MIN_DENSE_SIMILARITY = 0.62
MAX_GAP_FROM_BEST = 0.15
STRONG_KEYWORD_SHARE = 0.5

# Headlines spell out what people abbreviate (and vice versa), which BM25
# can't see: "ECB" never matches "European Central Bank keeps rate on hold".
ABBREVIATIONS: dict[str, str] = {
    "ecb": "european central bank",
    "fed": "federal reserve",
    "fomc": "federal reserve",
    "boe": "bank of england",
    "boj": "bank of japan",
    "pboc": "people's bank of china",
    "cpi": "inflation consumer price",
    "ppi": "producer price",
    "pce": "inflation",
    "gdp": "economy growth",
    "nfp": "payrolls jobs",
    "ipo": "listing",
    "opec": "oil",
    "m&a": "merger acquisition deal",
    "treasuries": "treasury bond yield",
}

_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing
    for from had has have having how i if in into is it its of on or so
    such than that the their them then there these they this those to
    too was we were what when where which who why will with would you
    your not no yes any all more most some between also however whether
    about after again today yesterday latest news happened happening going
    tell me us any update updates week day hours last past recent recently
    """.split()
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9&.\-]*[a-z0-9]|[a-z0-9]")


def _stem(token: str) -> str:
    """Fold plurals so "rates" matches "rate" (deliberately minimal)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def expand_query(text: str) -> str:
    extra = [ABBREVIATIONS[t] for t in _TOKEN_RE.findall(text.lower()) if t in ABBREVIATIONS]
    return " ".join([text, *extra])


def recency_factor(age_hours: float, weight: float = RECENCY_WEIGHT) -> float:
    if weight <= 0:
        return 1.0
    decay = 0.5 ** (max(age_hours, 0.0) / RECENCY_HALF_LIFE_HOURS)
    return (1 - weight) + weight * decay


def _add_rrf(fused: np.ndarray, scores: np.ndarray, positive_only: bool = False) -> None:
    """Add each item's reciprocal-rank-fusion share for one ranking."""
    for rank, idx in enumerate(np.argsort(-scores)):
        if positive_only and scores[idx] <= 0:
            break
        fused[idx] += 1.0 / (RRF_K + rank + 1)


def _floors(dense: np.ndarray, lexical: np.ndarray, use_dense: bool) -> tuple[float, float, float]:
    """(absolute, relative) cosine cut-offs and the BM25 score that counts as a strong keyword match."""
    if not use_dense:  # keyword-only: any match counts
        return -np.inf, np.inf, 1e-9
    relative = max(MIN_DENSE_SIMILARITY, float(dense.max()) - MAX_GAP_FROM_BEST)
    keyword = STRONG_KEYWORD_SHARE * float(lexical.max()) if lexical.max() > 0 else np.inf
    return MIN_DENSE_SIMILARITY, relative, keyword


@dataclass
class Hit:
    story: Story
    score: float
    dense: float  # best cosine similarity among the story's items
    keyword: bool  # True if BM25 matched any of the story's items


class StoryIndex:
    """Search structure over one snapshot of stories."""

    def __init__(self, stories: list[Story]):
        self.stories = stories
        self._items = [(s_idx, item) for s_idx, s in enumerate(stories) for item in s.items]
        self._vectors = (
            np.stack([item.embedding for _, item in self._items]) if self._items else np.zeros((0, 384))
        )
        self._published = [item.published for _, item in self._items]
        corpus = [tokenize(f"{i.title} {i.summary} {' '.join(i.tickers)}") for _, i in self._items]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(
        self,
        query: str,
        k: int = RETRIEVAL_K,
        extra_terms: list[str] | None = None,
        now: float | None = None,
        mode: str = "hybrid",
        recency_weight: float = RECENCY_WEIGHT,
    ) -> list[Hit]:
        """Top-k stories for `query`.

        `mode` ("hybrid", "dense" or "bm25") and `recency_weight` exist so the
        evaluation can compare configurations; the app uses the defaults.
        """
        if not self._items:
            return []
        now = now or time.time()

        use_dense, use_bm25 = mode in ("hybrid", "dense"), mode in ("hybrid", "bm25")
        n = len(self._items)
        dense = self._vectors @ embed_query(query) if use_dense else np.zeros(n)
        terms = tokenize(expand_query(" ".join([query, *(extra_terms or [])])))
        lexical = self._bm25.get_scores(terms) if use_bm25 and terms else np.zeros(n)

        fused = np.zeros(n)
        if use_dense:
            _add_rrf(fused, dense)
        if use_bm25:
            _add_rrf(fused, lexical, positive_only=True)

        absolute_floor, relative_floor, keyword_floor = _floors(dense, lexical, use_dense)
        keyword = lexical >= keyword_floor
        keep = (dense >= absolute_floor) & ((dense >= relative_floor) | keyword)
        recency = np.array([recency_factor((now - p) / 3600, recency_weight) for p in self._published])
        return self._collapse(np.flatnonzero(keep), fused * recency, dense, keyword)[:k]

    def _collapse(self, kept: np.ndarray, scores: np.ndarray, dense: np.ndarray, keyword: np.ndarray) -> list[Hit]:
        """Merge item matches into story hits; a story scores as its best item."""
        best: dict[int, Hit] = {}
        for idx in kept:
            s_idx = self._items[idx][0]
            hit = best.setdefault(s_idx, Hit(self.stories[s_idx], 0.0, -1.0, False))
            hit.score = max(hit.score, float(scores[idx]))
            hit.dense = max(hit.dense, float(dense[idx]))
            hit.keyword = hit.keyword or bool(keyword[idx])
        return sorted(best.values(), key=lambda h: h.score, reverse=True)
