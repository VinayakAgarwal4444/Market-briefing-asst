"""Local sentence embeddings via fastembed (ONNX on CPU, no API key).

Vectors come back L2-normalised, so a dot product is cosine similarity.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return np.array(list(_model().embed(texts)), dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return np.array(next(iter(_model().query_embed([text]))), dtype=np.float32)
