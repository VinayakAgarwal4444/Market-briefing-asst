"""Smoke test: verify the provider refactor compiles and dispatches correctly.

We deliberately do NOT instantiate the cloud factories (Groq / HF) here -
HuggingFaceEndpointEmbeddings validates the endpoint at construction
time, which makes a network round-trip and is unfriendly to a smoke test.
Instead we:

1. Import every module in the refactored chain (so import errors fail fast).
2. Confirm Groq/HF langchain packages are importable (the deploy depends
   on them).
3. Build the *real* Ollama factories and ping the model to confirm the
   non-cloud path still works end-to-end.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    print("[smoke] (1/3) Project imports compile...")
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["EMBED_PROVIDER"] = "ollama"
    import config  # noqa: F401
    from src.chain import get_chain  # noqa: F401
    from src.llm import (  # noqa: F401
        ProviderUnavailableError,
        get_embeddings,
        get_llm,
    )
    from src.retriever import get_retriever  # noqa: F401

    print("        ok")

    print("[smoke] (2/3) Cloud provider packages importable...")
    from langchain_groq import ChatGroq  # noqa: F401
    from langchain_huggingface import HuggingFaceEndpointEmbeddings  # noqa: F401

    print("        ok (langchain_groq, langchain_huggingface)")

    print("[smoke] (3/3) Ollama path still works (no refactor regression)...")
    llm = get_llm()
    emb = get_embeddings()
    print(f"        llm  -> {llm.__class__.__name__}")
    print(f"        emb  -> {emb.__class__.__name__}")
    test_vec = emb.embed_query("hello")
    print(f"        embed_query('hello') -> dim {len(test_vec)} (first 3: {test_vec[:3]})")

    print("\n[smoke] OK -- refactor compiles and the local path is intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
