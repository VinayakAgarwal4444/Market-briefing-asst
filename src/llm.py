"""LLM + embeddings factories with provider abstraction.

Two providers are supported on each side:

* **LLM**: ``ollama`` (default, local, GPU) or ``groq`` (cloud, free tier).
* **Embeddings**: ``ollama`` (default) or ``hf_endpoint`` (HuggingFace
  Inference API). The HF endpoint serves the same ``nomic-embed-text-v1.5``
  weights Ollama bundles, so an index built locally is reusable in cloud
  deploys with no re-indexing.

Pick a provider with the ``LLM_PROVIDER`` / ``EMBED_PROVIDER`` env vars.
"""
from __future__ import annotations

import sys
from functools import lru_cache

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import (
    EMBED_PROVIDER,
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    HF_EMBED_MODEL,
    HF_TOKEN,
    LLM_PROVIDER,
    OLLAMA_EMBED_MODEL,
    OLLAMA_HOST,
    OLLAMA_LLM_MODEL,
)


class ProviderUnavailableError(RuntimeError):
    """Raised when a configured provider cannot be reached or is misconfigured."""


# Backwards-compat alias -- existing callers (and tests) still import this name.
OllamaUnavailableError = ProviderUnavailableError


# --------------------------------------------------------------------------- #
# Ollama health check
# --------------------------------------------------------------------------- #


def _check_ollama(required_models: list[str]) -> None:
    """Verify the Ollama server is reachable and required models are pulled."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ProviderUnavailableError(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            f"Start it with `ollama serve` and try again. (Underlying error: {exc})"
        ) from exc

    installed = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
    missing = [m for m in required_models if m.split(":")[0] not in installed]
    if missing:
        raise ProviderUnavailableError(
            "Required Ollama models are not pulled: "
            + ", ".join(missing)
            + ". Run: "
            + " && ".join(f"ollama pull {m}" for m in missing)
        )


# --------------------------------------------------------------------------- #
# Embeddings factory
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached Embeddings instance for the configured provider."""
    if EMBED_PROVIDER == "ollama":
        # Imported lazily so cloud deployments without Ollama don't break.
        from langchain_ollama import OllamaEmbeddings

        _check_ollama([OLLAMA_EMBED_MODEL])
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_HOST)

    if EMBED_PROVIDER == "hf_endpoint":
        if not HF_TOKEN:
            raise ProviderUnavailableError(
                "EMBED_PROVIDER=hf_endpoint requires HF_TOKEN. "
                "Get a free token at https://huggingface.co/settings/tokens."
            )
        try:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
        except ImportError as exc:
            raise ProviderUnavailableError(
                "langchain-huggingface is not installed. Install it with: "
                "pip install langchain-huggingface"
            ) from exc

        return HuggingFaceEndpointEmbeddings(
            model=HF_EMBED_MODEL,
            huggingfacehub_api_token=HF_TOKEN,
        )

    raise ProviderUnavailableError(
        f"Unknown EMBED_PROVIDER={EMBED_PROVIDER!r}. "
        "Expected one of: ollama, hf_endpoint."
    )


# --------------------------------------------------------------------------- #
# Chat LLM factory
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a cached chat-model instance for the configured provider.

    Temperature is pinned to 0 for deterministic, grounded answers.
    """
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        _check_ollama([OLLAMA_LLM_MODEL])
        return ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.0,
        )

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ProviderUnavailableError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY. "
                "Get a free key at https://console.groq.com."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ProviderUnavailableError(
                "langchain-groq is not installed. Install it with: "
                "pip install langchain-groq"
            ) from exc

        return ChatGroq(
            model=GROQ_LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.0,
        )

    raise ProviderUnavailableError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Expected one of: ollama, groq."
    )


# --------------------------------------------------------------------------- #
# CLI sanity check
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    try:
        emb = get_embeddings()
        llm = get_llm()
    except ProviderUnavailableError as exc:
        print(f"[providers] {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[providers] OK -- LLM={LLM_PROVIDER} ({llm.__class__.__name__}), "
        f"embeddings={EMBED_PROVIDER} ({emb.__class__.__name__})"
    )
