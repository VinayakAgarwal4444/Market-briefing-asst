"""Chat-model factory with provider abstraction.

* ``ollama`` (default): local, GPU, free.
* ``groq``: cloud, free tier -- used by the Streamlit Cloud deploy.

Pick one with the ``LLM_PROVIDER`` env var. Embeddings are not here: they
run in-process (see ``src/embed.py``) and are the same everywhere.
"""
from __future__ import annotations

import sys
from functools import lru_cache

import requests
from langchain_core.language_models import BaseChatModel

from config import (
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    GROQ_REASONING_EFFORT,
    LLM_PROVIDER,
    OLLAMA_HOST,
    OLLAMA_LLM_MODEL,
)


class ProviderUnavailableError(RuntimeError):
    """Raised when the configured provider cannot be reached or is misconfigured."""


def _check_ollama(model: str) -> None:
    """Verify the Ollama server is reachable and the model is pulled."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ProviderUnavailableError(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            f"Start it with `ollama serve` and try again. (Underlying error: {exc})"
        ) from exc

    installed = {m["name"] for m in resp.json().get("models", [])}
    if model not in installed and f"{model}:latest" not in installed:
        raise ProviderUnavailableError(f"Ollama model {model} is not pulled. Run: ollama pull {model}")


def model_label() -> str:
    if LLM_PROVIDER == "groq":
        return f"Groq · {GROQ_LLM_MODEL}"
    return f"Ollama · {OLLAMA_LLM_MODEL}"


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a cached chat model for the configured provider.

    Temperature is pinned to 0 for repeatable, grounded answers.
    """
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        _check_ollama(OLLAMA_LLM_MODEL)
        return ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.0)

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ProviderUnavailableError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY. Get a free key at https://console.groq.com."
            )
        from langchain_groq import ChatGroq

        # The model's reasoning arrives in a separate field, so only the answer is streamed.
        return ChatGroq(
            model=GROQ_LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.0,
            reasoning_effort=GROQ_REASONING_EFFORT or None,
        )

    raise ProviderUnavailableError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Expected one of: ollama, groq.")


if __name__ == "__main__":
    try:
        llm = get_llm()
    except ProviderUnavailableError as exc:
        print(f"[llm] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[llm] OK -- {model_label()} ({llm.__class__.__name__})")
