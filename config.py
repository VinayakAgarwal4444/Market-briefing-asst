"""Central configuration for the RAG teaching assistant.

Single source of truth for paths, model names, and tunables.
Values can be overridden via environment variables (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
VECTORSTORE_DIR: Path = PROJECT_ROOT / "vectorstore"
EVAL_DIR: Path = PROJECT_ROOT / "eval"

# --- Provider selection -----------------------------------------------------
# Each side of the stack (LLM and embeddings) is configurable independently
# so local dev (Ollama, free, GPU-accelerated) and cloud deploy (Groq +
# HuggingFace Inference, free tier, fits in Streamlit Cloud) share one
# codebase.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()  # ollama | groq
EMBED_PROVIDER: str = os.getenv("EMBED_PROVIDER", "ollama").lower()  # ollama | hf_endpoint

# --- Ollama (local) ---------------------------------------------------------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- Groq (cloud LLM) -------------------------------------------------------
# Free tier, blazing fast, no card. Get a key at https://console.groq.com.
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL: str = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")

# --- HuggingFace Inference (cloud embeddings) -------------------------------
# Free tier; the model below is the SAME nomic-embed-text-v1.5 we use locally
# via Ollama, so swapping providers does not require re-indexing.
# Get a token at https://huggingface.co/settings/tokens.
HF_TOKEN: str = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_EMBED_MODEL: str = os.getenv("HF_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "8"))
RETRIEVAL_FETCH_K: int = int(os.getenv("RETRIEVAL_FETCH_K", "20"))
RETRIEVAL_SEARCH_TYPE: str = os.getenv("RETRIEVAL_SEARCH_TYPE", "mmr")
RETRIEVAL_HYBRID: bool = os.getenv("RETRIEVAL_HYBRID", "true").lower() == "true"
RETRIEVAL_BM25_WEIGHT: float = float(os.getenv("RETRIEVAL_BM25_WEIGHT", "0.5"))

REQUEST_USER_AGENT: str = (
    "rag-teaching-assistant/0.1 (educational project; contact: student@example.com)"
)
REQUEST_DELAY_SECONDS: float = 1.0


def ensure_dirs() -> None:
    """Create all project directories if they do not already exist."""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
