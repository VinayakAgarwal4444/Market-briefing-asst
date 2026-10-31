"""Central configuration for the market briefing assistant.

Single source of truth for feeds, paths, model names, and tunables.
Values can be overridden via environment variables (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
DB_PATH: Path = Path(os.getenv("DB_PATH", str(DATA_DIR / "news.db")))
EVAL_DIR: Path = PROJECT_ROOT / "eval"

# --- News sources -----------------------------------------------------------
# Free, public RSS/Atom feeds. Only the headline, the feed's own summary and
# the link are stored -- never the full article -- so every answer sends the
# reader back to the publisher.
FEEDS: dict[str, tuple[str, str]] = {
    # key: (outlet label, feed URL)
    "cnbc_top": ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    "cnbc_economy": ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    "cnbc_investing": ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"),
    "marketwatch": ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    "ft_markets": ("Financial Times", "https://www.ft.com/markets?format=rss"),
    "bbc_business": ("BBC", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    "guardian_economics": ("The Guardian", "https://www.theguardian.com/business/economics/rss"),
    "nasdaq_markets": ("Nasdaq", "https://www.nasdaq.com/feed/rssoutbound?category=Markets"),
    "seeking_alpha": ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    "fed": ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
    "ecb": ("ECB", "https://www.ecb.europa.eu/rss/press.html"),
}

# SEC EDGAR asks every client to identify itself with a contact email
# ("Name email@domain"). Leave unset to skip the 8-K filings feed.
SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "")
EDGAR_8K_URL: str = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K"
    "&company=&dateb=&owner=include&count=100&output=atom"
)

BROWSER_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

# --- Time windows -----------------------------------------------------------
BRIEFING_WINDOW_HOURS: int = int(os.getenv("BRIEFING_WINDOW_HOURS", "24"))
SEARCH_WINDOW_HOURS: int = int(os.getenv("SEARCH_WINDOW_HOURS", "48"))
RETENTION_HOURS: int = int(os.getenv("RETENTION_HOURS", "72"))
# The app refreshes the feeds itself when the newest fetch is older than this.
REFRESH_MINUTES: int = int(os.getenv("REFRESH_MINUTES", "20"))

# --- LLM provider -----------------------------------------------------------
# Local dev runs on Ollama (free, GPU); the cloud deploy swaps to Groq.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()  # ollama | groq

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

# Free tier, no card. Get a key at https://console.groq.com.
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL: str = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")

# --- Embeddings -------------------------------------------------------------
# Runs in-process on CPU via ONNX (fastembed), so local and cloud use the
# exact same model with no API key. 384 dimensions, ~130 MB download.
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Retrieval --------------------------------------------------------------
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "8"))
# Reciprocal-rank-fusion constant; 60 is the value from the original paper.
RRF_K: int = int(os.getenv("RRF_K", "60"))
# Search recency boost: a story loses up to this share of its score as it
# ages, halving every RECENCY_HALF_LIFE_HOURS. Off by default -- it lowered
# ranking quality in the evaluation (see src/retriever.py).
RECENCY_WEIGHT: float = float(os.getenv("RECENCY_WEIGHT", "0"))
# The briefing ranks "most important in the last 24 hours", where freshness
# is the point, so it keeps its own, stronger weight.
BRIEFING_RECENCY_WEIGHT: float = float(os.getenv("BRIEFING_RECENCY_WEIGHT", "0.4"))
RECENCY_HALF_LIFE_HOURS: float = float(os.getenv("RECENCY_HALF_LIFE_HOURS", "12"))
# Cosine similarity above which two headlines are treated as the same story.
DUPLICATE_THRESHOLD: float = float(os.getenv("DUPLICATE_THRESHOLD", "0.8"))


def ensure_dirs() -> None:
    """Create all project directories if they do not already exist."""
    for d in (DATA_DIR, DB_PATH.parent, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
