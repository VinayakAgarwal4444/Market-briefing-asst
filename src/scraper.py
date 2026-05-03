"""Polite scraper for curated pandas/NumPy/scikit-learn documentation pages.

Each URL is fetched with a real User-Agent and a 1s delay between requests.
Raw HTML is cached under data/raw/<source>/<slug>.html so re-runs are free.
The main content area is extracted, converted to markdown, and written to
data/processed/<source>/<slug>.md with YAML front-matter holding source_url
and title (used later for citations in the RAG chain).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from config import (
    PROCESSED_DIR,
    RAW_DIR,
    REQUEST_DELAY_SECONDS,
    REQUEST_USER_AGENT,
    ensure_dirs,
)


@dataclass(frozen=True)
class Source:
    name: str
    urls: tuple[str, ...]
    # CSS selector(s) for the main content; first match wins.
    content_selectors: tuple[str, ...]


SOURCES: tuple[Source, ...] = (
    Source(
        name="pandas",
        urls=(
            "https://pandas.pydata.org/docs/user_guide/10min.html",
            "https://pandas.pydata.org/docs/user_guide/io.html",
            "https://pandas.pydata.org/docs/user_guide/indexing.html",
            "https://pandas.pydata.org/docs/user_guide/groupby.html",
            "https://pandas.pydata.org/docs/user_guide/missing_data.html",
            "https://pandas.pydata.org/docs/user_guide/merging.html",
            "https://pandas.pydata.org/docs/user_guide/dsintro.html",
            "https://pandas.pydata.org/docs/user_guide/basics.html",
            "https://pandas.pydata.org/docs/user_guide/reshaping.html",
            "https://pandas.pydata.org/docs/user_guide/timeseries.html",
        ),
        content_selectors=("article.bd-article", "main", "div.document"),
    ),
    Source(
        name="numpy",
        urls=(
            "https://numpy.org/doc/stable/user/quickstart.html",
            "https://numpy.org/doc/stable/user/absolute_beginners.html",
            "https://numpy.org/doc/stable/user/basics.creation.html",
            "https://numpy.org/doc/stable/user/basics.indexing.html",
            "https://numpy.org/doc/stable/user/basics.broadcasting.html",
            "https://numpy.org/doc/stable/user/basics.copies.html",
            "https://numpy.org/doc/stable/user/basics.types.html",
            "https://numpy.org/doc/stable/user/how-to-index.html",
        ),
        content_selectors=("article.bd-article", "main", "div.document"),
    ),
    Source(
        name="sklearn",
        urls=(
            "https://scikit-learn.org/stable/getting_started.html",
            "https://scikit-learn.org/stable/modules/cross_validation.html",
            "https://scikit-learn.org/stable/modules/compose.html",
            "https://scikit-learn.org/stable/modules/preprocessing.html",
            "https://scikit-learn.org/stable/modules/model_evaluation.html",
            "https://scikit-learn.org/stable/modules/linear_model.html",
            "https://scikit-learn.org/stable/modules/tree.html",
            "https://scikit-learn.org/stable/modules/ensemble.html",
            "https://scikit-learn.org/stable/modules/svm.html",
            "https://scikit-learn.org/stable/common_pitfalls.html",
        ),
        content_selectors=("article.bd-article", "main", "div.document"),
    ),
)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    if path.endswith(".html"):
        path = path[: -len(".html")]
    return _SLUG_RE.sub("-", path.lower()).strip("-") or "index"


def _fetch(url: str, raw_path: Path) -> str:
    """Fetch URL and cache to disk; return HTML text."""
    if raw_path.exists():
        return raw_path.read_text(encoding="utf-8")

    print(f"[scrape] GET {url}")
    headers = {"User-Agent": REQUEST_USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(resp.text, encoding="utf-8")
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.text


def _extract_main(html: str, selectors: tuple[str, ...]) -> tuple[str, str]:
    """Return (title, cleaned_markdown) from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""

    main = None
    for sel in selectors:
        main = soup.select_one(sel)
        if main is not None:
            break
    if main is None:
        main = soup.body or soup

    # Strip noise: navigation, sidebars, footers, scripts, styles.
    for tag in main.select(
        "nav, aside, footer, script, style, "
        ".prev-next-area, .bd-sidebar, .headerlink, .toctree-wrapper"
    ):
        tag.decompose()

    md = markdownify(str(main), heading_style="ATX", strip=["a"])
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return title, md


def _write_processed(
    processed_path: Path, title: str, source_url: str, content_md: str
) -> None:
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    safe_title = title.replace('"', "'")
    front_matter = (
        "---\n"
        f'title: "{safe_title}"\n'
        f"source_url: {source_url}\n"
        "---\n\n"
    )
    processed_path.write_text(front_matter + content_md, encoding="utf-8")


def run() -> int:
    """Scrape all SOURCES; return number of pages written."""
    ensure_dirs()
    written = 0
    for source in SOURCES:
        for url in source.urls:
            slug = _slugify_url(url)
            raw_path = RAW_DIR / source.name / f"{slug}.html"
            processed_path = PROCESSED_DIR / source.name / f"{slug}.md"
            try:
                html = _fetch(url, raw_path)
                title, md = _extract_main(html, source.content_selectors)
                _write_processed(processed_path, title, url, md)
                written += 1
            except requests.RequestException as exc:
                print(f"[scrape] FAILED {url}: {exc}")
            except Exception as exc:  # noqa: BLE001 - keep one failure isolated
                print(f"[scrape] PARSE FAILED {url}: {exc}")
    print(f"[scrape] wrote {written} processed pages to {PROCESSED_DIR}")
    return written


if __name__ == "__main__":
    run()
