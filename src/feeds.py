"""Fetch the configured RSS/Atom feeds and normalise them into NewsItems.

Each feed has its own quirks (HTML in summaries, tracking parameters in
links, summaries that just repeat the headline); they are all smoothed out
here so the rest of the pipeline sees one clean shape.
"""
from __future__ import annotations

import calendar
import hashlib
import html
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import feedparser
import requests

from config import BROWSER_USER_AGENT, EDGAR_8K_URL, FEEDS, SEC_USER_AGENT

SUMMARY_MAX_CHARS = 400


@dataclass
class NewsItem:
    feed: str
    outlet: str
    title: str
    summary: str
    url: str
    published: float  # unix seconds, UTC
    tickers: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return hashlib.sha1((self.url or self.outlet + self.title).encode()).hexdigest()[:16]

    @property
    def text(self) -> str:
        """What gets embedded and searched."""
        return f"{self.title}. {self.summary}" if self.summary else self.title


@dataclass
class FeedResult:
    feed: str
    outlet: str
    items: list[NewsItem]
    error: str = ""


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #

_TAG_RE = re.compile(r"<[^>]+>")
# Nasdaq summaries can start with an inline stylesheet ("@media (...) { ... }").
_STYLE_BLOCK_RE = re.compile(r"<(style|script)\b.*?</\1>", re.I | re.S)
_CSS_RULE_RE = re.compile(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}")
_WS_RE = re.compile(r"\s+")
# Boilerplate some feeds append to (or prefix onto) every summary.
_BOILERPLATE = (
    re.compile(r"^Key Points\s*"),  # Nasdaq
    re.compile(r"^Image source:.*$"),  # Nasdaq: a photo credit is the whole summary
    re.compile(r"Get our breaking news email.*$", re.I),  # The Guardian
    re.compile(r"Continue reading\.*\s*$", re.I),
    re.compile(r"This live blog is now closed\.?", re.I),
)


def clean_text(raw: str) -> str:
    text = _STYLE_BLOCK_RE.sub(" ", raw or "")
    text = html.unescape(_TAG_RE.sub(" ", text))
    text = _CSS_RULE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    for pattern in _BOILERPLATE:
        text = pattern.sub("", text).strip()
    return text


def _shorten(text: str, limit: int = SUMMARY_MAX_CHARS) -> str:
    """Trim to `limit` characters, preferably at a sentence end."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = cut.rfind(". ")
    return cut[: end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0] + "..."


def canonical_url(url: str) -> str:
    """Drop query strings and fragments (tracking tags) and doubled slashes."""
    parts = urlsplit(url.strip())
    path = re.sub(r"/{2,}", "/", parts.path)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _published(entry) -> float | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return float(calendar.timegm(parsed)) if parsed else None


# --------------------------------------------------------------------------- #
# Feed parsers
# --------------------------------------------------------------------------- #


def _parse_rss(feed: str, outlet: str, content: bytes, fetched: float) -> list[NewsItem]:
    items = []
    for entry in feedparser.parse(content).entries:
        title = clean_text(entry.get("title", ""))
        if not title:
            continue
        summary = clean_text(entry.get("summary", ""))
        if summary.rstrip(".") == title.rstrip("."):  # Fed: summary == title
            summary = ""
        # Seeking Alpha tags entries with lower-case tickers ("intc"), plus
        # futures codes ("cl1:com") which we skip.
        tickers = [
            t["term"].upper()
            for t in entry.get("tags", [])
            if feed == "seeking_alpha" and t.get("term") and ":" not in t["term"]
        ]
        items.append(
            NewsItem(
                feed=feed,
                outlet=outlet,
                title=title,
                summary=_shorten(summary),
                url=canonical_url(entry.get("link", "")),
                published=_published(entry) or fetched,
                tickers=tickers,
            )
        )
    return items


# Only the 8-K items a market reader cares about; the rest (votes, Reg FD
# exhibits, ...) would flood the index with hundreds of non-events a day.
EDGAR_ITEMS = {
    "1.01": "entered a material agreement",
    "1.03": "filed for bankruptcy or receivership",
    "2.01": "completed an acquisition or disposal",
    "2.02": "reported results of operations",
    "2.05": "announced exit or restructuring costs",
    "2.06": "reported a material impairment",
    "3.01": "received a delisting notice",
    "4.02": "said prior financial statements can no longer be relied on",
    "5.02": "announced a director or officer change",
}
_EDGAR_TITLE_RE = re.compile(r"^8-K(?:/A)? - (?P<company>.+?) \(\d+\)")
_EDGAR_ITEM_RE = re.compile(r"Item (\d\.\d\d)")


def _parse_edgar(content: bytes, fetched: float) -> list[NewsItem]:
    items = []
    for entry in feedparser.parse(content).entries:
        match = _EDGAR_TITLE_RE.match(entry.get("title", ""))
        if not match:
            continue
        codes = [c for c in _EDGAR_ITEM_RE.findall(entry.get("summary", "")) if c in EDGAR_ITEMS]
        if not codes:
            continue
        company = match.group("company").title()
        events = "; ".join(EDGAR_ITEMS[c] for c in codes)
        items.append(
            NewsItem(
                feed="sec_8k",
                outlet="SEC EDGAR",
                title=f"{company} 8-K filing: {events}",
                summary=f"{company} filed a Form 8-K covering item(s) {', '.join(codes)}.",
                url=entry.get("link", ""),
                published=_published(entry) or fetched,
            )
        )
    return items


def _fetch_one(feed: str, outlet: str, url: str) -> FeedResult:
    is_sec = feed == "sec_8k"
    headers = {"User-Agent": SEC_USER_AGENT if is_sec else BROWSER_USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return FeedResult(feed, outlet, [], error=str(exc)[:200])
    fetched = time.time()
    items = _parse_edgar(resp.content, fetched) if is_sec else _parse_rss(feed, outlet, resp.content, fetched)
    if not items and not is_sec:
        return FeedResult(feed, outlet, [], error="feed returned no entries")
    return FeedResult(feed, outlet, items)


def fetch_all() -> list[FeedResult]:
    """Fetch every configured feed in parallel."""
    jobs = [(key, outlet, url) for key, (outlet, url) in FEEDS.items()]
    if SEC_USER_AGENT:
        jobs.append(("sec_8k", "SEC EDGAR", EDGAR_8K_URL))
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda job: _fetch_one(*job), jobs))
