"""SQLite store for news items, their embeddings, and per-feed health.

One small file holds the rolling window of news. Embeddings are cached next
to each item, so a refresh only embeds the handful of headlines that are
new since the last run.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import DB_PATH, ensure_dirs
from src.feeds import FeedResult, NewsItem

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id         TEXT PRIMARY KEY,
    feed       TEXT NOT NULL,
    outlet     TEXT NOT NULL,
    title      TEXT NOT NULL,
    summary    TEXT NOT NULL,
    url        TEXT NOT NULL,
    published  REAL NOT NULL,
    fetched    REAL NOT NULL,
    tickers    TEXT NOT NULL DEFAULT '[]',
    embedding  BLOB,  -- headline + summary: used for search
    title_embedding BLOB  -- headline only: used to spot duplicate stories
);
CREATE INDEX IF NOT EXISTS items_published ON items (published);
CREATE TABLE IF NOT EXISTS feed_status (
    feed        TEXT PRIMARY KEY,
    outlet      TEXT NOT NULL,
    last_try    REAL NOT NULL,
    last_ok     REAL,
    last_error  TEXT NOT NULL DEFAULT '',
    last_count  INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class StoredItem(NewsItem):
    item_id: str = ""
    embedding: np.ndarray | None = None
    title_embedding: np.ndarray | None = None


def connect(path: Path | None = None) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(path or DB_PATH, timeout=30)
    conn.executescript(_SCHEMA)
    return conn


def save_results(conn: sqlite3.Connection, results: list[FeedResult], retention_hours: float) -> int:
    """Insert new items (first sighting wins) and record feed health.

    Items already older than the retention window are skipped: some feeds
    (the BBC's) keep months-old articles in rotation.
    Returns the number of items that were not already stored.
    """
    now = time.time()
    cutoff = now - retention_hours * 3600
    before = conn.total_changes
    for res in results:
        conn.executemany(
            "INSERT OR IGNORE INTO items (id, feed, outlet, title, summary, url, published, fetched, tickers)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (i.id, i.feed, i.outlet, i.title, i.summary, i.url, i.published, now, json.dumps(i.tickers))
                for i in res.items
                if i.published >= cutoff
            ],
        )
    new_items = conn.total_changes - before
    for res in results:
        conn.execute(
            "INSERT INTO feed_status (feed, outlet, last_try, last_ok, last_error, last_count)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (feed) DO UPDATE SET last_try = excluded.last_try,"
            " last_ok = COALESCE(excluded.last_ok, feed_status.last_ok),"
            " last_error = excluded.last_error, last_count = excluded.last_count",
            (res.feed, res.outlet, now, None if res.error else now, res.error, len(res.items)),
        )
    conn.commit()
    return new_items


def unembedded(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """(id, title, title + summary) for every item still missing embeddings."""
    rows = conn.execute(
        "SELECT id, title, summary FROM items WHERE embedding IS NULL OR title_embedding IS NULL"
    ).fetchall()
    return [(r[0], r[1], f"{r[1]}. {r[2]}" if r[2] else r[1]) for r in rows]


def save_embeddings(
    conn: sqlite3.Connection, ids: list[str], vectors: np.ndarray, title_vectors: np.ndarray
) -> None:
    conn.executemany(
        "UPDATE items SET embedding = ?, title_embedding = ? WHERE id = ?",
        [
            (v.astype(np.float32).tobytes(), t.astype(np.float32).tobytes(), i)
            for i, v, t in zip(ids, vectors, title_vectors)
        ],
    )
    conn.commit()


def prune(conn: sqlite3.Connection, retention_hours: float, now: float | None = None) -> int:
    cutoff = (now or time.time()) - retention_hours * 3600
    cur = conn.execute("DELETE FROM items WHERE published < ?", (cutoff,))
    conn.commit()
    return cur.rowcount


def load_items(conn: sqlite3.Connection, window_hours: float, now: float | None = None) -> list[StoredItem]:
    """Embedded items published within the window, newest first."""
    now = now or time.time()
    rows = conn.execute(
        "SELECT id, feed, outlet, title, summary, url, published, tickers, embedding, title_embedding"
        " FROM items WHERE published >= ? AND published <= ? AND title_embedding IS NOT NULL"
        " ORDER BY published DESC",
        (now - window_hours * 3600, now + 3600),  # +1h tolerates feeds with skewed clocks
    ).fetchall()
    return [
        StoredItem(
            feed=r[1], outlet=r[2], title=r[3], summary=r[4], url=r[5], published=r[6],
            tickers=json.loads(r[7]), item_id=r[0],
            embedding=np.frombuffer(r[8], dtype=np.float32),
            title_embedding=np.frombuffer(r[9], dtype=np.float32),
        )
        for r in rows
    ]


def feed_status(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT feed, outlet, last_try, last_ok, last_error, last_count FROM feed_status ORDER BY outlet, feed"
    ).fetchall()
    keys = ("feed", "outlet", "last_try", "last_ok", "last_error", "last_count")
    return [dict(zip(keys, r)) for r in rows]


def last_refresh(conn: sqlite3.Connection) -> float | None:
    return conn.execute("SELECT MAX(last_try) FROM feed_status").fetchone()[0]


def latest_item_time(conn: sqlite3.Connection) -> float | None:
    return conn.execute("SELECT MAX(fetched) FROM items").fetchone()[0]


def snapshot(conn: sqlite3.Connection, dest: Path) -> None:
    """Copy the whole database to `dest` (used to freeze an eval set)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(dest)) as out:
        conn.backup(out)
