"""One refresh cycle: fetch feeds -> store new items -> embed them -> prune."""
from __future__ import annotations

import time
from dataclasses import dataclass

from config import RETENTION_HOURS
from src import store
from src.embed import embed_passages
from src.feeds import fetch_all


@dataclass
class RefreshStats:
    fetched: int
    new: int
    embedded: int
    pruned: int
    failed_feeds: list[str]
    seconds: float


def run(conn=None) -> RefreshStats:
    started = time.time()
    conn = conn or store.connect()
    results = fetch_all()
    new = store.save_results(conn, results, RETENTION_HOURS)

    pending = store.unembedded(conn)
    if pending:
        ids, titles, texts = zip(*pending)
        store.save_embeddings(conn, list(ids), embed_passages(list(texts)), embed_passages(list(titles)))

    return RefreshStats(
        fetched=sum(len(r.items) for r in results),
        new=new,
        embedded=len(pending),
        pruned=store.prune(conn, RETENTION_HOURS),
        failed_feeds=[f"{r.feed}: {r.error}" for r in results if r.error],
        seconds=time.time() - started,
    )
