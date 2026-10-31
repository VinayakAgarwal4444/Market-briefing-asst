"""Group items from different outlets that report the same event into Stories.

The same news breaks on CNBC, the FT and the BBC within the hour. Grouping
them lets search return one slot per *event* instead of five near-identical
headlines, and the number of outlets covering an event is a free signal
for how big it is.

Grouping compares *headlines only*. Summaries made it worse in testing:
a live blog whose summary rounds up the day's news resembles every story
it mentions and pulled unrelated events into one group.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from config import DUPLICATE_THRESHOLD
from src.store import StoredItem

# "Cavco (CVCO) Q2 Earnings Call Transcript" vs "Covenant Logistics (CVLG)
# Q3 ..." look alike to an embedding model; different tickers settle it.
_TITLE_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")


@dataclass
class Story:
    items: list[StoredItem]  # earliest first

    @property
    def lead(self) -> StoredItem:
        """The most informative item: longest summary, earliest on ties."""
        return max(self.items, key=lambda i: (len(i.summary), -i.published))

    @property
    def first_seen(self) -> float:
        return self.items[0].published

    @property
    def last_seen(self) -> float:
        return self.items[-1].published

    @property
    def outlets(self) -> list[str]:
        return list(dict.fromkeys(i.outlet for i in self.items))

    @property
    def tickers(self) -> list[str]:
        return list(dict.fromkeys(t for i in self.items for t in i.tickers))


def _tickers(item: StoredItem) -> set[str]:
    return set(item.tickers) | set(_TITLE_TICKER_RE.findall(item.title))


def group(items: list[StoredItem], threshold: float = DUPLICATE_THRESHOLD) -> list[Story]:
    """Single-pass clustering in publication order.

    Each item joins the story containing its most similar earlier headline,
    if that similarity reaches `threshold` and the two don't name different
    tickers; otherwise it starts a new story.
    """
    if not items:
        return []
    ordered = sorted(items, key=lambda i: i.published)
    vectors = np.stack([i.title_embedding for i in ordered])
    sims = vectors @ vectors.T
    tickers = [_tickers(i) for i in ordered]

    story_of: list[int] = []
    members: list[list[int]] = []
    for idx in range(len(ordered)):
        best, best_sim = -1, threshold
        for prev in range(idx):
            if sims[idx, prev] < best_sim:
                continue
            if tickers[idx] and tickers[prev] and not tickers[idx] & tickers[prev]:
                continue
            best, best_sim = story_of[prev], sims[idx, prev]
        if best < 0:
            best = len(members)
            members.append([])
        members[best].append(idx)
        story_of.append(best)
    return [Story([ordered[i] for i in m]) for m in members]
