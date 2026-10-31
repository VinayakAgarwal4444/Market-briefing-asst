"""The "last 24 hours" briefing shown when the app opens.

Candidate stories are ranked without the LLM -- by how many outlets covered
them, how many articles they produced and how recent they are -- and the
LLM only writes up the top of that list. Ranking stays cheap, repeatable
and explainable; the model does what it is good at.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from config import BRIEFING_RECENCY_WEIGHT, BRIEFING_WINDOW_HOURS
from src.chain import Corpus, Prepared, format_story, utc
from src.market import Quote, format_quote
from src.retriever import Hit, recency_factor
from src.stories import Story

CANDIDATES = 14

BRIEFING_PROMPT = """You write a short market briefing. It is now {now}.
Use ONLY the STORIES and MARKET DATA below, which cover the last {hours} hours.

Write:
1. A line starting "Markets:" that states the moves in MARKET DATA in one sentence, ending with [M].
2. Then the five most market-moving stories as bullet points, most important first. Each bullet starts with a bold label of two to four words, then one or two sentences, and ends with the story's citation, like [3].

Rules: never state a number that is not in STORIES or MARKET DATA. No advice or predictions. No headings, no introduction, no conclusion.

MARKET DATA (Yahoo Finance; cite as [M]):
{market}

STORIES (numbered; the same event reported by several outlets is grouped):
{news}"""


def _is_noise(story: Story) -> bool:
    """Filings and call transcripts are searchable but too granular to headline."""
    lead = story.lead
    return lead.feed == "sec_8k" or "Earnings Call Transcript" in lead.title


def importance(story: Story, now: float) -> float:
    coverage = len(story.outlets) + 0.3 * (len(story.items) - 1)
    return coverage * recency_factor((now - story.last_seen) / 3600, BRIEFING_RECENCY_WEIGHT)


def top_stories(corpus: Corpus, hours: int = BRIEFING_WINDOW_HOURS, n: int = CANDIDATES) -> list[Story]:
    cutoff = corpus.now - hours * 3600
    recent = [s for s in corpus.stories if s.last_seen >= cutoff and not _is_noise(s)]
    return sorted(recent, key=lambda s: importance(s, corpus.now), reverse=True)[:n]


def prepare_briefing(corpus: Corpus, quotes: list[Quote], hours: int = BRIEFING_WINDOW_HOURS) -> Prepared:
    stories = top_stories(corpus, hours)
    hits = [Hit(s, importance(s, corpus.now), 1.0, False) for s in stories]
    news = "\n\n".join(format_story(n, s, corpus.now) for n, s in enumerate(stories, 1)) or "(no stories)"
    market = "\n".join(f"- {format_quote(q)}" for q in quotes) or "(unavailable)"
    system = BRIEFING_PROMPT.format(now=utc(corpus.now), hours=hours, market=market, news=news)
    messages = [SystemMessage(system), HumanMessage("Write the briefing.")]
    return Prepared("briefing", hits, quotes, messages)
