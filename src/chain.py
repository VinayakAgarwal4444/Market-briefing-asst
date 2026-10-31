"""Answer a question from the news window: retrieve -> quote -> prompt -> cite.

Two kinds of grounding go into the prompt, each with its own citation tag:

* ``[1]``, ``[2]``, ... -- stories from the news window (``src/retriever.py``)
* ``[M]`` -- live quotes for instruments the question mentions (``src/market.py``)

After generation, every citation tag is checked against what was actually
supplied, so the UI (and the evaluation) can flag an answer that cites a
source it was never given.
"""
from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config import RETRIEVAL_K, SEARCH_WINDOW_HOURS
from src import store
from src.llm import ProviderUnavailableError, get_llm, model_label
from src.market import Quote, detect_symbols, format_quote, get_quotes, symbol_terms
from src.retriever import Hit, StoryIndex
from src.stories import Story, group

NO_COVERAGE = "Nothing in the last {hours} hours of coverage addresses that."

SYSTEM_PROMPT = """You are a market news assistant. It is now {now}.
You answer using ONLY the NEWS and MARKET DATA below, which cover the last {hours} hours of financial news.

Rules:
1. End every factual sentence with the citation of the story it came from, like [2] or [1][4]. Use [M] for figures taken from MARKET DATA.
2. Never state a price, percentage or other number that does not appear in NEWS or MARKET DATA. Do not use what you remember from training: it is out of date.
3. If the question is about a price move and MARKET DATA lists that instrument, give the move from MARKET DATA with [M] in your first sentence.
4. Use only the stories that answer the question. Ignore the others; never mention a story just because it is listed.
5. If no story answers the question, reply with exactly: "{no_coverage}" You may then point to the closest related story, with its citation.
6. Never give buy, sell or hold advice, price targets or predictions of your own. Attribute any analyst's view to that analyst.
7. Mention how recent the news is when it matters, for example "reported 3 hours ago".
8. Start with a direct answer in one or two sentences, then at most four short bullet points if more detail helps. No headings.

MARKET DATA (Yahoo Finance; cite as [M]):
{market}

NEWS (one block per story; the same event reported by several outlets is grouped together):
{news}"""

_CITATION_RE = re.compile(r"\[(\d+|M)\]")
# Advice requests are caught here rather than left to the prompt: an 8B
# model told "never give advice" still slips into "analysts say buy" or a
# confused "nothing was reported". The app shows a fixed notice instead,
# and the model is asked the neutral version of the question.
ADVICE_NOTICE = "I can't give investment advice, but here is what was reported."
_ADVICE_RE = re.compile(
    r"\b(should (i|we)|is it (a )?(good )?time to|worth (buying|selling)|"
    r"is \S+ a (good )?(buy|sell)|"
    r"price target|will \S+ (go|rise|fall|drop|rebound|recover)|predict)",
    re.I,
)
# Follow-ups like "what about AMD?" are searched together with the
# previous question so the topic carries over.
_FOLLOW_UP_MAX_WORDS = 8
_FOLLOW_UP_OPENERS = ("what about", "how about", "and ", "why", "what else", "anything else")


def _is_follow_up(question: str, own_symbols: list[str]) -> bool:
    """Short questions lean on the previous turn -- unless they name their own subject.

    "How many subscribers did Netflix add last quarter?" is short but complete;
    merging it with an earlier AMD question pulled chip stories into the answer.
    """
    if len(question.split()) > _FOLLOW_UP_MAX_WORDS:
        return False
    return not own_symbols or question.lower().lstrip().startswith(_FOLLOW_UP_OPENERS)


def ago(seconds: float) -> str:
    minutes = max(int(seconds // 60), 0)
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    return f"{hours}h ago" if hours < 48 else f"{hours // 24} days ago"


def utc(ts: float) -> str:
    # The year matters: without it the model guessed "October 2026" for a
    # date its sources never gave a year for.
    return time.strftime("%a %d %b %Y %H:%M UTC", time.gmtime(ts))


@dataclass
class Corpus:
    """Everything searchable at one moment: the stories and their index."""

    now: float
    stories: list[Story]
    index: StoryIndex

    @classmethod
    def load(
        cls,
        conn: sqlite3.Connection | None = None,
        now: float | None = None,
        window_hours: float = SEARCH_WINDOW_HOURS,
    ) -> "Corpus":
        now = now or time.time()
        stories = group(store.load_items(conn or store.connect(), window_hours, now=now))
        return cls(now=now, stories=stories, index=StoryIndex(stories))


@dataclass
class Prepared:
    question: str
    hits: list[Hit]
    quotes: list[Quote]
    messages: list[BaseMessage]
    advice: bool = False


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    quotes: list[Quote]
    advice: bool = False
    cited: list[int] = field(default_factory=list)  # 1-based story numbers actually cited
    cited_market: bool = False
    invalid: list[str] = field(default_factory=list)  # tags pointing at nothing supplied
    unsupported: list[str] = field(default_factory=list)  # numbers found in no source

    @property
    def abstained(self) -> bool:
        return self.text.strip().startswith("Nothing in the last")


def format_story(n: int, story: Story, now: float, max_items: int = 3) -> str:
    lead = story.lead
    lines = [f"[{n}] {lead.title}"]
    if lead.summary:
        lines.append(f"    {lead.summary}")
    lines.append(
        f"    Reported by {', '.join(story.outlets)}; first {ago(now - story.first_seen)} "
        f"({utc(story.first_seen)})"
        + (f", latest {ago(now - story.last_seen)}" if len(story.items) > 1 else "")
    )
    others = [i for i in story.items if i is not lead][: max_items - 1]
    for item in others:
        lines.append(f"    Also: \"{item.title}\" ({item.outlet})" + (f" -- {item.summary}" if item.summary else ""))
    return "\n".join(lines)


def build_messages(
    question: str,
    hits: list[Hit],
    quotes: list[Quote],
    now: float,
    history: list[tuple[str, str]] | None = None,
    hours: int = SEARCH_WINDOW_HOURS,
) -> list[BaseMessage]:
    news = "\n\n".join(format_story(n, h.story, now) for n, h in enumerate(hits, 1)) or "(no matching stories)"
    market = "\n".join(f"- {format_quote(q)}" for q in quotes) or "(none requested)"
    system = SYSTEM_PROMPT.format(
        now=utc(now), hours=hours, news=news, market=market, no_coverage=NO_COVERAGE.format(hours=hours)
    )
    messages: list[BaseMessage] = [SystemMessage(system)]
    for user, assistant in (history or [])[-2:]:
        messages += [HumanMessage(user), AIMessage(assistant)]
    messages.append(HumanMessage(question))
    return messages


def prepare(
    corpus: Corpus,
    question: str,
    history: list[tuple[str, str]] | None = None,
    with_quotes: bool = True,
    k: int = RETRIEVAL_K,
) -> Prepared:
    own_symbols = detect_symbols(question)
    search_text = question
    if history and _is_follow_up(question, own_symbols):
        search_text = f"{history[-1][0]} {question}"
    # "What about Intel?" after an Amazon question is about Intel only.
    symbols = own_symbols or detect_symbols(search_text)
    hits = corpus.index.search(search_text, k=k, extra_terms=symbol_terms(symbols), now=corpus.now)
    quotes = get_quotes(symbols) if with_quotes else []
    advice = bool(_ADVICE_RE.search(question))
    # gpt-oss, given only the first line, closed an Amazon answer with "the
    # consensus is that Amazon is still considered a buy" -- advice by proxy.
    prompt = (
        f"Summarise what has been reported that is relevant to this question: {question}\n"
        "Report facts and attribute each view to whoever holds it. Do not say whether to buy, sell or hold, "
        "and do not sum analysts' views up into a verdict such as \"it is still a buy\"."
    ) if advice else question
    messages = build_messages(prompt, hits, quotes, corpus.now, history)
    return Prepared(question, hits, quotes, messages, advice)


def stream(prepared: Prepared) -> Iterator[str]:
    if not prepared.hits and not prepared.quotes:
        # Nothing to ground an answer in: don't give the model the chance to improvise.
        yield NO_COVERAGE.format(hours=SEARCH_WINDOW_HOURS)
        return
    try:
        for chunk in get_llm().stream(prepared.messages):
            if chunk.content:
                yield chunk.content
    except ProviderUnavailableError:
        raise
    except Exception as exc:
        # A retired model, a rate limit or an outage: show a message, not a crash.
        raise ProviderUnavailableError(
            f"{model_label()} returned an error ({type(exc).__name__}). Try again shortly; if it "
            "keeps happening, the model name or API key in the app's settings needs checking."
        ) from exc


def check_citations(text: str, n_hits: int, has_quotes: bool) -> tuple[list[int], bool, list[str]]:
    cited: list[int] = []
    market = False
    invalid: list[str] = []
    for tag in _CITATION_RE.findall(text):
        if tag == "M":
            if has_quotes:
                market = True
            elif "[M]" not in invalid:
                invalid.append("[M]")
        elif 1 <= int(tag) <= n_hits:
            if int(tag) not in cited:
                cited.append(int(tag))
        elif f"[{tag}]" not in invalid:
            invalid.append(f"[{tag}]")
    return cited, market, invalid


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def trim_after_abstaining(text: str, n_hits: int) -> str:
    """After "Nothing ... addresses that.", keep only sentences citing a supplied story.

    The prompt allows pointing to "the closest related story". In the
    evaluation the model used that opening to invent one -- a made-up
    Netflix quote, cited as [1] when it had been given no stories at all.
    """
    head, _, tail = text.strip().partition(".")
    kept = [head + "."]
    for sentence in _SENTENCE_END_RE.split(tail.strip()):
        cited, _, invalid = check_citations(sentence, n_hits, has_quotes=False)
        if cited and not invalid:
            kept.append(sentence.strip())
    return " ".join(kept)


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def unsupported_numbers(text: str, prepared: Prepared) -> list[str]:
    """Numbers in the answer that appear nowhere in what the model was given.

    In the evaluation the model once added "better than the expected 2.8%"
    to a GDP answer -- a figure in none of its sources. Wrong numbers are the
    costliest mistake in a finance tool, so the app checks every answer.
    """
    source = " ".join(str(m.content) for m in prepared.messages).replace(",", "")
    found: list[str] = []
    for raw in _NUMBER_RE.findall(_CITATION_RE.sub("", text)):
        if raw.replace(",", "").rstrip(".") not in source and raw not in found:
            found.append(raw)
    return found


# gpt-oss was trained to cite with lenticular brackets (U+3010/U+3011),
# with or without a line reference after a dagger, and sometimes does so
# despite the prompt. Normalise to [1] before anything checks citations.
_LENTICULAR_RE = re.compile("\u3010(\\d+|M)[^\u3011]*\u3011")


def normalize_citations(text: str) -> str:
    return _LENTICULAR_RE.sub(r"[\1]", text)


def finish(prepared: Prepared, text: str) -> Answer:
    text = normalize_citations(text)
    if text.strip().startswith("Nothing in the last"):
        text = trim_after_abstaining(text, len(prepared.hits))
        rest = text.partition(".")[2].strip()
        if prepared.advice and rest:
            # "Should I buy?" has no reported answer, so the model opens with
            # "Nothing ... addresses that" before the relevant news. The app's
            # no-advice notice already frames the answer; the opener only confuses.
            text = rest[len("However, "):] if rest.startswith("However, ") else rest
            text = text[:1].upper() + text[1:]
    cited, market, invalid = check_citations(text, len(prepared.hits), bool(prepared.quotes))
    return Answer(
        text, prepared.hits, prepared.quotes, prepared.advice, cited, market, invalid,
        unsupported_numbers(text, prepared),
    )


def ask(corpus: Corpus, question: str, **kwargs) -> Answer:
    prepared = prepare(corpus, question, **kwargs)
    return finish(prepared, "".join(stream(prepared)))
