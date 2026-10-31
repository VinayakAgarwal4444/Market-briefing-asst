"""Streamlit app: market strip, 24-hour briefing, cited Q&A, top stories, feed health.

Bootstrap order matters: ``st.secrets`` is mirrored into ``os.environ``
*before* ``config`` (and therefore any other project module) is imported,
so the same ``os.getenv``-based config works locally and on Streamlit Cloud.
"""
from __future__ import annotations

import dataclasses
import html
import os
import re
import threading
import time

import streamlit as st

# --- Streamlit Cloud secrets -> env vars (must run before our imports) ----- #
try:
    for _key, _value in st.secrets.items():
        # Don't clobber a value the developer set in their shell.
        if isinstance(_value, str) and _key not in os.environ:
            os.environ[_key] = _value
except (FileNotFoundError, AttributeError):
    # No secrets.toml locally -- that's fine, we'll fall through to .env.
    pass

# --- Project imports (must come AFTER the secrets shim) -------------------- #
from config import BRIEFING_WINDOW_HOURS, REFRESH_MINUTES, SEARCH_WINDOW_HOURS  # noqa: E402
from src import refresh, store  # noqa: E402
from src.briefing import prepare_briefing, top_stories  # noqa: E402
from src.chain import ADVICE_NOTICE, Answer, Corpus, Prepared, ago, finish, prepare, stream  # noqa: E402
from src.llm import ProviderUnavailableError, model_label  # noqa: E402
from src.market import WATCHLIST, Quote, get_quotes  # noqa: E402

st.set_page_config(page_title="Market Briefing", page_icon=":chart_with_upwards_trend:", layout="centered")

st.html(
    """<style>
    .ticker { display: flex; flex-wrap: wrap; gap: .4rem 1.4rem; margin: .2rem 0 1rem; font-size: .9rem; }
    .ticker div { white-space: nowrap; }
    .ticker b { font-weight: 600; margin-right: .3rem; }
    .ticker .up { color: #15803d; } .ticker .down { color: #b91c1c; }
    @media (prefers-color-scheme: dark) { .ticker .up { color: #4ade80; } .ticker .down { color: #f87171; } }
    </style>"""
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@st.cache_resource
def _refresh_lock() -> threading.Lock:
    return threading.Lock()


def ensure_fresh() -> float:
    """Refresh the feeds if the last attempt is older than REFRESH_MINUTES.

    Returns the time of the last refresh. A lock shared by every session
    stops two visitors from fetching at once.
    """
    conn = store.connect()
    last = store.last_refresh(conn) or 0.0
    if time.time() - last < REFRESH_MINUTES * 60:
        return last
    with _refresh_lock():
        last = store.last_refresh(conn) or 0.0  # another session may have just done it
        if time.time() - last >= REFRESH_MINUTES * 60:
            with st.spinner("Fetching the latest headlines..."):
                refresh.run(conn)
            last = store.last_refresh(conn) or time.time()
    return last


@st.cache_resource(max_entries=2)
def load_corpus(refreshed_at: float) -> Corpus:
    """One corpus per refresh; `refreshed_at` is only the cache key."""
    return Corpus.load()


@st.cache_resource
def briefing_cache() -> dict[float, str]:
    """Briefing text per refresh, shared by every visitor."""
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def watchlist_quotes() -> list[Quote]:
    return get_quotes(list(WATCHLIST))


def current_corpus() -> tuple[Corpus, float]:
    """The cached corpus, with ages measured from now rather than load time."""
    refreshed_at = ensure_fresh()
    return dataclasses.replace(load_corpus(refreshed_at), now=time.time()), refreshed_at


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

_TAG_RE = re.compile(r"\[(\d+|M)\]")
_BULLET = "\u2022 "  # the model sometimes writes inline bullet characters


def tidy(text: str) -> str:
    """Normalise bullets and escape $ (Streamlit reads $...$ as LaTeX)."""
    return text.replace(_BULLET, "\n- ").replace("$", r"\$").strip()


def link_citations(text: str, prepared: Prepared) -> str:
    """Turn [3] into a link to story 3, and [M] into a link to the quote page."""

    def repl(m: re.Match) -> str:
        tag = m.group(1)
        if tag == "M":
            if len(prepared.quotes) == 1:
                return f"[[M]](https://finance.yahoo.com/quote/{prepared.quotes[0].symbol})"
            return "[M]"
        n = int(tag)
        if 1 <= n <= len(prepared.hits):
            return f"[[{n}]]({prepared.hits[n - 1].story.lead.url})"
        return m.group(0)

    return _TAG_RE.sub(repl, tidy(text))


def _price(q: Quote) -> str:
    return f"{q.price:,.3f}%" if q.symbol == "^TNX" else f"{q.price:,.2f}"


def ticker_strip(quotes: list[Quote]) -> None:
    cells = []
    for q in quotes:
        cls = "up" if q.change_pct >= 0 else "down"
        cells.append(
            f'<div title="{q.symbol}, last trade {ago(time.time() - q.as_of)}"><b>{html.escape(q.name)}</b>'
            f'{_price(q)} <span class="{cls}">{q.change_pct:+.2f}%</span></div>'
        )
    st.html(f'<div class="ticker">{"".join(cells)}</div>')


def quote_cards(quotes: list[Quote]) -> None:
    cols = st.columns(min(len(quotes), 4))
    for col, q in zip(cols, quotes[:4]):
        col.metric(q.name, _price(q), f"{q.change_pct:+.2f}%", help=f"{q.symbol}, last trade {ago(time.time() - q.as_of)}")


def sources_block(answer: Answer, now: float) -> None:
    cited = [(n, answer.hits[n - 1]) for n in answer.cited]
    others = [(n, h) for n, h in enumerate(answer.hits, 1) if n not in answer.cited]
    if not answer.hits and not answer.quotes:
        return
    if answer.hits:
        label = f"Sources: {len(cited)} cited" + (f", {len(others)} more retrieved" if others else "")
    else:
        label = "Sources: market data only"
    with st.expander(label):
        for title, group in (("Cited", cited), ("Retrieved but not used", others)):
            if not group:
                continue
            st.markdown(f"**{title}**")
            for n, hit in group:
                story = hit.story
                links = " | ".join(f"[{i.outlet}]({i.url})" for i in story.items[:4])
                st.markdown(
                    f"[{n}] {tidy(story.lead.title)}  \n"
                    f"<small>{links} &middot; first reported {ago(now - story.first_seen)}</small>",
                    unsafe_allow_html=True,
                )
        if answer.quotes:
            st.markdown("**Market data**")
            st.markdown(", ".join(f"[{q.name}](https://finance.yahoo.com/quote/{q.symbol})" for q in answer.quotes))
        if answer.invalid:
            st.warning(f"The answer cites {', '.join(answer.invalid)}, which it was not given. Treat that part with care.")
    if answer.unsupported:
        st.warning(f"Not found in any source: {', '.join(answer.unsupported)}. Check before relying on it.")


def run_llm(prepared: Prepared) -> Answer:
    """Stream the model's answer into the page, then re-render it with links."""
    box = st.empty()
    text = ""
    for chunk in stream(prepared):
        text += chunk
        box.markdown(tidy(text) + " ...")
    answer = finish(prepared, text)  # may trim an unsupported tail; show the checked version
    box.markdown(link_citations(answer.text, prepared))
    return answer


def header(corpus: Corpus, refreshed_at: float) -> None:
    n_items = sum(len(s.items) for s in corpus.stories)
    n_outlets = len({o for s in corpus.stories for o in s.outlets})
    st.caption(
        f"{len(corpus.stories)} stories ({n_items} articles) from {n_outlets} outlets in the last "
        f"{SEARCH_WINDOW_HOURS} hours, updated {ago(time.time() - refreshed_at)} "
        f"&middot; {model_label()} &middot; Not investment advice"
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def briefing_card(corpus: Corpus, refreshed_at: float, quotes: list[Quote]) -> None:
    """The 24-hour briefing, written once per refresh and shared by every visitor."""
    with st.container(border=True):
        st.markdown(f"#### The last {BRIEFING_WINDOW_HOURS} hours")
        briefings = briefing_cache()
        if refreshed_at in briefings:
            st.markdown(briefings[refreshed_at])
        else:
            try:
                prepared = prepare_briefing(corpus, quotes)
                answer = run_llm(prepared)
                briefings.clear()  # keep only the briefing for the current refresh
                briefings[refreshed_at] = link_citations(answer.text, prepared)
            except ProviderUnavailableError as exc:
                st.error(str(exc))
        st.caption(
            "Stories are ranked by how many outlets covered them and how recent they are; "
            "the model writes up the top of that list."
        )


def briefing_page() -> None:
    corpus, refreshed_at = current_corpus()
    st.title("Market Briefing")
    header(corpus, refreshed_at)
    quotes = watchlist_quotes()
    if quotes:
        ticker_strip(quotes)

    briefing_card(corpus, refreshed_at, quotes)

    messages = st.session_state.setdefault("messages", [])
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg.get("notice"):
                st.info(msg["notice"])
            st.markdown(msg["content"])
            if msg.get("answer"):
                sources_block(msg["answer"], corpus.now)

    if not messages:
        st.caption(
            "Ask a follow-up below, for example: *Why did Intel stock fall?* &middot; "
            "*What did the ECB decide?* &middot; *How did the S&P 500 do and why?*"
        )

    question = st.chat_input(f"Ask about anything in the last {SEARCH_WINDOW_HOURS} hours of market news...")
    if not question:
        return
    history = [(u["content"], a["raw"]) for u, a in zip(messages[::2], messages[1::2])]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the news..."):
                prepared = prepare(corpus, question, history=history)
            if prepared.advice:
                st.info(ADVICE_NOTICE)
            if prepared.quotes:
                quote_cards(prepared.quotes)
            answer = run_llm(prepared)
        except ProviderUnavailableError as exc:
            messages.pop()  # keep user/assistant turns paired
            st.error(str(exc))
            return
        sources_block(answer, corpus.now)
    messages.append(
        {
            "role": "assistant",
            "content": link_citations(answer.text, prepared),
            "raw": answer.text,
            "answer": answer,
            "notice": ADVICE_NOTICE if answer.advice else "",
        }
    )


def stories_page() -> None:
    corpus, refreshed_at = current_corpus()
    st.title("Top stories")
    header(corpus, refreshed_at)
    outlets = sorted({o for s in corpus.stories for o in s.outlets})
    picked = st.selectbox("Outlet", ["All outlets", *outlets])
    for story in top_stories(corpus, hours=SEARCH_WINDOW_HOURS, n=60):
        if picked != "All outlets" and picked not in story.outlets:
            continue
        lead = story.lead
        st.markdown(
            f"**[{tidy(lead.title)}]({lead.url})**  \n"
            f"<small>{', '.join(story.outlets)} &middot; {len(story.items)} article(s) &middot; "
            f"first {ago(corpus.now - story.first_seen)}</small>",
            unsafe_allow_html=True,
        )


def sources_page() -> None:
    st.title("Sources")
    conn = store.connect()
    rows = [
        {
            "Outlet": f["outlet"],
            "Feed": f["feed"],
            "Items last fetch": f["last_count"],
            "Last success": ago(time.time() - f["last_ok"]) if f["last_ok"] else "never",
            "Error": f["last_error"],
        }
        for f in store.feed_status(conn)
    ]
    st.dataframe(rows, hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 2)
    st.caption(
        f"Feeds are re-read when the newest fetch is over {REFRESH_MINUTES} minutes old. "
        "Only headlines, each feed's own summary and the link are stored; every answer links back to the publisher. "
        "Prices come from Yahoo Finance and may be delayed."
    )
    if st.button("Refresh now"):
        with st.spinner("Fetching the latest headlines..."):
            refresh.run(conn)
        st.rerun()


st.navigation(
    [
        st.Page(briefing_page, title="Briefing", default=True),
        st.Page(stories_page, title="Top stories", url_path="stories"),
        st.Page(sources_page, title="Sources", url_path="sources"),
    ],
    position="top",
).run()
