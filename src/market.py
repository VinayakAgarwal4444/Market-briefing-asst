"""Live quotes from Yahoo Finance, and spotting which instruments a question is about.

Prices are always looked up, never generated: the LLM only sees them as a
labelled MARKET DATA block and is told not to state any other figures.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import yfinance as yf

SP500, NASDAQ, DOW, RUSSELL = "^GSPC", "^IXIC", "^DJI", "^RUT"
INDEX_SYMBOLS = frozenset((SP500, NASDAQ, DOW, RUSSELL))

# The strip shown at the top of the app.
WATCHLIST: dict[str, str] = {
    SP500: "S&P 500",
    NASDAQ: "Nasdaq",
    DOW: "Dow Jones",
    "^VIX": "VIX",
    "^TNX": "US 10Y yield",
    "CL=F": "WTI crude",
    "GC=F": "Gold",
    "EURUSD=X": "EUR/USD",
    "BTC-USD": "Bitcoin",
}

# Names people actually type -> Yahoo symbol. Matched case-insensitively on
# word boundaries. Doubles as query expansion for keyword search, so asking
# about "NVDA" also finds headlines that only say "Nvidia".
ALIASES: dict[str, str] = {
    "s&p 500": SP500, "s&p": SP500, "spx": SP500, "nasdaq": NASDAQ,
    "dow jones": DOW, "the dow": DOW, "russell 2000": RUSSELL, "vix": "^VIX",
    "volatility index": "^VIX", "10-year": "^TNX", "10 year yield": "^TNX",
    "treasury yield": "^TNX", "ftse 100": "^FTSE", "ftse": "^FTSE", "dax": "^GDAXI",
    "nikkei": "^N225", "hang seng": "^HSI", "stoxx 600": "^STOXX",
    "oil": "CL=F", "crude": "CL=F", "wti": "CL=F", "brent": "BZ=F", "gold": "GC=F",
    "silver": "SI=F", "copper": "HG=F", "natural gas": "NG=F",
    "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "dollar index": "DX-Y.NYB",
    "euro": "EURUSD=X", "pound": "GBPUSD=X", "sterling": "GBPUSD=X", "yen": "JPY=X",
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "meta": "META", "facebook": "META",
    "tesla": "TSLA", "broadcom": "AVGO", "berkshire": "BRK-B", "jpmorgan": "JPM",
    "jp morgan": "JPM", "visa": "V", "mastercard": "MA", "walmart": "WMT",
    "exxon": "XOM", "chevron": "CVX", "unitedhealth": "UNH", "eli lilly": "LLY",
    "lilly": "LLY", "novo nordisk": "NVO", "johnson & johnson": "JNJ",
    "procter & gamble": "PG", "home depot": "HD", "costco": "COST", "netflix": "NFLX",
    "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE", "amd": "AMD",
    "intel": "INTC", "qualcomm": "QCOM", "micron": "MU", "arm holdings": "ARM",
    "tsmc": "TSM", "taiwan semiconductor": "TSM", "asml": "ASML", "palantir": "PLTR",
    "ibm": "IBM", "cisco": "CSCO", "uber": "UBER", "airbnb": "ABNB", "paypal": "PYPL",
    "coinbase": "COIN", "robinhood": "HOOD", "goldman sachs": "GS", "goldman": "GS",
    "morgan stanley": "MS", "bank of america": "BAC", "citigroup": "C", "citi": "C",
    "wells fargo": "WFC", "blackrock": "BLK", "boeing": "BA", "caterpillar": "CAT",
    "disney": "DIS", "nike": "NKE", "starbucks": "SBUX", "mcdonald's": "MCD",
    "coca-cola": "KO", "pepsico": "PEP", "pfizer": "PFE", "moderna": "MRNA",
    "ford": "F", "general motors": "GM", "rivian": "RIVN", "okta": "OKTA",
    "kenvue": "KVUE", "shell": "SHEL", "bp": "BP", "hsbc": "HSBC", "barclays": "BCS",
    "alibaba": "BABA", "sony": "SONY", "toyota": "TM", "samsung": "005930.KS",
}

_ALIAS_RE = re.compile(
    r"(?<![\w&])(" + "|".join(re.escape(a) for a in sorted(ALIASES, key=len, reverse=True)) + r")(?![\w&])",
    re.I,
)
_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b")
_UPPER_RE = re.compile(r"\b([A-Z]{2,5})\b")
# Upper-case words that are almost never what the user means as a ticker.
_NOT_TICKERS = frozenset(
    """AI CEO CFO CTO US USA UK EU ECB FED FOMC GDP CPI PPI PCE IPO ETF EPS
    SEC IMF OPEC FX USD EUR GBP JPY YOY QOQ AND OR THE WHAT WHY HOW WHO NYSE
    API LLM RAG ESG M&A PE EV IT OK TV""".split()
)


@dataclass
class Quote:
    symbol: str
    name: str
    price: float
    previous_close: float
    change_pct: float
    as_of: float  # unix seconds of the last trade
    currency: str
    market_state: str  # REGULAR, CLOSED, PRE, POST, ...

    @property
    def change(self) -> float:
        return self.price - self.previous_close


def detect_symbols(question: str, limit: int = 5) -> list[str]:
    """Symbols the question refers to, in order of first mention."""
    found: list[str] = []

    def add(sym: str) -> None:
        if sym not in found:
            found.append(sym)

    for m in _CASHTAG_RE.finditer(question):
        add(m.group(1).upper())
    for m in _ALIAS_RE.finditer(question):
        add(ALIASES[m.group(1).lower()])
    for m in _UPPER_RE.finditer(question):
        # Unknown symbols are kept too: get_quotes() drops any Yahoo doesn't know.
        if m.group(1) not in _NOT_TICKERS:
            add(m.group(1))
    return found[:limit]


def symbol_terms(symbols: list[str]) -> list[str]:
    """Every alias for the given symbols (used to widen keyword search).

    Index questions also search for the words market wraps actually use:
    headlines say "Wall Street sells off", rarely "S&P 500 falls".
    """
    terms = [alias for alias, sym in ALIASES.items() if sym in symbols]
    if INDEX_SYMBOLS.intersection(symbols):
        terms += ["wall street", "stocks", "stock market", "selloff", "rally"]
    return terms


def _fetch_quote(symbol: str) -> Quote | None:
    try:
        info = yf.Ticker(symbol).info
    except Exception:  # yfinance raises a zoo of errors for unknown symbols
        return None
    price = info.get("regularMarketPrice")
    prev = info.get("regularMarketPreviousClose")
    if price is None or not prev:
        return None
    return Quote(
        symbol=symbol,
        name=WATCHLIST.get(symbol) or info.get("shortName") or symbol,
        price=float(price),
        previous_close=float(prev),
        change_pct=float(info.get("regularMarketChangePercent") or (price / prev - 1) * 100),
        as_of=float(info.get("regularMarketTime") or time.time()),
        currency=info.get("currency") or "",
        market_state=info.get("marketState") or "",
    )


def get_quotes(symbols: list[str]) -> list[Quote]:
    """Quotes for the symbols Yahoo recognises; unknown ones are skipped."""
    if not symbols:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
        return [q for q in pool.map(_fetch_quote, symbols) if q]


def format_quote(q: Quote) -> str:
    """One line for the LLM's MARKET DATA block."""
    when = time.strftime("%a %d %b %H:%M UTC", time.gmtime(q.as_of))
    unit = "%" if q.symbol == "^TNX" else f" {q.currency}".rstrip()
    return (
        f"{q.name} ({q.symbol}): {q.price:,.2f}{unit}, {q.change_pct:+.2f}% vs previous close "
        f"{q.previous_close:,.2f} (last trade {when}, market {q.market_state.lower() or 'unknown'})"
    )
