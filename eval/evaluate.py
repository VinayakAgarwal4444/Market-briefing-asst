"""Evaluate the market briefing assistant against a frozen news snapshot.

Three parts, written to eval/results.md:

1. **Retrieval ablation** (no LLM): does the right story reach the model?
   Dense only vs BM25 only vs hybrid vs hybrid + recency (the app).
2. **End-to-end** (the app's configuration): are the facts in the answer,
   do the citations point at the right story, does it say "nothing was
   reported" when that's true, and does it stay out of giving advice?
3. **Closed-book baseline**: the same model with no news at all, to show
   what retrieval buys on questions about the last 48 hours.

Live quotes are switched off: prices move, and the evaluation must be
repeatable. Run with --no-llm for part 1 only (seconds instead of minutes).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from config import EVAL_DIR, RETRIEVAL_K  # noqa: E402
from src.chain import Corpus, finish, prepare, stream  # noqa: E402
from src.llm import get_llm, model_label  # noqa: E402
from src.market import detect_symbols, symbol_terms  # noqa: E402
from src.stories import Story  # noqa: E402

ABLATIONS = [
    ("Dense only", {"mode": "dense", "recency_weight": 0.0}),
    ("BM25 only", {"mode": "bm25", "recency_weight": 0.0}),
    ("Hybrid + recency boost (w = 0.4)", {"mode": "hybrid", "recency_weight": 0.4}),
    ("Hybrid (the app)", {}),
]

# First- or second-person recommendations. Reporting that an analyst
# "says buy" is allowed; the assistant recommending it is not.
_ADVICE_LANGUAGE = re.compile(
    r"\b(you should (buy|sell|hold|consider)|i (would )?(recommend|suggest|advise)|"
    r"i would (buy|sell|hold)|(now is|it is|it's) a good time to (buy|sell)|consider (buying|selling))\b",
    re.I,
)
_ADMITS_NOT_KNOWING = re.compile(
    r"(don't|do not) have (access|real-time|current|information)|knowledge cutoff|as of my|"
    r"not able to provide|cannot provide|can't provide|i'm not aware|no information",
    re.I,
)
_CLOSED_BOOK_PROMPT = "You are a market news assistant. Answer the question about recent financial news."


@dataclass
class Row:
    q: str
    kind: str
    rank: int | None = None  # 1-based rank of the first correct story, None if not retrieved
    n_context: int = 0
    facts: float | None = None
    cited_correct: bool | None = None
    abstained: bool = False
    invalid: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    advice_flagged: bool = False
    advice_language: bool = False
    answer: str = ""
    baseline_facts: float | None = None
    baseline_admits: bool | None = None


def has_source(story: Story, fragments: list[str]) -> bool:
    return any(f in item.url for item in story.items for f in fragments)


def fact_recall(text: str, facts: list[str]) -> float:
    low = text.lower()
    return sum(any(alt in low for alt in fact.split("|")) for fact in facts) / len(facts)


def first_rank(stories: list[Story], fragments: list[str]) -> int | None:
    return next((n for n, s in enumerate(stories, 1) if has_source(s, fragments)), None)


def load(snapshot: str) -> tuple[list[dict], Corpus]:
    spec = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text(encoding="utf-8"))
    name = snapshot or spec["snapshot"]
    db = EVAL_DIR / "snapshots" / f"{name}.db"
    taken_at = json.loads(db.with_suffix(".json").read_text())["taken_at"]
    return spec["questions"], Corpus.load(sqlite3.connect(db), now=taken_at)


def retrieval_ablation(questions: list[dict], corpus: Corpus) -> list[dict]:
    rows = []
    answerable = [q for q in questions if q["kind"] == "answerable"]
    uncovered = [q for q in questions if q["kind"] == "uncovered"]
    for label, kwargs in ABLATIONS:
        ranks, noise = [], []
        for q in answerable + uncovered:
            hits = corpus.index.search(
                q["q"], k=RETRIEVAL_K, extra_terms=symbol_terms(detect_symbols(q["q"])), now=corpus.now, **kwargs
            )
            if q["kind"] == "answerable":
                ranks.append(first_rank([h.story for h in hits], q["sources"]))
            else:
                noise.append(len(hits))
        n = len(ranks)
        rows.append(
            {
                "config": label,
                "hit@1": sum(r == 1 for r in ranks),
                "hit@3": sum(r is not None and r <= 3 for r in ranks),
                "hit@k": sum(r is not None for r in ranks),
                "mrr": sum(1 / r for r in ranks if r) / n,
                "n": n,
                "uncovered_context": sum(noise) / len(noise),
            }
        )
    return rows


def end_to_end(questions: list[dict], corpus: Corpus, baseline: bool) -> list[Row]:
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[eval] {i}/{len(questions)} {q['q']}", flush=True)
        prepared = prepare(corpus, q["q"], with_quotes=False)
        answer = finish(prepared, "".join(stream(prepared)))
        row = Row(
            q=q["q"], kind=q["kind"], n_context=len(answer.hits), abstained=answer.abstained,
            invalid=answer.invalid, advice_flagged=answer.advice,
            unsupported_numbers=answer.unsupported,
            advice_language=bool(_ADVICE_LANGUAGE.search(answer.text)), answer=answer.text,
        )
        if q["kind"] == "answerable":
            row.rank = first_rank([h.story for h in answer.hits], q["sources"])
            row.facts = fact_recall(answer.text, q["facts"])
            row.cited_correct = any(has_source(answer.hits[n - 1].story, q["sources"]) for n in answer.cited)
            if baseline:
                text = get_llm().invoke([SystemMessage(_CLOSED_BOOK_PROMPT), HumanMessage(q["q"])]).content
                row.baseline_facts = fact_recall(text, q["facts"])
                row.baseline_admits = bool(_ADMITS_NOT_KNOWING.search(text))
        rows.append(row)
    return rows


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "n/a"


def report(ablation: list[dict], rows: list[Row], corpus: Corpus, seconds: float, snapshot: str) -> str:
    when = time.strftime("%d %b %Y %H:%M UTC", time.gmtime(corpus.now))
    n_items = sum(len(s.items) for s in corpus.stories)
    out = [
        "# Evaluation",
        "",
        f"Snapshot `{snapshot}`: {len(corpus.stories)} stories ({n_items} articles), news up to {when}. "
        f"Model: {model_label()}, temperature 0. Top-k = {RETRIEVAL_K}. Live quotes off. Ran in {seconds:.0f}s.",
        "",
        "## 1. Retrieval: does the right story reach the model?",
        "",
        "Answerable questions only; *hit@k* counts questions whose correct story is anywhere in the context. "
        "*Uncovered context* is the average number of stories passed to the model for questions the news "
        "does not cover (lower means less temptation to stretch a loose match).",
        "",
        "| Configuration | hit@1 | hit@3 | hit@k | MRR | Uncovered context |",
        "|---|---|---|---|---|---|",
    ]
    for r in ablation:
        out.append(
            f"| {r['config']} | {pct(r['hit@1'], r['n'])} | {pct(r['hit@3'], r['n'])} | {pct(r['hit@k'], r['n'])} "
            f"| {r['mrr']:.2f} | {r['uncovered_context']:.1f} stories |"
        )

    if rows:
        ans = [r for r in rows if r.kind == "answerable"]
        unc = [r for r in rows if r.kind == "uncovered"]
        adv = [r for r in rows if r.kind == "advice"]
        all_invalid = sum(bool(r.invalid) for r in rows)
        out += [
            "",
            "## 2. End to end",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| Answerable: all expected facts in the answer | {pct(sum(r.facts == 1 for r in ans), len(ans))} |",
            f"| Answerable: average fact recall | {100 * sum(r.facts for r in ans) / len(ans):.0f}% |",
            f"| Answerable: cites the story that holds the answer | {pct(sum(bool(r.cited_correct) for r in ans), len(ans))} |",
            f"| Answerable: wrongly says nothing was reported | {pct(sum(r.abstained for r in ans), len(ans))} |",
            f"| Uncovered: correctly says nothing was reported | {pct(sum(r.abstained for r in unc), len(unc))} |",
            f"| Advice: caught and shown the no-advice notice | {pct(sum(r.advice_flagged for r in adv), len(adv))} |",
            f"| Advice: answer free of recommendation language | {pct(sum(not r.advice_language for r in adv), len(adv))} |",
            f"| Any answer citing a source it was not given | {pct(all_invalid, len(rows))} |",
            f"| Any answer stating a number not found in its sources | "
            f"{pct(sum(bool(r.unsupported_numbers) for r in rows), len(rows))} |",
        ]
        if any(r.baseline_facts is not None for r in ans):
            out += [
                "",
                "## 3. Closed-book baseline (same model, no news)",
                "",
                "| | With retrieval | Closed book |",
                "|---|---|---|",
                f"| All expected facts in the answer | {pct(sum(r.facts == 1 for r in ans), len(ans))} "
                f"| {pct(sum(r.baseline_facts == 1 for r in ans), len(ans))} |",
                f"| Average fact recall | {100 * sum(r.facts for r in ans) / len(ans):.0f}% "
                f"| {100 * sum(r.baseline_facts for r in ans) / len(ans):.0f}% |",
                f"| Says it doesn't have current information | - | {pct(sum(bool(r.baseline_admits) for r in ans), len(ans))} |",
            ]
        out += ["", "## Per question", "", "| # | Kind | Question | Rank | Facts | Cited right | Flags | Answer (start) |", "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            facts = f"{100 * r.facts:.0f}%" if r.facts is not None else ("abstained" if r.abstained else "answered")
            cited = {True: "yes", False: "no", None: "-"}[r.cited_correct]
            excerpt = r.answer.replace("\n", " ").replace("|", "/")[:140]
            flags = "; ".join(
                ([f"unsupported numbers: {', '.join(r.unsupported_numbers)}"] if r.unsupported_numbers else [])
                + ([f"invalid citations: {', '.join(r.invalid)}"] if r.invalid else [])
            )
            out.append(f"| {i} | {r.kind} | {r.q} | {r.rank or '-'} | {facts} | {cited} | {flags or '-'} | {excerpt} |")
    return "\n".join(out) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers on a frozen snapshot.")
    parser.add_argument("--snapshot", default="", help="Snapshot name (default: the one in questions.yaml)")
    parser.add_argument("--no-llm", action="store_true", help="Retrieval ablation only")
    parser.add_argument("--no-baseline", action="store_true", help="Skip the closed-book baseline")
    args = parser.parse_args()

    started = time.time()
    questions, corpus = load(args.snapshot)
    ablation = retrieval_ablation(questions, corpus)
    rows = [] if args.no_llm else end_to_end(questions, corpus, baseline=not args.no_baseline)
    snapshot = args.snapshot or yaml.safe_load((EVAL_DIR / "questions.yaml").read_text(encoding="utf-8"))["snapshot"]
    text = report(ablation, rows, corpus, time.time() - started, snapshot)
    # Quoted answers contain bullets and curly quotes; write them as character
    # references (they render the same) so the file stays plain ASCII.
    ascii_text = text.encode("ascii", "xmlcharrefreplace").decode("ascii")
    (EVAL_DIR / "results.md").write_text(ascii_text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
