"""Ask the market briefing assistant one question from the command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chain import ADVICE_NOTICE, Corpus, ago, ask  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask the market briefing assistant.")
    parser.add_argument("question", nargs="+", help="Question text")
    parser.add_argument("--no-quotes", action="store_true", help="Skip the live price lookup")
    args = parser.parse_args()

    corpus = Corpus.load()
    answer = ask(corpus, " ".join(args.question), with_quotes=not args.no_quotes)

    print("\n=== Answer ===")
    if answer.advice:
        print(f"({ADVICE_NOTICE})")
    print(answer.text)
    if answer.unsupported:
        print(f"\n(!) Figures not found in any source: {', '.join(answer.unsupported)}")
    if answer.invalid:
        print(f"\n(!) Cited sources it was not given: {', '.join(answer.invalid)}")

    print("\n=== Sources ===")
    for n, hit in enumerate(answer.hits, 1):
        mark = "*" if n in answer.cited else " "
        lead = hit.story.lead
        print(f"{mark}[{n}] {lead.title} -- {', '.join(hit.story.outlets)}, "
              f"{ago(corpus.now - hit.story.first_seen)} (score {hit.score:.4f}, cos {hit.dense:.2f})")
        print(f"      {lead.url}")
    for q in answer.quotes:
        print(f"{'*' if answer.cited_market else ' '}[M] {q.name} ({q.symbol}) {q.price:,.2f} {q.change_pct:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
