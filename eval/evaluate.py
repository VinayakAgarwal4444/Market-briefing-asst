"""Evaluate RAG vs baseline (no-context) LLM on domain-specific questions.

For each question we run:
  - baseline: get_llm().invoke(question)              (vanilla LLM, no docs)
  - rag:     chain.invoke({"input": question})        (retrieved context)
We score by case-insensitive keyword recall against expected_keywords, then
write a side-by-side markdown report with an aggregate metric usable on a resume.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EVAL_DIR  # noqa: E402
from src.chain import ask  # noqa: E402
from src.llm import get_llm  # noqa: E402


def _keyword_recall(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords)


def _baseline(question: str) -> str:
    """Vanilla LLM call with no context, for comparison."""
    response = get_llm().invoke(question)
    return getattr(response, "content", str(response))


def _load_questions() -> list[dict]:
    path = EVAL_DIR / "questions.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def _format_answer(text: str, max_chars: int = 1200) -> str:
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def main() -> int:
    questions = _load_questions()
    if not questions:
        print("[eval] no questions found in eval/questions.yaml")
        return 1

    rows: list[dict] = []
    started = time.time()

    for i, item in enumerate(questions, start=1):
        q = item["question"]
        keywords = item.get("expected_keywords", [])
        print(f"[eval] ({i}/{len(questions)}) {q}")

        try:
            baseline_answer = _baseline(q)
        except Exception as exc:  # noqa: BLE001
            baseline_answer = f"<baseline error: {exc}>"

        try:
            rag_result = ask(q)
            rag_answer = rag_result.get("answer", "")
            sources = []
            seen: set[str] = set()
            for doc in rag_result.get("context", []):
                url = doc.metadata.get("source_url", "")
                if url and url not in seen:
                    seen.add(url)
                    sources.append(url)
        except Exception as exc:  # noqa: BLE001
            rag_answer = f"<rag error: {exc}>"
            sources = []

        rows.append(
            {
                "question": q,
                "keywords": keywords,
                "baseline": baseline_answer,
                "baseline_score": _keyword_recall(baseline_answer, keywords),
                "rag": rag_answer,
                "rag_score": _keyword_recall(rag_answer, keywords),
                "sources": sources,
            }
        )

    elapsed = time.time() - started

    base_avg = sum(r["baseline_score"] for r in rows) / len(rows)
    rag_avg = sum(r["rag_score"] for r in rows) / len(rows)
    delta_pct = (rag_avg - base_avg) * 100
    rel_improvement = (
        ((rag_avg - base_avg) / base_avg * 100) if base_avg > 0 else float("inf")
    )

    out_lines: list[str] = []
    out_lines.append("# RAG vs Baseline Evaluation\n")
    out_lines.append(
        f"_{len(rows)} domain-specific questions, "
        f"evaluated in {elapsed:.1f}s._\n"
    )
    out_lines.append("## Summary\n")
    out_lines.append("| Metric | Baseline (no RAG) | RAG |")
    out_lines.append("|---|---|---|")
    out_lines.append(
        f"| Avg keyword recall | **{base_avg * 100:.1f}%** | **{rag_avg * 100:.1f}%** |"
    )
    out_lines.append(f"| Absolute improvement | -- | **+{delta_pct:.1f} pts** |")
    if base_avg > 0:
        out_lines.append(
            f"| Relative improvement | -- | **+{rel_improvement:.0f}%** |"
        )
    out_lines.append("")
    out_lines.append("## Per-question results\n")
    for i, r in enumerate(rows, start=1):
        out_lines.append(f"### Q{i}. {r['question']}\n")
        out_lines.append(
            f"_Expected keywords: {', '.join(r['keywords']) or '(none)'}_  "
        )
        out_lines.append(
            f"_Baseline score: {r['baseline_score'] * 100:.0f}% | "
            f"RAG score: {r['rag_score'] * 100:.0f}%_\n"
        )
        out_lines.append("**Baseline (no context):**\n")
        out_lines.append(f"> {_format_answer(r['baseline']).replace(chr(10), chr(10) + '> ')}\n")
        out_lines.append("**RAG:**\n")
        out_lines.append(f"> {_format_answer(r['rag']).replace(chr(10), chr(10) + '> ')}\n")
        if r["sources"]:
            out_lines.append("**Sources:**")
            for s in r["sources"]:
                out_lines.append(f"- {s}")
        out_lines.append("")

    report_path = EVAL_DIR / "results.md"
    report_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(
        f"\n[eval] Baseline avg recall: {base_avg * 100:.1f}%  "
        f"RAG avg recall: {rag_avg * 100:.1f}%  "
        f"(+{delta_pct:.1f} pts)"
    )
    print(f"[eval] Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
