"""End-to-end pipeline: scrape -> ingest -> persist FAISS index."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make project root importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ingest, scraper  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the RAG vector index from scratch."
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Reuse already-processed docs and only rebuild the FAISS index.",
    )
    args = parser.parse_args()

    started = time.time()

    if not args.skip_scrape:
        print("=== Phase 1: Scraping curated docs ===")
        scraper.run()
    else:
        print("=== Phase 1: Skipped (using existing processed docs) ===")

    print("\n=== Phase 2: Building FAISS index ===")
    n_chunks = ingest.run()

    elapsed = time.time() - started
    print(
        f"\n[build] Done. {n_chunks} chunks indexed in {elapsed:.1f}s. "
        "You can now run: streamlit run app.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
