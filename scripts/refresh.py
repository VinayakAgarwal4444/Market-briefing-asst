"""Fetch the feeds once (for cron / Task Scheduler, or to warm the store)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import refresh  # noqa: E402


def main() -> int:
    stats = refresh.run()
    print(
        f"[refresh] {stats.fetched} items fetched, {stats.new} new, {stats.embedded} embedded, "
        f"{stats.pruned} pruned in {stats.seconds:.1f}s"
    )
    for failure in stats.failed_feeds:
        print(f"[refresh] FAILED {failure}", file=sys.stderr)
    return 1 if stats.failed_feeds else 0


if __name__ == "__main__":
    raise SystemExit(main())
