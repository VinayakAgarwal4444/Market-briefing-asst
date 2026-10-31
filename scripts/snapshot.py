"""Freeze the current news store as an evaluation snapshot.

Live news changes every few minutes, so the evaluation runs against a
frozen copy: eval/snapshots/<name>.db plus the moment it was taken, which
becomes "now" for every question asked of it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EVAL_DIR  # noqa: E402
from src import store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", default=time.strftime("%Y-%m-%d", time.gmtime()))
    args = parser.parse_args()

    conn = store.connect()
    taken_at = store.last_refresh(conn) or time.time()
    dest = EVAL_DIR / "snapshots" / f"{args.name}.db"
    store.snapshot(conn, dest)
    dest.with_suffix(".json").write_text(json.dumps({"taken_at": taken_at}, indent=2) + "\n")
    print(f"[snapshot] {dest} (now = {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(taken_at))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
