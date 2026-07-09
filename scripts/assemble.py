#!/usr/bin/env python3
"""Assemble data/events.jsonl from the per-batch files + controls.

Steps:
  1. read every data/_batch_*.jsonl and data/controls.jsonl
  2. sanitize HTML entities (&amp; -> &, &lt; -> <, &gt; -> >)
  3. dedupe by id, and warn on the same (subject, category) appearing twice
  4. sort by month then id and write data/events.jsonl
  5. print category / month / predictability / region breakdowns

Run:  .venv/bin/python scripts/assemble.py
Then: .venv/bin/kc validate
"""

from __future__ import annotations

import glob
import html
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "events.jsonl")


def _clean(obj: dict) -> dict:
    def fix(v):
        if isinstance(v, str):
            return html.unescape(v)
        if isinstance(v, list):
            return [fix(x) for x in v]
        return v
    return {k: fix(v) for k, v in obj.items()}


def main() -> None:
    sources = sorted(glob.glob(os.path.join(DATA, "_batch_*.jsonl")))
    controls = os.path.join(DATA, "controls.jsonl")
    if os.path.exists(controls):
        sources.append(controls)
    if not sources:
        raise SystemExit("no _batch_*.jsonl or controls.jsonl in data/")

    by_id: dict[str, dict] = {}
    subj_seen: dict[tuple[str, str], str] = {}
    dupe_ids = 0
    for path in sources:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = _clean(json.loads(line))
                oid = obj["id"]
                if oid in by_id:
                    dupe_ids += 1
                    continue
                key = (obj.get("subject", "").lower(), obj.get("category", ""))
                if key in subj_seen and obj.get("category") in {"death", "control_alive", "fake_event"}:
                    print(f"  warn: {obj['subject']} ({obj['category']}) also in "
                          f"{subj_seen[key]} -- possible duplicate person")
                subj_seen[key] = oid
                by_id[oid] = obj

    events = sorted(by_id.values(), key=lambda e: (e["month"], e["id"]))
    with open(OUT, "w", encoding="utf-8") as out:
        for e in events:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")

    cats = Counter(e["category"] for e in events)
    months = Counter(e["month"] for e in events)
    pred = Counter(e["predictability"] for e in events)
    region = Counter(e["region"] for e in events)
    per_month_real = defaultdict(int)
    for e in events:
        if e["category"] in {"death", "office_change"}:
            per_month_real[e["month"]] += 1

    print(f"\nwrote {len(events)} events to {os.path.relpath(OUT, HERE)}")
    if dupe_ids:
        print(f"  (dropped {dupe_ids} duplicate ids across batches)")
    print(f"  sources: {[os.path.basename(s) for s in sources]}")
    print(f"  categories:     {dict(cats)}")
    print(f"  predictability: {dict(pred)}")
    print(f"  region:         {dict(region)}")
    print(f"  months: {len(months)} ({min(months)} .. {max(months)})")
    thin = {m: per_month_real[m] for m in sorted(months) if per_month_real[m] < 8}
    if thin:
        print(f"  months with <8 real events: {thin}")


if __name__ == "__main__":
    main()
