#!/usr/bin/env python3
"""Assemble a HuggingFace dataset bundle in hf_dataset/.

Produces two config files + copies the card:
  events.jsonl   — the 330 benchmark items (one row per event)
  results.jsonl  — long-form eval outputs: one row per (model, probe, event)
                   with the 3-way label and the raw model response

The dataset card (README.md) is maintained separately and copied in.
Run:  .venv/bin/python scripts/build_hf.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_viz import MODEL_META  # noqa: E402  (the published-model allowlist)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# graded/ is a local working directory and may hold runs that must NOT be published:
# internal endpoints, unreleased checkpoints, private finetunes. Publish only models
# explicitly listed in MODEL_META, so appearing in the dataset is opt-in rather than a
# side effect of having been evaluated.
PUBLISHABLE = {m for m, _, _ in MODEL_META}
OUT = os.path.join(HERE, "hf_dataset")
LABEL = {"c": "correct", "w": "incorrect", "a": "abstain"}  # from graded codes
FULL = {"correct": "correct", "incorrect": "incorrect", "abstain": "abstain"}
# Rows the pipeline could not grade (API error, truncated completion, judge
# failure) are NOT results and are dropped from the published dataset rather than
# coerced to "abstain" -- a consumer computing accuracy would otherwise read an
# outage as the model declining to answer.
UNGRADED = "ungraded"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # events.jsonl — straight copy of the curated benchmark
    shutil.copyfile(os.path.join(HERE, "data", "events.jsonl"),
                    os.path.join(OUT, "events.jsonl"))
    n_events = sum(1 for _ in open(os.path.join(OUT, "events.jsonl")))

    # results.jsonl — flatten graded/*.jsonl
    rows = []
    n_ungraded = 0
    withheld = set()
    for g in sorted(glob.glob(os.path.join(HERE, "graded", "*.jsonl"))):
        base = os.path.basename(g)[:-6]
        model, probe = base.rsplit("__", 1)
        if model not in PUBLISHABLE:
            withheld.add(model)
            continue
        with open(g, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                label = r.get("label", "abstain")
                if label == UNGRADED:
                    n_ungraded += 1
                    continue
                if label not in FULL:
                    raise RuntimeError(
                        f"{base}: unexpected label {label!r} -- refusing to publish "
                        f"a dataset with labels this script cannot represent")
                rows.append({
                    "model": model,
                    "probe": probe,
                    "event_id": r["event_id"],
                    "month": r["month"],
                    "category": r["category"],
                    "predictability": r["predictability"],
                    "label": FULL[label],
                    "response": r.get("response", ""),
                })
    with open(os.path.join(OUT, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # card
    card = os.path.join(HERE, "hf_dataset_card.md")
    if os.path.exists(card):
        shutil.copyfile(card, os.path.join(OUT, "README.md"))

    models = sorted({r["model"] for r in rows})
    print(f"hf_dataset/ ready: {n_events} events, {len(rows)} result rows, "
          f"{len(models)} models")
    if n_ungraded:
        print(f"  dropped {n_ungraded} ungraded row(s) (not results — re-grade to include)")
    if withheld:
        print(f"  WITHHELD (not in MODEL_META, so not published): {', '.join(sorted(withheld))}")
    print("  models:", ", ".join(models))


if __name__ == "__main__":
    main()
