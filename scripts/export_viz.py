#!/usr/bin/env python3
"""Export a single JSON blob powering the interactive results page.

Reads data/events.jsonl + every graded/*.jsonl and emits report_data.json:

  {
    "months":  [...],
    "models":  ["claude-fable-5", ...],          # display order
    "events":  [ {id, month, category, predictability, region, subject,
                  q, expected, source, mcq_q, mcq_choices, mcq_answer}, ... ],
    "summary": { "<model>": { "<probe>": {curve, cutoff, controls} } },
    "answers": { "<model>": { "<probe>": { "<event_id>": {l, r} } } }
  }

where l = label code (c=correct, w=wrong, a=abstain) and r = response text
(capped). Response is dropped when it equals the empty string.

Run:  .venv/bin/python scripts/export_viz.py
"""

from __future__ import annotations

import glob
import json
import os

from kc.schema import load_events
from kc.score import summarize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "report_data.json")
CAP = 700
LCODE = {"correct": "c", "incorrect": "w", "abstain": "a"}

# Display metadata: order + advertised cutoffs.
# Advertised cutoffs sourced from official provider docs (July 2026):
#   Claude Opus 4.8 / Sonnet 5 / Fable 5 -> Jan 2026 (Claude Platform docs)
#   GPT-5.6 (sol) -> Feb 2026 (OpenAI model page)
#   GPT-4o -> Oct 2023 (OpenAI model page)
#   Gemini 3.5 Flash / 3.1 Pro -> Jan 2025 (Gemini API docs + DeepMind model card)
#   Grok 4.5 -> no official cutoff published by xAI (the "Dec 2025" figure is Grok 4.3)
#   GLM-5.2, DeepSeek-V4-Pro -> no official cutoff published (secondary figures unattributed)
MODEL_META = [
    ("claude-fable-5", "Claude Fable 5", "Jan 2026"),
    ("gpt-5.6-sol", "GPT-5.6 (sol)", "Feb 2026"),
    ("grok-4.5", "Grok 4.5", "not published"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash", "Jan 2025"),
    ("gemini-3.1-pro", "Gemini 3.1 Pro", "Jan 2025"),
    ("claude-opus-4-8", "Claude Opus 4.8", "Jan 2026"),
    ("claude-sonnet-5", "Claude Sonnet 5", "Jan 2026"),
    ("glm-5.2", "GLM-5.2", "not published"),
    ("deepseek-v4-pro", "DeepSeek-V4-Pro", "not published"),
    ("gpt-4o", "GPT-4o", "Oct 2023"),
]


def main() -> None:
    events = load_events(os.path.join(HERE, "data", "events.jsonl"))
    ev_out = [{
        "id": e.id, "month": e.month, "category": e.category,
        "predictability": e.predictability, "region": e.region,
        "subject": e.subject, "q": e.question_direct, "expected": e.expected_direct,
        "fact": e.fact, "source": e.source,
        "mcq_q": e.mcq_question, "mcq_choices": e.mcq_choices, "mcq_answer": e.mcq_answer,
    } for e in events]
    months = sorted({e.month for e in events})

    graded = glob.glob(os.path.join(HERE, "graded", "*.jsonl"))
    present = set()
    for g in graded:
        base = os.path.basename(g)[:-6]
        model, probe = base.rsplit("__", 1)
        present.add(model)

    models = [m for m, _, _ in MODEL_META if m in present]
    labels = {m: {"name": n, "advertised": a} for m, n, a in MODEL_META}

    answers: dict = {}
    summary: dict = {}
    for g in sorted(graded):
        base = os.path.basename(g)[:-6]
        model, probe = base.rsplit("__", 1)
        if model not in models:
            continue
        s = summarize(g)
        summary.setdefault(model, {})[probe] = {
            "curve": [{"m": c["month"], "k": round(c["known_rate"], 3),
                       "w": round(c["wrong_rate"], 3), "a": round(c["abstain_rate"], 3),
                       "n": c["n"]} for c in s["curve"]],
            "cutoff": s["crossover"],
            "controls": s["controls"],
        }
        amap: dict = {}
        with open(g, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entry = {"l": LCODE.get(row.get("label", "abstain"), "a")}
                resp = (row.get("response") or "").strip()
                if resp:
                    entry["r"] = resp[:CAP] + ("…" if len(resp) > CAP else "")
                amap[row["event_id"]] = entry
        answers.setdefault(model, {})[probe] = amap

    blob = {
        "months": months, "models": models, "labels": labels,
        "events": ev_out, "summary": summary, "answers": answers,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, separators=(",", ":"))
    kb = os.path.getsize(OUT) / 1024
    print(f"wrote {os.path.relpath(OUT, HERE)}  ({kb:.0f} KB)")
    print(f"  models: {models}")
    print(f"  probes per model: " +
          ", ".join(f"{m}={list(summary.get(m, {}))}" for m in models))


if __name__ == "__main__":
    main()
