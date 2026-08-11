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
LCODE = {"correct": "c", "incorrect": "w", "abstain": "a", "ungraded": "u"}
UNGRADED_MAX = 0.05   # skip a probe if >5% of its rows never got a grade

# Display metadata: order + advertised cutoffs.
# Advertised cutoffs sourced from official provider docs (July 2026):
#   Claude Opus 4.8 / Sonnet 5 / Fable 5 -> Jan 2026 (Claude Platform docs)
#   GPT-5.6 (sol) -> Feb 2026 (OpenAI model page)
#   GPT-4o -> Oct 2023 (OpenAI model page)
#   Gemini 3.5 Flash / 3.1 Pro -> Jan 2025 (Gemini API docs + DeepMind model card)
#   Grok 4.5 -> no official cutoff published by xAI (the "Dec 2025" figure is Grok 4.3)
#   GLM-5.2, DeepSeek-V4-Pro -> no official cutoff published (secondary figures unattributed)
#   Muse Glimmer 30B -> Jan 2026, stated verbatim on the official HF model card
#     ("Knowledge cutoff: January 4, 2026"), huggingface.co/meta-models/Muse-Glimmer-30B
#   Muse Spark 1.2 -> no official cutoff published. It is Glimmer's teacher (Glimmer is
#     distilled from it), so Jan 2026 is a tempting inference — deliberately NOT made here:
#     distillation does not pin the teacher's own horizon.
#   DeepSeek V4 Flash 0731 -> no official cutoff published; DeepSeek documents 0731 as a
#     post-training revision of V4 Flash and publishes no data cutoff for it.
#   Qwen3.8 Max -> no official cutoff published; as of Aug 2026 Alibaba has shipped no
#     model card or technical report for it at all.
#   Inkling -> no official cutoff published. Its model card (thinkingmachines.ai/
#     model-card/inkling/) concedes one exists without naming it: "Inkling's knowledge
#     is limited to information available as of its training cutoff."
#   Kimi K3 -> no official cutoff published; absent from the HF model card
#     (huggingface.co/moonshotai/Kimi-K3) and the public technical report.
#   Gemini 3.6 Flash -> Mar 2026, per the DeepMind model card, which hedges it:
#     "The knowledge cutoff date for Gemini 3.6 Flash is March 2026 - users can expect
#     updated information for some domains while in others they may experience the
#     model's knowledge is limited to January 2025". Only the headline figure is
#     recorded here; the caveat is the sort of claim this benchmark exists to test, so
#     it should be read off the measured curve rather than encoded as the advertised
#     value.
MODEL_META = [
    ("claude-fable-5", "Claude Fable 5", "Jan 2026"),
    ("gpt-5.6-sol", "GPT-5.6 (sol)", "Feb 2026"),
    ("gpt-5.5", "GPT-5.5", "Dec 2025"),
    ("gpt-5.4", "GPT-5.4", "Aug 2025"),
    ("grok-4.5", "Grok 4.5", "not published"),
    ("gemini-3.6-flash", "Gemini 3.6 Flash", "Mar 2026"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash", "Jan 2025"),
    ("gemini-3.1-pro", "Gemini 3.1 Pro", "Jan 2025"),
    ("claude-opus-4-8", "Claude Opus 4.8", "Jan 2026"),
    ("claude-sonnet-5", "Claude Sonnet 5", "Jan 2026"),
    ("muse-spark-1.2", "Muse Spark 1.2", "not published"),
    ("muse-glimmer-30b", "Muse Glimmer 30B", "Jan 2026"),
    ("qwen3.8-max", "Qwen3.8 Max", "not published"),
    ("inkling", "Inkling", "not published"),
    ("kimi-k3", "Kimi K3", "not published"),
    ("glm-5.2", "GLM-5.2", "not published"),
    ("deepseek-v4-pro", "DeepSeek-V4-Pro", "not published"),
    ("deepseek-v4-flash-0731", "DeepSeek V4 Flash (0731)", "not published"),
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

    known = [m for m, _, _ in MODEL_META if m in present]
    labels = {m: {"name": n, "advertised": a} for m, n, a in MODEL_META}

    answers: dict = {}
    summary: dict = {}
    for g in sorted(graded):
        base = os.path.basename(g)[:-6]
        model, probe = base.rsplit("__", 1)
        if model not in known:
            continue
        s = summarize(g)
        # Refuse to publish a (model, probe) whose grades are largely missing --
        # e.g. a judge that ran out of API credit turns every row ungraded, which
        # would render as a confident "this model knows nothing" curve. Re-grade
        # from the preserved runs/ file and re-export instead.
        ungraded = s.get("ungraded", 0)
        total = s["n_events"] + ungraded
        if total and ungraded / total > UNGRADED_MAX:
            print(f"  SKIP {model}/{probe}: {ungraded}/{total} rows ungraded "
                  f"(>{UNGRADED_MAX:.0%}) -- re-grade before publishing")
            continue
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

    # Only models with at least one publishable probe, in MODEL_META order.
    models = [m for m in known if summary.get(m)]

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
