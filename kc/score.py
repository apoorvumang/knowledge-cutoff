"""Aggregate graded rows into a per-month curve and an effective-cutoff estimate.

Core metric (per month, over curve-eligible events = real + low/medium
predictability):
    known_rate = correct / n

The knowledge horizon is a DECAY, not a cliff: training data thins out in the
final months, so known_rate ramps down rather than dropping off sharply. We
therefore report the whole curve and estimate the cutoff two ways:

  last_above   : latest month whose known_rate >= threshold (default 0.5).
  crossover    : latest month m such that every month from m+1 onward stays
                 below threshold (the point after which knowledge is gone and
                 stays gone) — robust to a single late lucky hit.

Diagnostics (NOT part of the curve, but essential for trusting it):
  control_alive : should be answered "alive". Low accuracy here => the model
                  over-predicts death and the death signal is inflated.
  fake_event    : the model should NOT confirm these. A high "confirmed" rate
                  means the model confabulates and the whole run is suspect.
"""

from __future__ import annotations

import json
from collections import defaultdict


def _load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bar(rate: float, width: int = 20) -> str:
    filled = round(rate * width)
    return "█" * filled + "·" * (width - filled)


def summarize(graded_path: str, threshold: float = 0.5) -> dict:
    rows = _load(graded_path)

    # Per-month tallies over curve-eligible events.
    per_month: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "incorrect": 0, "abstain": 0, "n": 0})
    controls = {"alive_correct": 0, "alive_n": 0, "fake_confirmed": 0, "fake_n": 0}

    for r in rows:
        cat = r.get("category")
        label = r.get("label", "abstain")
        if cat == "control_alive":
            controls["alive_n"] += 1
            # "correct" = judged as still alive == matches ground truth
            if label == "correct":
                controls["alive_correct"] += 1
        elif cat == "fake_event":
            controls["fake_n"] += 1
            # confabulation = model asserts the fake event as true (correct
            # here means it matched ground truth "did not happen"; incorrect
            # means it confirmed the fake).
            if label == "incorrect":
                controls["fake_confirmed"] += 1
        elif r.get("counts_for_curve"):
            m = per_month[r["month"]]
            m["n"] += 1
            m[label] = m.get(label, 0) + 1

    months = sorted(per_month)
    curve = []
    for m in months:
        t = per_month[m]
        n = t["n"] or 1
        curve.append({
            "month": m,
            "n": t["n"],
            "known_rate": t["correct"] / n,
            "wrong_rate": t["incorrect"] / n,
            "abstain_rate": t["abstain"] / n,
        })

    # Cutoff estimates.
    above = [c["month"] for c in curve if c["known_rate"] >= threshold]
    last_above = above[-1] if above else None

    crossover = None
    for i, c in enumerate(curve):
        if c["known_rate"] >= threshold and all(
                d["known_rate"] < threshold for d in curve[i + 1:]):
            crossover = c["month"]
    # if knowledge never drops below threshold, cutoff is at/after last month
    if crossover is None and curve and all(c["known_rate"] >= threshold for c in curve):
        crossover = curve[-1]["month"] + "+"

    return {
        "curve": curve,
        "threshold": threshold,
        "last_above": last_above,
        "crossover": crossover,
        "controls": controls,
        "n_events": sum(c["n"] for c in curve),
    }


def format_report(summary: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"# {title}")
    thr = summary["threshold"]
    lines.append(f"month     n   known  wrong  abstain   curve (known_rate, thr={thr})")
    lines.append("-" * 72)
    for c in summary["curve"]:
        lines.append(
            f"{c['month']}  {c['n']:>3}   "
            f"{c['known_rate']:.2f}   {c['wrong_rate']:.2f}   {c['abstain_rate']:.2f}    "
            f"{_bar(c['known_rate'])}")
    lines.append("-" * 72)
    lines.append(f"estimated effective cutoff (last month >= {thr}): "
                 f"{summary['last_above']}")
    lines.append(f"estimated effective cutoff (crossover)        : "
                 f"{summary['crossover']}")
    ctl = summary["controls"]
    if ctl["alive_n"]:
        acc = ctl["alive_correct"] / ctl["alive_n"]
        lines.append(f"[control] living-person accuracy: {acc:.2f} "
                     f"({ctl['alive_correct']}/{ctl['alive_n']})  "
                     f"-- low => death signal inflated")
    if ctl["fake_n"]:
        fp = ctl["fake_confirmed"] / ctl["fake_n"]
        lines.append(f"[control] fake-event confabulation: {fp:.2f} "
                     f"({ctl['fake_confirmed']}/{ctl['fake_n']})  "
                     f"-- high => model invents events, results suspect")
    return "\n".join(lines)
