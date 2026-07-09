"""Grade raw runs into three-way labels: correct / incorrect / abstain.

Why three-way (not binary): models are RLHF'd to hedge on recent events
("as of my last update I'm not aware..."). That is neither knowing nor being
wrong — it is abstention, and conflating it with "incorrect" corrupts the
cutoff estimate. So:
  correct   : the model stated the true fact.
  incorrect : the model stated something false / contradicting the fact
              (this is the strongest signal the event post-dates training —
              e.g. confidently describing a dead person as active).
  abstain   : the model declined / said it didn't know / hedged without
              committing to the truth.

MCQ is graded deterministically by extracting the chosen letter (no judge). A
forced choice can't "abstain", so MCQ yields correct/incorrect only (an
unparseable answer -> abstain).

Direct (free-response) is graded by an LLM judge that is GIVEN the ground truth,
so the judge's own knowledge cutoff is irrelevant.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .providers import Model, get_model
from .schema import Event, load_events

LABELS = {"correct", "incorrect", "abstain"}

JUDGE_SYSTEM = (
    "You are a strict grader. You are given a QUESTION, the GROUND TRUTH answer, "
    "and a MODEL ANSWER. Classify the MODEL ANSWER into exactly one label:\n"
    "  correct   - it asserts the ground-truth fact (minor wording/date-precision "
    "differences are fine; the key fact must match).\n"
    "  incorrect - it asserts something that contradicts the ground truth "
    "(e.g. says a person is alive/active when the truth is they died, names the "
    "wrong office-holder, gives a clearly wrong outcome).\n"
    "  abstain   - it declines to answer, says it does not know, says its "
    "information may be out of date, or hedges without committing to either the "
    "true or a false claim.\n\n"
    "Judge ONLY against the provided ground truth. Do not use your own knowledge "
    "of current events. Output STRICT JSON: {\"label\": \"...\", \"reason\": \"...\"}"
)


def extract_mcq_letter(text: str, valid: list[str]) -> str | None:
    """Best-effort extraction of a chosen A-D letter from a free-form answer."""
    if not text:
        return None
    t = text.strip()
    # 1) leading standalone letter, e.g. "B" or "B)" or "B."
    m = re.match(r"\s*\(?([A-D])\)?[\.\):]?\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2) phrases like "the answer is C"
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-D])\)?", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 3) any isolated capital letter A-D in the valid set
    for ch in re.findall(r"\b([A-D])\b", t):
        if ch.upper() in valid:
            return ch.upper()
    return None


def grade_mcq(row: dict, ev: Event) -> dict:
    if row.get("error"):
        return {**_base(row, ev), "label": "abstain", "reason": "api error"}
    letter = extract_mcq_letter(row["response"], ev.mcq_letters())
    if letter is None:
        label, reason = "abstain", "no parseable letter"
    elif letter == ev.mcq_answer:
        label, reason = "correct", f"chose {letter}"
    else:
        label, reason = "incorrect", f"chose {letter}, correct {ev.mcq_answer}"
    return {**_base(row, ev), "label": label, "reason": reason, "chosen": letter}


def _judge_prompt(ev: Event, answer: str) -> str:
    return (
        f"QUESTION:\n{ev.question_direct}\n\n"
        f"GROUND TRUTH:\n{ev.expected_direct}\n"
        f"(context: {ev.fact})\n\n"
        f"MODEL ANSWER:\n{answer}"
    )


def grade_direct(row: dict, ev: Event, judge: Model) -> dict:
    if row.get("error"):
        return {**_base(row, ev), "label": "abstain", "reason": "api error"}
    if not row["response"].strip():
        return {**_base(row, ev), "label": "abstain", "reason": "empty response"}
    try:
        text, _ = judge.complete(_judge_prompt(ev, row["response"]), JUDGE_SYSTEM,
                                 max_tokens=300)
        obj = _parse_json(text)
        label = str(obj.get("label", "")).lower().strip()
        if label not in LABELS:
            label = "abstain"
        reason = str(obj.get("reason", ""))[:300]
    except Exception as e:
        label, reason = "abstain", f"judge error: {e}"
    return {**_base(row, ev), "label": label, "reason": reason}


def _base(row: dict, ev: Event) -> dict:
    return {
        "event_id": ev.id,
        "month": ev.month,
        "category": ev.category,
        "predictability": ev.predictability,
        "region": ev.region,
        "probe": row.get("probe"),
        "counts_for_curve": ev.counts_for_curve,
        "response": row.get("response", ""),
    }


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def grade_run(run_path: str, events_path: str, out_path: str,
              judge_key: str = "claude-opus-4-8", concurrency: int = 8) -> str:
    events = {ev.id: ev for ev in load_events(events_path)}
    rows = []
    with open(run_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows in {run_path}")

    probe = rows[0].get("probe", "direct")
    judge = get_model(judge_key) if probe == "direct" else None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    results: list[dict] = []

    def work(row):
        ev = events.get(row["event_id"])
        if ev is None:
            return None
        if row.get("probe") == "mcq":
            return grade_mcq(row, ev)
        return grade_direct(row, ev, judge)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(work, r) for r in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"grade/{probe}"):
            r = fut.result()
            if r:
                results.append(r)

    with open(out_path, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path
