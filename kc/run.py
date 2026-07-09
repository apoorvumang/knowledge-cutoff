"""Run a model over the event set and record raw responses.

Output: one JSONL row per (event, probe) with the raw model text. No grading
happens here — see grade.py. Runs are resumable: existing rows for the same
(model, probe) output file are skipped.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .prompts import build_direct_prompt, build_mcq_prompt
from .providers import Model, get_model
from .schema import Event, load_events


def _load_done(out_path: str) -> set[str]:
    done: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["event_id"])
                except Exception:
                    pass
    return done


def _one(model: Model, ev: Event, probe: str) -> dict:
    if probe == "direct":
        system, user = build_direct_prompt(ev)
    elif probe == "mcq":
        system, user = build_mcq_prompt(ev)
    else:
        raise ValueError(f"unknown probe {probe!r}")
    try:
        text, meta = model.complete(user, system)
        err = None
    except Exception as e:  # capture; a single failure shouldn't kill the run
        text, meta, err = "", {}, f"{type(e).__name__}: {e}"
    return {
        "event_id": ev.id,
        "month": ev.month,
        "category": ev.category,
        "predictability": ev.predictability,
        "probe": probe,
        "prompt": user,
        "response": text,
        "meta": meta,
        "error": err,
    }


def run_model(model_key: str, events_path: str, probe: str, out_path: str,
              concurrency: int = 8, limit: int | None = None) -> str:
    events = load_events(events_path)
    if limit:
        events = events[:limit]
    model = get_model(model_key)
    done = _load_done(out_path)
    todo = [ev for ev in events if ev.id not in done]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if done:
        print(f"resuming: {len(done)} already done, {len(todo)} to go")

    with open(out_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(_one, model, ev, probe): ev for ev in todo}
        errors = 0
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"{model_key}/{probe}"):
            row = fut.result()
            if row["error"]:
                errors += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    if errors:
        print(f"warning: {errors} calls errored (recorded with error field)")
    return out_path
