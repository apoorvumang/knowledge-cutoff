"""Run a model over the event set and record raw responses.

Output: one JSONL row per (event, probe) with the raw model text. No grading
happens here — see grade.py. Runs are resumable: existing rows for the same
(model, probe) output file are skipped.
"""

from __future__ import annotations

import json
import os
import random
import time
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


def _is_unusable(row: dict) -> bool:
    """A row carrying no answer: an API error, or a truncated/blank completion.

    finish_reason == "length" means the model spent its budget (usually on
    reasoning tokens) before emitting an answer. Keeping such a row would let
    the judge score it as an abstain, which reads as "didn't know" and biases
    the model's effective cutoff earlier. It is a harness artifact, not a
    result, so it should be re-queried rather than cached as done.
    """
    if row.get("error"):
        return True
    return not (row.get("response") or "").strip()


def _prune_unusable(out_path: str) -> int:
    """Rewrite out_path keeping only usable rows; return how many were dropped."""
    if not os.path.exists(out_path):
        return 0
    kept, dropped = [], 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if _is_unusable(row):
                dropped += 1
            else:
                kept.append(line)
    if dropped:
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
        os.replace(tmp, out_path)
    return dropped


# Transient conditions worth waiting out rather than recording as a result. A rate
# limit is the provider asking us to slow down; writing it into the run file as a
# failed row means the event is either lost or has to be hunted down with
# --retry-failed later.
_RETRYABLE = ("ratelimit", "429", "timeout", "timedout", "apiconnection",
              "internalserver", "500", "502", "503", "504", "overloaded")


def _is_retryable(e: Exception) -> bool:
    blob = f"{type(e).__name__} {e}".lower().replace(" ", "").replace("_", "")
    return any(t in blob for t in _RETRYABLE)


def _one(model: Model, ev: Event, probe: str, max_tokens: int | None = None,
         attempts: int = 5) -> dict:
    if probe == "direct":
        system, user = build_direct_prompt(ev)
    elif probe == "mcq":
        system, user = build_mcq_prompt(ev)
    else:
        raise ValueError(f"unknown probe {probe!r}")
    kw = {"max_tokens": max_tokens} if max_tokens else {}
    text, meta, err = "", {}, None
    for attempt in range(attempts):
        try:
            text, meta = model.complete(user, system, **kw)
            err = None
            break
        except Exception as e:  # capture; a single failure shouldn't kill the run
            err = f"{type(e).__name__}: {e}"
            if attempt == attempts - 1 or not _is_retryable(e):
                text, meta = "", {}
                break
            # exponential backoff with jitter, so parallel workers don't retry in step
            time.sleep(min(30.0, 1.5 * (2 ** attempt)) * (0.6 + random.random() * 0.8))
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
              concurrency: int = 8, limit: int | None = None,
              max_tokens: int | None = None, retry_failed: bool = False) -> str:
    events = load_events(events_path)
    if limit:
        events = events[:limit]
    model = get_model(model_key)
    if retry_failed:
        dropped = _prune_unusable(out_path)
        if dropped:
            print(f"retry-failed: dropped {dropped} unusable row(s), will re-query")
    done = _load_done(out_path)
    todo = [ev for ev in events if ev.id not in done]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if done:
        print(f"resuming: {len(done)} already done, {len(todo)} to go")

    with open(out_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(_one, model, ev, probe, max_tokens): ev for ev in todo}
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
