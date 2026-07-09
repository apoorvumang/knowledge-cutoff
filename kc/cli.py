"""kc — command line for the knowledge-cutoff benchmark.

    kc validate                              # check the dataset parses
    kc models                                # list configured models
    kc run   --model M --probe direct|mcq    # query a model, save raw responses
    kc grade --run FILE                       # 3-way grade a raw run
    kc score --graded FILE                    # per-month curve + cutoff estimate
    kc eval  --model M --probe ...            # run -> grade -> score in one shot
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_EVENTS = "data/events.jsonl"


def _paths(model: str, probe: str) -> tuple[str, str]:
    raw = f"runs/{model}__{probe}.jsonl"
    graded = f"graded/{model}__{probe}.jsonl"
    return raw, graded


def cmd_validate(args):
    from .schema import load_events
    events = load_events(args.events)
    from collections import Counter
    cats = Counter(e.category for e in events)
    months = Counter(e.month for e in events)
    pred = Counter(e.predictability for e in events)
    print(f"OK: {len(events)} events parse cleanly")
    print(f"  categories: {dict(cats)}")
    print(f"  predictability: {dict(pred)}")
    print(f"  months: {len(months)} ({min(months)} .. {max(months)})")
    thin = [m for m, c in sorted(months.items()) if c < 5]
    if thin:
        print(f"  note: months with <5 events: {thin}")


def cmd_models(args):
    from .providers import load_registry
    reg = load_registry(args.registry)
    for k in reg.keys():
        spec = reg.models[k]
        have = "ok " if os.environ.get(spec.api_key_env) else "NO-KEY"
        print(f"  [{have}] {k:22s} {spec.provider}:{spec.model_id}")


def cmd_run(args):
    from .run import run_model
    raw, _ = _paths(args.model, args.probe)
    out = args.out or raw
    run_model(args.model, args.events, args.probe, out,
              concurrency=args.concurrency, limit=args.limit)
    print(f"wrote {out}")


def cmd_grade(args):
    from .grade import grade_run
    out = args.out or args.run.replace("runs/", "graded/")
    grade_run(args.run, args.events, out, judge_key=args.judge,
              concurrency=args.concurrency)
    print(f"wrote {out}")


def cmd_score(args):
    from .score import summarize, format_report
    summary = summarize(args.graded, threshold=args.threshold)
    print(format_report(summary, title=os.path.basename(args.graded)))


def cmd_eval(args):
    from .run import run_model
    from .grade import grade_run
    from .score import summarize, format_report
    raw, graded = _paths(args.model, args.probe)
    run_model(args.model, args.events, args.probe, raw,
              concurrency=args.concurrency, limit=args.limit)
    grade_run(raw, args.events, graded, judge_key=args.judge,
              concurrency=args.concurrency)
    summary = summarize(graded, threshold=args.threshold)
    print()
    print(format_report(summary, title=f"{args.model} / {args.probe}"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="kc", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", default=DEFAULT_EVENTS)
    p.add_argument("--registry", default="models.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("validate"); sp.set_defaults(func=cmd_validate)
    sp = sub.add_parser("models"); sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("run")
    sp.add_argument("--model", required=True)
    sp.add_argument("--probe", choices=["direct", "mcq"], default="direct")
    sp.add_argument("--out")
    sp.add_argument("--concurrency", type=int, default=8)
    sp.add_argument("--limit", type=int)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("grade")
    sp.add_argument("--run", required=True)
    sp.add_argument("--out")
    sp.add_argument("--judge", default="claude-opus-4-8")
    sp.add_argument("--concurrency", type=int, default=8)
    sp.set_defaults(func=cmd_grade)

    sp = sub.add_parser("score")
    sp.add_argument("--graded", required=True)
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("eval")
    sp.add_argument("--model", required=True)
    sp.add_argument("--probe", choices=["direct", "mcq"], default="direct")
    sp.add_argument("--judge", default="claude-opus-4-8")
    sp.add_argument("--concurrency", type=int, default=8)
    sp.add_argument("--limit", type=int)
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
