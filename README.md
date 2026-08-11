# knowledge-cutoff

[![Live explorer](https://img.shields.io/badge/live-explorer-1f6feb)](https://apoorvumang.github.io/knowledge-cutoff/)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-HuggingFace-ffcc00)](https://huggingface.co/datasets/apoorvumang/knowledge-cutoff-benchmark)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)

**Interactive results → https://apoorvumang.github.io/knowledge-cutoff/ · Dataset → https://huggingface.co/datasets/apoorvumang/knowledge-cutoff-benchmark**

A benchmark to estimate the **effective knowledge cutoff** of a language model —
what it *actually* knows about the world, which is often earlier than the cutoff
date it advertises.

The idea: probe the model on a curated set of real-world events spread across
recent months. For each month, measure how many events the model gets right.
Knowledge decays as you approach the true horizon, so the per-month curve reveals
roughly when the model's world-knowledge ends.

## Why it's built the way it is

Naive versions of this benchmark are misleading. The design defends against four
specific failure modes:

1. **Predictable events leak fake "knowledge".** A model can answer "who won the
   2024 election" from pre-cutoff polling without having seen the result. So
   every event is tagged with `predictability` (low/medium/high) and the cutoff
   curve is computed over **low/medium-predictability events only** —
   assassinations, sudden deaths, shock resignations, upsets — things
   unforecastable before they happened.

2. **Hedging isn't the same as ignorance.** Models are trained to say "as of my
   last update I'm not aware…". Scoring that as "wrong" corrupts the estimate.
   Grading is **three-way**: `correct` / `incorrect` (confidently wrong — the
   strongest signal the event post-dates training) / `abstain`.

3. **Confabulation and over-prediction.** A model that always says "dead" looks
   knowledgeable on death questions by accident. The dataset includes
   **`control_alive`** rows (famous people who are alive) and **`fake_event`**
   rows (events that never happened). If living-person accuracy is low or
   fake-event confabulation is high, the run is untrustworthy — reported
   alongside the curve.

4. **Date leakage.** Prompts never mention today's date and the system prompt
   never hints this is a recency test, so the model can't infer the answer or
   strategically hedge.

It's a **decay, not a cliff** (training data thins out near the cutoff), so we
report the whole curve plus two cutoff estimates: `last_above` (latest month
above threshold) and `crossover` (last month above threshold that stays gone
afterward).

## Install

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e .
```

Provider API keys are read from environment variables named in `models.yaml`
(e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, …).

## Usage

```bash
kc validate                                   # check the dataset parses
kc models                                     # list configured models + key status

# one-shot: run -> grade -> score
kc eval  --model gpt-4o --probe direct
kc eval  --model claude-fable-5 --probe mcq

# or step by step
kc run   --model grok-4 --probe direct        # -> runs/grok-4__direct.jsonl
kc grade --run runs/grok-4__direct.jsonl      # -> graded/grok-4__direct.jsonl (LLM judge)
kc score --graded graded/grok-4__direct.jsonl # per-month curve + cutoff estimate
```

`--probe direct` asks an open question graded by an LLM judge (default judge:
`judge-opus-4-8`, given the ground truth so its own cutoff is irrelevant).

The judge is routed over **OpenRouter** (`anthropic/claude-opus-4.8`) so grading
needs only `OPENROUTER_API_KEY`, not a separate Anthropic balance. It is the same
underlying model as the native `claude-opus-4-8` entry, so grades stay comparable
with runs graded before the switch. Keep it that way: swapping the judge for a
different *model* would require re-grading every run to keep the leaderboard
self-consistent. (`claude-opus-4-8` stays on the native Anthropic route because it
is also a benchmark *subject*, and subjects are measured on their vendor's API.)
`--probe mcq` is a 4-way forced choice graded deterministically by letter.
Run both: `direct` under-counts (the model knows but doesn't volunteer), `mcq`
over-counts (guessing), so they bracket the truth.

## Dataset

`data/events.jsonl`, one event per line. Schema and category semantics are
documented in `kc/schema.py`. To add events, append lines and run `kc validate`.

| field | meaning |
|---|---|
| `category` | `death`, `office_change`, `control_alive`, `fake_event` |
| `predictability` | `low` = unforecastable (best signal) → `high` |
| `question_direct` / `expected_direct` | open probe + ground-truth answer for the judge |
| `mcq_question` / `mcq_choices` / `mcq_answer` | forced-choice probe |
| `region`, `source` | US/international balance; provenance URL |

## Adding a model

Add an entry under `models:` in `models.yaml`. Any OpenAI-compatible endpoint
works via a `kind: openai` provider block (set `base_url` + `api_key_env`);
Anthropic models use `kind: anthropic`. Pin dated revisions (`…-0731`) rather
than floating aliases — a re-post-trained build can move the knowledge horizon,
so the run needs to say which one it measured.

To surface the model in the explorer, also add it to `MODEL_META` in
`scripts/export_viz.py` (display name + **advertised** cutoff, sourced from
official provider docs — use `not published` rather than an unattributed
secondary figure) and to `MC` / `MODEL_PROV` in `report_template.html`. A
provider logo is picked up automatically from `assets/logos/<provider>.svg`;
without one the badge falls back to the provider's initial.

Note the categorical palette is full at ~12 series: no sRGB color clears the
ΔE 15 normal-vision floor against the existing set (the ceiling is 14.8). Past
that, series identity rests on the legend, badge and tooltip rather than hue —
prefer faceting or a smaller default selection over a 16th color.

### Data hygiene

Runs are resumable by `event_id`, so a row that errored or came back blank is
otherwise cached as "done" forever. Two flags handle that:

```bash
kc eval --model <key> --probe mcq --retry-failed --max-tokens 4096
```

* `--retry-failed` drops rows that errored or returned blank/truncated output
  (`finish_reason: length`) and re-queries them.
* `--max-tokens` raises the per-call budget (default 2048). Reasoning models can
  spend the whole budget before emitting an answer.

Anything that couldn't be graded — an API error, a truncated completion, or a
judge that errored — is labelled **`ungraded`**, never `abstain`. An abstention
is a real model behaviour; ungraded is missing data. Scoring drops ungraded rows
from every denominator and prints a warning, and `export_viz.py` refuses to
publish a probe with >5% ungraded rows. (Without this, a judge whose API credit
ran out silently renders *every* model as 100% abstain, which looks like a
finding rather than an outage.)
