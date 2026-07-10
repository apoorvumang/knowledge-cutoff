---
license: cc-by-4.0
language:
- en
task_categories:
- question-answering
tags:
- knowledge-cutoff
- llm-evaluation
- temporal-reasoning
- benchmark
pretty_name: Knowledge Cutoff Benchmark
size_categories:
- 1K<n<10K
configs:
- config_name: events
  default: true
  data_files: events.jsonl
- config_name: results
  data_files: results.jsonl
---

# Knowledge Cutoff Benchmark

A benchmark for estimating a language model's **effective knowledge cutoff** —
what it *actually* knows about the world — which is usually **earlier than the
cutoff date the model advertises**.

Each model is probed on curated, **surprising / unforecastable** real-world
events (deaths, changes of office) spread month-by-month across Jan 2024 –
Jun 2026. The month where per-month accuracy collapses is the model's effective
knowledge horizon.

- Code, methodology, and an interactive explorer: https://github.com/apoorvumang/knowledge-cutoff
- Live visualization: https://apoorvumang.github.io/knowledge-cutoff/

## Configs

### `events` (default) — the benchmark items (one row per event)

| field | meaning |
|---|---|
| `id` | unique event id |
| `date`, `month` | when it happened (`YYYY-MM-DD`, `YYYY-MM`) |
| `category` | `death`, `office_change`, `control_alive`, `fake_event` |
| `predictability` | `low` = unforecastable (highest signal) → `high` |
| `region` | `US` / `International` |
| `subject`, `fact` | who/what, and a one-sentence ground truth |
| `question_direct`, `expected_direct` | open probe + canonical answer |
| `mcq_question`, `mcq_choices`, `mcq_answer` | 4-way forced-choice probe |
| `source` | provenance URL |

`control_alive` (a famous person still living) and `fake_event` (an event that
never happened) are diagnostics: a trustworthy run answers them correctly, which
rules out a model that just always guesses "dead" or confabulates.

### `results` — model answers (one row per model × probe × event)

| field | meaning |
|---|---|
| `model`, `probe` | model key and `direct` (open) or `mcq` (forced choice) |
| `event_id`, `month`, `category`, `predictability` | joins back to `events` |
| `label` | `correct` / `incorrect` (confidently wrong) / `abstain` |
| `response` | the model's raw answer |

## Method (why the numbers are trustworthy)

1. **Surprising events only** — a model can't score by extrapolating pre-cutoff
   trends the way it could for a scheduled election.
2. **Three-way grading** — `correct` / `incorrect` / `abstain`. Hedging
   ("I'm not aware…") is abstention, not error; conflating them would corrupt
   the estimate. (Direct answers are graded by an LLM judge given the ground
   truth, so the judge's own cutoff is irrelevant; MCQ is graded by letter.)
3. **Controls** — living-person and fabricated-event rows that every run must
   pass.
4. **No date leakage** — prompts never reveal the current date.

`direct` under-counts (the model knows but doesn't volunteer); `mcq` over-counts
(guessing) — together they bracket the truth.

## Key finding

Across the evaluated frontier models, the **effective** knowledge cutoff is
consistently **~1–5 months earlier** than the **advertised** cutoff. Models that
publish no cutoff (e.g. some open-weight releases) can only be characterized by
this benchmark. See the live explorer for the per-model leaderboard, decay
curves, a heatmap with each model's claimed cutoff highlighted, and every
individual answer.

## Usage

```python
from datasets import load_dataset

events  = load_dataset("<your-username>/knowledge-cutoff-benchmark", "events",  split="train")
results = load_dataset("<your-username>/knowledge-cutoff-benchmark", "results", split="train")
```

## License

`cc-by-4.0`. Events are factual and drawn from public reporting (see each row's
`source`); please cite this dataset if you use it.
