"""Event schema + loading/validation for the benchmark dataset.

An *event* is a single verifiable real-world fact tied to a month. Each event
carries two probes:
  - direct : an open, neutrally-phrased question (graded by an LLM judge into
             correct / incorrect / abstain)
  - mcq    : a 4-way multiple-choice question (graded by letter extraction)

Categories:
  death         : a notable person died (crisp binary ground truth)
  office_change : someone gained/left an office or role
  control_alive : a notable person who is STILL ALIVE as of the dataset date.
                  Catches models that over-predict deaths. `month` is the
                  as-of date, not an event date; these are excluded from the
                  cutoff-curve numerator.
  fake_event    : an event that never happened. Measures confabulation /
                  false-positive rate. Also excluded from the curve.

`predictability` (low|medium|high) is the ex-ante surprise: low = unforecastable
(assassination, sudden death, upset) = the highest-signal events. The cutoff
curve is computed over low/medium-predictability real events only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

REAL_CATEGORIES = {"death", "office_change"}
CONTROL_CATEGORIES = {"control_alive", "fake_event"}
ALL_CATEGORIES = REAL_CATEGORIES | CONTROL_CATEGORIES
PREDICTABILITY = {"low", "medium", "high"}


@dataclass
class Event:
    id: str
    date: str            # YYYY-MM-DD (as-of date for controls)
    month: str           # YYYY-MM
    category: str
    region: str
    predictability: str
    subject: str
    fact: str
    question_direct: str
    expected_direct: str
    mcq_question: str
    mcq_choices: list[str]
    mcq_answer: str      # single letter A-D
    source: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_real(self) -> bool:
        return self.category in REAL_CATEGORIES

    @property
    def counts_for_curve(self) -> bool:
        """Real events with genuine surprise drive the cutoff estimate."""
        return self.is_real and self.predictability in {"low", "medium"}

    def mcq_letters(self) -> list[str]:
        return [c.strip()[0].upper() for c in self.mcq_choices]


class SchemaError(ValueError):
    pass


_REQUIRED = [
    "id", "date", "month", "category", "region", "predictability", "subject",
    "fact", "question_direct", "expected_direct",
    "mcq_question", "mcq_choices", "mcq_answer",
]


def _validate(d: dict[str, Any], lineno: int) -> Event:
    missing = [k for k in _REQUIRED if k not in d]
    if missing:
        raise SchemaError(f"line {lineno}: missing fields {missing}")
    if d["category"] not in ALL_CATEGORIES:
        raise SchemaError(f"line {lineno} ({d['id']}): bad category {d['category']!r}")
    if d["predictability"] not in PREDICTABILITY:
        raise SchemaError(
            f"line {lineno} ({d['id']}): bad predictability {d['predictability']!r}")
    if not isinstance(d["mcq_choices"], list) or len(d["mcq_choices"]) != 4:
        raise SchemaError(f"line {lineno} ({d['id']}): mcq_choices must be 4 items")
    ans = str(d["mcq_answer"]).strip().upper()
    if ans not in {"A", "B", "C", "D"}:
        raise SchemaError(f"line {lineno} ({d['id']}): mcq_answer must be A-D, got {ans!r}")
    if not str(d["month"]).count("-") == 1 or len(str(d["month"])) != 7:
        raise SchemaError(f"line {lineno} ({d['id']}): month must be YYYY-MM, got {d['month']!r}")
    return Event(
        id=str(d["id"]),
        date=str(d["date"]),
        month=str(d["month"]),
        category=str(d["category"]),
        region=str(d["region"]),
        predictability=str(d["predictability"]),
        subject=str(d["subject"]),
        fact=str(d["fact"]),
        question_direct=str(d["question_direct"]),
        expected_direct=str(d["expected_direct"]),
        mcq_question=str(d["mcq_question"]),
        mcq_choices=[str(c) for c in d["mcq_choices"]],
        mcq_answer=ans,
        source=str(d.get("source", "")),
        raw=d,
    )


def load_events(path: str) -> list[Event]:
    events: list[Event] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise SchemaError(f"line {lineno}: invalid JSON: {e}") from e
            ev = _validate(d, lineno)
            if ev.id in seen:
                raise SchemaError(f"line {lineno}: duplicate id {ev.id!r}")
            seen.add(ev.id)
            events.append(ev)
    return events
