"""Prompt construction for the two probe modes.

Guiding principle: reveal NOTHING about the current date or the expected answer.
The whole point is to see what the model volunteers on its own.
"""

from __future__ import annotations

from .schema import Event

# Deliberately minimal. We do NOT tell the model the date or that this is a
# recency test — that would let it hedge strategically.
DIRECT_SYSTEM = (
    "Answer the question directly and factually based on what you know. "
    "If you are not sure, say so, but give your best answer."
)


def build_direct_prompt(ev: Event) -> tuple[str, str]:
    """Open free-response probe. Returns (system, user)."""
    return DIRECT_SYSTEM, ev.question_direct


def build_mcq_prompt(ev: Event) -> tuple[str, str]:
    """Forced-choice probe. Returns (system, user)."""
    system = (
        "Answer the multiple-choice question. Respond with ONLY the single "
        "letter (A, B, C, or D) of the best answer, nothing else."
    )
    choices = "\n".join(ev.mcq_choices)
    user = f"{ev.mcq_question}\n\n{choices}\n\nAnswer with a single letter:"
    return system, user
