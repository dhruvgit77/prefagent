"""GSM8K answer extraction — the objective yardstick for label accuracy.

On math, a preference label is checkably right or wrong: a judge that prefers a
candidate with the wrong final number over one with the right number made an error,
no human annotator needed.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANSWER_CUES = (
    re.compile(r"####\s*(-?[\d,]*\.?\d+)"),
    re.compile(r"answer is[:\s]*\$?\\?(?:boxed\{)?\s*(-?[\d,]*\.?\d+)", re.I),
    re.compile(r"\\boxed\{\s*\$?(-?[\d,]*\.?\d+)"),
)


def gold_answer(solution: str) -> str:
    """GSM8K solutions end with '#### <answer>'."""
    m = re.search(r"####\s*(.+)$", solution.strip())
    if not m:
        raise ValueError("GSM8K solution without '####' answer line")
    return _canonical(m.group(1))


def extract_answer(text: str) -> str | None:
    """Final numeric answer from a free-form response: explicit cues first
    ('####', 'the answer is', \\boxed{}), else the last number in the text."""
    for pattern in _ANSWER_CUES:
        matches = pattern.findall(text)
        if matches:
            return _canonical(matches[-1])
    numbers = _NUMBER.findall(text)
    return _canonical(numbers[-1]) if numbers else None


def is_correct(response: str, gold: str) -> bool:
    pred = extract_answer(response)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def _canonical(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s
