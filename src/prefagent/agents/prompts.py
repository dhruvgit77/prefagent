"""Prompt templates for the judges.

Kept in one file so the exact wording used to produce the dataset can be quoted in the
paper's appendix and changed in one place (rubric wording is itself an ablation axis).
"""

from __future__ import annotations

import json
import string

LABELS = string.ascii_uppercase  # responses are shown as A, B, C, ... never as indices


def judge_system(rubric: dict) -> str:
    lo, hi = rubric["scale"]
    criteria = rubric["criteria"]
    example = {
        "scores": {"A": {c: hi for c in criteria}, "B": {c: lo for c in criteria}},
        "best": "A",
        "worst": "B",
        "rationale": "One or two sentences.",
    }
    return (
        "You are an impartial expert judge of AI assistant responses.\n"
        f"{rubric['instructions']}\n\n"
        f"Score EVERY response on each criterion from {lo} (very poor) to {hi} (excellent): "
        f"{', '.join(criteria)}.\n"
        "Then name the single best and the single worst response (they must differ).\n"
        "Respond with ONLY a JSON object in exactly this shape, covering every response:\n"
        f"{json.dumps(example)}\n"
        "Keep the rationale under 80 words."
    )


def judge_user(prompt: str, responses: list[str]) -> str:
    parts = [f"## User request\n{prompt}"]
    for label, text in zip(LABELS, responses):
        parts.append(f"## Response {label}\n{text}")
    return "\n\n".join(parts)


def debate_user(prompt: str, responses: list[str], peer_views: list[str],
                devils_advocate_target: str | None = None) -> str:
    """Round 2: the same responses plus every judge's round-1 verdict (anonymised)."""
    parts = [judge_user(prompt, responses), "## Round-1 verdicts from the panel"]
    parts.extend(peer_views)
    if devils_advocate_target:
        parts.append(
            "## Your role this round\n"
            f"Before scoring, make the strongest honest case FOR Response "
            f"{devils_advocate_target}, which the panel rated lowest. Then score all "
            "responses on their merits; change scores only if the case is convincing."
        )
    parts.append(
        "## Instructions\nReconsider the responses in light of the other verdicts. "
        "Keep or revise your scores — do not simply defer to the majority. "
        "Return the same JSON format."
    )
    return "\n\n".join(parts)


def render_view(judge_no: int, is_self: bool, scores: dict[str, dict[str, int]],
                best: str, worst: str, rationale: str) -> str:
    who = f"Judge {judge_no}" + (" (you)" if is_self else "")
    lines = [f"### {who}: best={best}, worst={worst}"]
    for label in sorted(scores):
        crit = ", ".join(f"{k}={v}" for k, v in scores[label].items())
        lines.append(f"- {label}: {crit}")
    if rationale:
        lines.append(f"Rationale: {rationale}")
    return "\n".join(lines)
