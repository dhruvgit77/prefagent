"""Parse Anthropic HH-RLHF rows into (prompt, human_chosen, human_rejected).

HH-RLHF has NO prompt column — each row is two full transcripts:
    chosen   = "\\n\\nHuman: <prompt>\\n\\nAssistant: <good reply>"
    rejected = "\\n\\nHuman: <prompt>\\n\\nAssistant: <bad reply>"
The prompt is the shared prefix up to the last "Assistant:" turn. This is the bug in
the original implementation guide, which assumed a `prompt` column existed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HUMAN = "\n\nHuman:"
ASSISTANT = "\n\nAssistant:"


@dataclass
class HHExample:
    prompt: str
    chosen: str
    rejected: str


def split_transcript(text: str) -> tuple[str, str]:
    """Return (prefix up to and including the last Assistant tag, final reply)."""
    idx = text.rfind(ASSISTANT)
    if idx == -1:
        raise ValueError("no Assistant turn")
    cut = idx + len(ASSISTANT)
    return text[:cut], text[cut:].strip()


def parse_single_turn(chosen: str, rejected: str) -> HHExample | None:
    """Parse a row if it is a clean single-turn exchange, else return None.

    Rejected rows: multi-turn dialogues (out of scope per the PRD), transcripts whose
    prefixes differ (malformed pair), empty replies, and identical replies (DPO gets
    zero gradient from them).
    """
    if chosen.count(HUMAN) != 1 or chosen.count(ASSISTANT) != 1:
        return None
    try:
        prefix_c, reply_c = split_transcript(chosen)
        prefix_r, reply_r = split_transcript(rejected)
    except ValueError:
        return None
    if prefix_c != prefix_r or not reply_c or not reply_r or reply_c == reply_r:
        return None
    prompt = prefix_c[len(HUMAN):-len(ASSISTANT)] if prefix_c.startswith(HUMAN) else None
    if prompt is None or not prompt.strip():
        return None
    return HHExample(prompt=prompt.strip(), chosen=reply_c, rejected=reply_r)


_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for exact-duplicate detection.
    HH-RLHF contains many repeated prompts with different reply pairs."""
    return _SPACE.sub(" ", _PUNCT.sub("", text.lower())).strip()
