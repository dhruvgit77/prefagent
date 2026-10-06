"""Tolerant JSON extraction from LLM output.

Even in JSON mode, models wrap output in ```json fences, prepend a sentence, or (for
providers without JSON mode) add commentary. We extract the outermost {...} object;
anything that still fails to parse is treated as a failed call and re-asked.
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict | None:
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for chunk in candidates:
        start, end = chunk.find("{"), chunk.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            obj = json.loads(chunk[start:end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
