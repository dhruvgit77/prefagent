"""JSONL helpers. Every dataset in this project is JSONL: one record per line, human-
readable, diff-able, appendable (so an interrupted labelling run can resume)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # A half-written last line from a crash is the usual cause.
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """Write atomically: to a temp file first, then rename, so a crash never leaves
    a truncated dataset that looks complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record and flush, used for resumable long-running loops."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
