"""SQLite cache of every LLM request/response.

Why it matters for the research, not just for speed:
  * Re-running a stage after a crash or a code fix costs zero free-tier quota.
  * The raw outputs behind the published dataset are archived verbatim.
  * Token counts per model are recorded, so the paper can report exact API compute.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


def request_key(payload: dict) -> str:
    """Stable hash of everything that determines a response (incl. sample index)."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across worker threads, guarded by a lock: simpler and
        # safer than per-thread connections for a write-heavy, low-volume workload.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS responses (
                       key TEXT PRIMARY KEY,
                       model TEXT NOT NULL,
                       request TEXT NOT NULL,
                       response TEXT NOT NULL,
                       prompt_tokens INTEGER,
                       completion_tokens INTEGER,
                       created_at REAL NOT NULL)"""
            )
            self._conn.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, request: dict, response: dict) -> None:
        usage = response.get("usage") or {}
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, model, json.dumps(request, ensure_ascii=False),
                 json.dumps(response, ensure_ascii=False),
                 usage.get("prompt_tokens"), usage.get("completion_tokens"), time.time()),
            )
            self._conn.commit()

    def usage_by_model(self) -> dict[str, dict]:
        """Total calls and tokens per model — for the paper's compute statement."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens)
                   FROM responses GROUP BY model"""
            ).fetchall()
        return {m: {"calls": c, "prompt_tokens": p or 0, "completion_tokens": o or 0}
                for m, c, p, o in rows}
