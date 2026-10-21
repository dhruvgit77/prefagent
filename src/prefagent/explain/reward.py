"""Train and explain a lightweight reward model from preference pairs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

HEDGE_TERMS = ("might", "possibly")


def preference_pairs(rows: Iterable[dict], limit: int | None = None) -> list[dict]:
    """Normalize HH-RLHF prompt rows or DPO preference-pair rows."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    pairs = []
    for line_no, row in enumerate(rows, 1):
        if "chosen" in row and "rejected" in row:
            chosen, rejected = row["chosen"], row["rejected"]
            source = "preference_pairs"
        else:
            chosen, rejected = row.get("human_chosen"), row.get("human_rejected")
            source = "human_labels"
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            raise ValueError(f"record {line_no}: missing string chosen/rejected responses")
        if not chosen.strip() or not rejected.strip():
            raise ValueError(f"record {line_no}: chosen/rejected responses must not be empty")
        if chosen.strip() == rejected.strip():
            raise ValueError(f"record {line_no}: chosen and rejected responses are identical")
        pairs.append({"chosen": chosen, "rejected": rejected, "source": source})
        if limit is not None and len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError("input contains no preference pairs")
    return pairs


def response_features(embeddings: np.ndarray, responses: list[str]) -> tuple[np.ndarray, list[str]]:
    """Combine sentence embeddings with the plan's response-length and hedging signals."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if len(embeddings) != len(responses):
        raise ValueError("embedding count must match response count")
    word_counts = np.asarray([len(text.split()) for text in responses], dtype=np.float32)
    hedge_counts = np.asarray(
        [sum(text.lower().count(term) for term in HEDGE_TERMS) for text in responses],
        dtype=np.float32,
    )
    features = np.column_stack((embeddings, word_counts, hedge_counts))
    names = [f"sentence_embedding_{i}" for i in range(embeddings.shape[1])]
    names.extend(("word_count", "hedge_count"))
    return features, names
