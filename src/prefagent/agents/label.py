"""Turn candidates (+ judgements) into DPO preference pairs, one rule per condition.

  human         HH-RLHF's own chosen/rejected                      (reference)
  random        two candidates picked by a seeded coin             (sanity floor)
  longer        longest vs shortest candidate by word count        (length-bias probe)
  single_judge  aggregate of the single judge's verdicts           (2×2)
  panel         aggregate of the panel's round-2 verdicts          (2×2)
"""

from __future__ import annotations

import random

from prefagent.data.schema import PreferencePair


def human_pair(rec: dict, condition: str) -> PreferencePair | None:
    if rec.get("human_chosen") is None:
        return None
    return _pair(rec, rec["human_chosen"], rec["human_rejected"], condition,
                 meta={"label_source": "hh_rlhf"})


def random_pair(rec: dict, cands: dict, condition: str, seed: int) -> PreferencePair | None:
    texts = [c["text"] for c in cands["candidates"]]
    if len(texts) < 2:
        return None
    i, j = random.Random(f"{seed}|{rec['id']}|random").sample(range(len(texts)), 2)
    return _pair(rec, texts[i], texts[j], condition, meta={"chosen_idx": i, "rejected_idx": j})


def longer_pair(rec: dict, cands: dict, condition: str) -> PreferencePair | None:
    texts = [c["text"] for c in cands["candidates"]]
    if len(texts) < 2:
        return None
    words = [len(t.split()) for t in texts]
    if max(words) == min(words):
        return None
    i, j = words.index(max(words)), words.index(min(words))
    return _pair(rec, texts[i], texts[j], condition,
                 meta={"chosen_idx": i, "rejected_idx": j, "words": words})


def judged_pair(rec: dict, cands: dict, judgement: dict, condition: str) -> PreferencePair | None:
    agg = judgement.get("aggregate")
    if agg is None:
        return None
    texts = [c["text"] for c in cands["candidates"]]
    i, j = agg["chosen"], agg["rejected"]
    return _pair(rec, texts[i], texts[j], condition, agreement=agg["agreement"],
                 meta={"chosen_idx": i, "rejected_idx": j, "mean_scores": agg["mean_scores"],
                       "n_final": agg["n_final"], "method": judgement["method"]})


def _pair(rec: dict, chosen: str, rejected: str, condition: str,
          agreement: float | None = None, meta: dict | None = None) -> PreferencePair | None:
    if chosen.strip() == rejected.strip():
        return None
    return PreferencePair(prompt_id=rec["id"], prompt=rec["prompt"], chosen=chosen,
                          rejected=rejected, condition=condition, agreement=agreement,
                          meta=meta or {})
