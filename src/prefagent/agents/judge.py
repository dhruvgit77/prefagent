"""Judging: turns a prompt + N candidate responses into a preference label.

Two compute-matched methods (the DELIBERATION factor of the 2×2):
  single_judge  one model, S independent samples, each a final verdict
  panel         J models × 2 rounds; round 2 sees round-1 verdicts; round-2 = final

Both share everything else: prompt wording, shuffling, validation, aggregation.
Every call is stored with its permutation so position bias can be measured later.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from statistics import mean

from prefagent.agents import prompts
from prefagent.llm.client import ChatRequest, LLMClient
from prefagent.llm.parsing import extract_json

log = logging.getLogger(__name__)


@dataclass
class Verdict:
    """One validated judge call, expressed in ORIGINAL candidate indices."""

    judge: str
    round: int
    sample_idx: int
    permutation: list[int]          # shown position -> original candidate index
    scores: list[dict[str, int]]    # scores[i] = criterion -> score for candidate i
    best: int
    worst: int
    rationale: str
    attempts: int

    def overall(self, i: int) -> float:
        return mean(self.scores[i].values())


def validate(obj: dict | None, n: int, rubric: dict) -> tuple[dict, str, str, str]:
    """Check a parsed reply covers every response with in-range integer scores."""
    if obj is None:
        raise ValueError("no JSON object found")
    labels = list(prompts.LABELS[:n])
    lo, hi = rubric["scale"]
    raw = obj.get("scores")
    if not isinstance(raw, dict) or set(raw) != set(labels):
        raise ValueError(f"scores must cover exactly {labels}")
    scores = {}
    for label in labels:
        per = raw[label]
        if not isinstance(per, dict):
            raise ValueError(f"scores for {label} not an object")
        scores[label] = {}
        for c in rubric["criteria"]:
            v = per.get(c)
            if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                v = int(v)
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
                raise ValueError(f"{label}.{c} = {v!r} outside {lo}-{hi}")
            scores[label][c] = v
    best, worst = obj.get("best"), obj.get("worst")
    if best not in labels or worst not in labels or best == worst:
        raise ValueError(f"invalid best/worst: {best!r}/{worst!r}")
    return scores, best, worst, str(obj.get("rationale", ""))[:1000]


class Judge:
    def __init__(self, client: LLMClient, cfg: dict):
        self.client = client
        self.jcfg = cfg["judging"]
        self.rubric = self.jcfg["rubric"]
        self.seed = cfg["seeds"]["data"]
        self.system = prompts.judge_system(self.rubric)

    # -- one call -----------------------------------------------------------

    def permutation(self, n: int, key: str) -> list[int]:
        perm = list(range(n))
        if self.jcfg["shuffle_candidates"]:
            # String seeds are hashed deterministically (independent of PYTHONHASHSEED).
            random.Random(f"{self.seed}|{key}").shuffle(perm)
        return perm

    def call(self, judge: str, prompt: str, candidates: list[str], key: str,
             round_no: int = 1, sample_idx: int = 0,
             peers: list[Verdict] | None = None, self_idx: int | None = None,
             devils_advocate: bool = False) -> Verdict | None:
        n = len(candidates)
        perm = self.permutation(n, key)
        shown = [candidates[i] for i in perm]
        label_of = {orig: prompts.LABELS[pos] for pos, orig in enumerate(perm)}

        if peers is None:
            user = prompts.judge_user(prompt, shown)
        else:
            views = [self._view(k + 1, k == self_idx, v, label_of) for k, v in enumerate(peers)]
            target = None
            if devils_advocate:
                lowest = min(range(n), key=lambda i: mean(v.overall(i) for v in peers))
                target = label_of[lowest]
            user = prompts.debate_user(prompt, shown, views, target)

        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": user}]
        for attempt in range(1 + self.jcfg["max_parse_retries"]):
            resp = self.client.chat(ChatRequest(
                model=judge, messages=messages, temperature=self.jcfg["temperature"],
                max_tokens=self.jcfg["max_tokens"],
                json_mode=self.jcfg["response_format"] == "json",
                sample_idx=sample_idx, attempt=attempt, tag=f"judge:r{round_no}:{key}"))
            try:
                scores, best, worst, rationale = validate(extract_json(resp.text), n,
                                                          self.rubric)
            except ValueError as e:
                log.debug("%s attempt %d unparseable: %s", judge, attempt, e)
                continue
            pos = {label: p for p, label in enumerate(prompts.LABELS[:n])}
            return Verdict(
                judge=judge, round=round_no, sample_idx=sample_idx, permutation=perm,
                scores=[scores[label_of[i]] for i in range(n)],
                best=perm[pos[best]], worst=perm[pos[worst]],
                rationale=rationale, attempts=attempt + 1)
        log.warning("%s gave no valid verdict for %s after retries", judge, key)
        return None

    @staticmethod
    def _view(judge_no: int, is_self: bool, v: Verdict, label_of: dict[int, str]) -> str:
        # Peer verdicts are re-labelled into THIS call's shuffled labels.
        scores = {label_of[i]: s for i, s in enumerate(v.scores)}
        return prompts.render_view(judge_no, is_self, scores, label_of[v.best],
                                   label_of[v.worst], v.rationale)

    # -- the two methods ----------------------------------------------------

    def single_judge(self, prompt_id: str, prompt: str, candidates: list[str]) -> dict:
        cfg = self.jcfg["single_judge"]
        calls = [self.call(cfg["model"], prompt, candidates, f"{prompt_id}|single|{k}",
                           sample_idx=k) for k in range(cfg["samples"])]
        valid = [c for c in calls if c is not None]
        return self._result(prompt_id, "single_judge", valid, final=valid,
                            n=len(candidates), failed=len(calls) - len(valid))

    def panel(self, prompt_id: str, prompt: str, candidates: list[str]) -> dict:
        cfg = self.jcfg["panel"]
        judges = cfg["judges"]
        r1 = [self.call(j, prompt, candidates, f"{prompt_id}|panel|r1|{k}", round_no=1)
              for k, j in enumerate(judges)]
        r1_valid = [v for v in r1 if v is not None]
        if not r1_valid:
            return self._result(prompt_id, "panel", [], final=[], n=len(candidates),
                                failed=len(judges))
        r2 = []
        for k, j in enumerate(judges):
            # self_idx: where this judge's own round-1 verdict sits among the peers.
            self_idx = next((p for p, v in enumerate(r1_valid) if v.judge == j), None)
            r2.append(self.call(
                j, prompt, candidates, f"{prompt_id}|panel|r2|{k}", round_no=2,
                peers=r1_valid, self_idx=self_idx,
                devils_advocate=cfg["devils_advocate"] and k == 0))
        r2_valid = [v for v in r2 if v is not None]
        return self._result(prompt_id, "panel", r1_valid + r2_valid, final=r2_valid,
                            n=len(candidates),
                            failed=(len(r1) - len(r1_valid)) + (len(r2) - len(r2_valid)))

    # -- aggregation --------------------------------------------------------

    def _result(self, prompt_id: str, method: str, calls: list[Verdict],
                final: list[Verdict], n: int, failed: int) -> dict:
        return {"prompt_id": prompt_id, "method": method, "n_candidates": n,
                "calls": [asdict(c) for c in calls],
                "final": [i for i, c in enumerate(calls) if any(c is f for f in final)],
                "failed_calls": failed,
                "aggregate": aggregate(final, n, random.Random(f"{self.seed}|{prompt_id}|tie"))}


def aggregate(final: list[Verdict], n: int, rng: random.Random) -> dict | None:
    """Mean overall score per candidate across final verdicts → chosen / rejected.

    agreement = share of final verdicts that also rank chosen above rejected
    (ties count half). It is the per-pair confidence used by disagreement-aware DPO.
    """
    if not final or n < 2:
        return None
    means = [mean(v.overall(i) for v in final) for i in range(n)]
    hi, lo = max(means), min(means)
    if hi == lo:
        return None                      # judges saw no difference: no preference
    chosen = rng.choice([i for i in range(n) if means[i] == hi])
    rejected = rng.choice([i for i in range(n) if means[i] == lo])
    votes = [1.0 if v.overall(chosen) > v.overall(rejected)
             else 0.5 if v.overall(chosen) == v.overall(rejected) else 0.0 for v in final]
    return {"mean_scores": means, "chosen": chosen, "rejected": rejected,
            "agreement": mean(votes), "n_final": len(final)}


def from_dict(d: dict) -> Verdict:
    return Verdict(**d)
