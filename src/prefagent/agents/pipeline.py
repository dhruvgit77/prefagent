"""Orchestration for Stage 2: generate candidates → judge → build pairs.

Files (all JSONL, one record per prompt, all resumable):
  data/candidates/<split>/<source>.jsonl
  data/judgements/<split>/<source>__<method>.jsonl
  data/pairs/<split>/<condition>.jsonl   (+ manifest.json)

Every stage skips prompts already present in its output, so an interrupted run (quota
exhausted, laptop asleep) resumes where it stopped; the LLM cache makes repeats free.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path

from tqdm import tqdm

from prefagent.agents import label
from prefagent.agents.generate import generate
from prefagent.agents.judge import Judge
from prefagent.llm.client import LLMClient
from prefagent.utils.io import append_jsonl, read_jsonl, write_json, write_jsonl

log = logging.getLogger(__name__)

JUDGED = ("single_judge", "panel")


# ---------------------------------------------------------------------------
# paths and loading
# ---------------------------------------------------------------------------

def data_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"])


def prompts_path(cfg: dict, split: str) -> Path:
    return data_dir(cfg) / "prompts" / f"{split}.jsonl"


def candidates_path(cfg: dict, split: str, source: str) -> Path:
    return data_dir(cfg) / "candidates" / split / f"{source}.jsonl"


def judgements_path(cfg: dict, split: str, source: str, method: str) -> Path:
    return data_dir(cfg) / "judgements" / split / f"{source}__{method}.jsonl"


def pairs_dir(cfg: dict, split: str) -> Path:
    return data_dir(cfg) / "pairs" / split


def load_prompts(cfg: dict, split: str, limit: int | None = None) -> list[dict]:
    rows = read_jsonl(prompts_path(cfg, split))
    return rows[:limit] if limit else rows


def _done_ids(path: Path) -> set[str]:
    return {r["prompt_id"] for r in read_jsonl(path)} if path.exists() else set()


def _by_id(path: Path) -> dict[str, dict]:
    return {r["prompt_id"]: r for r in read_jsonl(path)} if path.exists() else {}


def _run_resumable(items: list, fn, out: Path, workers: int, desc: str) -> int:
    """Apply fn to items in parallel, appending each result to out as it finishes.

    A prompt that still fails after the client's retries (e.g. daily quota exhausted) is
    logged and skipped, not fatal: it is simply absent from `out`, so re-running the same
    command later picks it up.
    """
    lock = threading.Lock()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, it): it for it in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001 — any failure just leaves the item undone
                failed += 1
                log.error("%s: item failed, will retry on next run: %s", desc, e)
                continue
            with lock:
                append_jsonl(out, result)
            done += 1
    if failed:
        log.warning("%s: %d done, %d failed — re-run the same command to retry", desc,
                    done, failed)
    return done


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def run_generate(cfg: dict, client: LLMClient, split: str, source: str,
                 limit: int | None = None, workers: int = 4) -> int:
    out = candidates_path(cfg, split, source)
    done = _done_ids(out)
    todo = [r for r in load_prompts(cfg, split, limit) if r["id"] not in done]
    log.info("generate %s/%s: %d to do, %d already done", split, source, len(todo), len(done))
    return _run_resumable(todo, lambda r: generate(client, cfg, r, source), out, workers,
                          f"gen {split}/{source}")


def run_judge(cfg: dict, client: LLMClient, split: str, source: str, method: str,
              limit: int | None = None, workers: int = 4) -> int:
    if method not in JUDGED:
        raise ValueError(f"method must be one of {JUDGED}")
    prompts = {r["id"]: r for r in load_prompts(cfg, split, limit)}
    cands_file = candidates_path(cfg, split, source)
    if source == "human_pair" and not cands_file.exists():
        run_generate(cfg, None, split, source, limit)      # no API calls needed
    cands = _by_id(cands_file)
    out = judgements_path(cfg, split, source, method)
    done = _done_ids(out)
    todo = [pid for pid in prompts if pid in cands and pid not in done]
    missing = [pid for pid in prompts if pid not in cands]
    if missing:
        log.warning("%d prompts have no candidates yet for %s/%s", len(missing), split, source)
    judge = Judge(client, cfg)
    run = getattr(judge, method)

    def one(pid: str) -> dict:
        texts = [c["text"] for c in cands[pid]["candidates"]]
        if len(texts) < 2:
            return {"prompt_id": pid, "method": method, "n_candidates": len(texts),
                    "calls": [], "final": [], "failed_calls": 0, "aggregate": None}
        return run(pid, prompts[pid]["prompt"], texts)

    log.info("judge %s/%s/%s: %d to do, %d already done", split, source, method,
             len(todo), len(done))
    return _run_resumable(todo, one, out, workers, f"judge {split}/{source}/{method}")


# ---------------------------------------------------------------------------
# the labelling plan and its API budget
# ---------------------------------------------------------------------------

def label_plan(cfg: dict) -> list[dict]:
    """Every (split, source, method, n_prompts) the experiments need, in run order.

    Train prompts: the main grid needs data_sizes.main prompts per combination; the
    scaling conditions need the largest scaling size. Dev: all of dev for every combo.
    Label quality: human pairs (label_agreement) and GSM8K, judged both ways.
    """
    conds = {k: v for k, v in cfg["conditions"].items() if v.get("train", True)}
    main_n = cfg["data_sizes"]["main"]
    scale_n = max(cfg["data_sizes"]["scaling"], default=0)
    scaling = set(cfg["data_sizes"]["scaling_conditions"])
    n_dev = cfg["splits"]["dev_pairs"]

    needs: dict[tuple, int] = defaultdict(int)  # (source, method|None) -> train prompts
    for name, c in conds.items():
        src, lab = c["candidates"], c["labels"]
        if src in ("hh_rlhf", "policy_samples"):
            continue          # human labels need no API; policy samples come from the GPU
        n = scale_n if name in scaling else main_n
        needs[(src, None)] = max(needs[(src, None)], n)
        if lab in JUDGED:
            needs[(src, lab)] = max(needs[(src, lab)], n)

    plan = []
    for (src, method), n in sorted(needs.items(), key=lambda kv: (kv[0][1] is not None, kv[0])):
        plan.append({"split": "train", "source": src, "method": method, "n": n})
        if n_dev:
            plan.append({"split": "dev", "source": src, "method": method, "n": n_dev})
    n_agree = cfg["splits"]["label_agreement_pairs"]
    for method in JUDGED if n_agree else ():
        plan.append({"split": "label_agreement", "source": "human_pair", "method": method,
                     "n": n_agree})
    if cfg["gsm8k"]["enabled"]:
        n_g = cfg["splits"]["gsm8k_label_problems"]
        plan.append({"split": "gsm8k_label", "source": "multi_persona", "method": None, "n": n_g})
        for method in JUDGED:
            plan.append({"split": "gsm8k_label", "source": "multi_persona", "method": method,
                         "n": n_g})
    return plan


def budget(cfg: dict, plan: list[dict]) -> dict[str, dict]:
    """Upper-bound API calls per model for a plan (excludes parse retries, ~a few %)."""
    calls: dict[str, int] = defaultdict(int)
    g, j = cfg["generation"], cfg["judging"]
    for step in plan:
        n = step["n"]
        if step["method"] is None:
            if step["source"] != "human_pair":
                calls[g["model"]] += n * g["n_candidates"]
        elif step["method"] == "single_judge":
            calls[j["single_judge"]["model"]] += n * j["single_judge"]["samples"]
        else:
            for judge in j["panel"]["judges"]:
                calls[judge] += n * j["panel"]["rounds"]
    out = {}
    for model, n in sorted(calls.items()):
        rpd = cfg["llms"][model].get("rpd")
        out[model] = {"calls": n, "rpd": rpd, "days": ceil(n / rpd) if rpd else None}
    return out


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------

def build_pairs(cfg: dict, split: str, force: bool = False) -> dict:
    """Write one pairs file per trained condition, all over the SAME prompts.

    Conditions can lose different prompts (judges tie, candidates truncated). Training
    each on whatever survived would change the prompts along with the labels, so we keep
    only prompts valid in every condition of the group, in prompt-file order. Trainers
    take the first N rows, which makes the scaling subsets nested (250 ⊂ 500 ⊂ ...).
    """
    seed = cfg["seeds"]["data"]
    prompts = load_prompts(cfg, split)
    conds = {k: v for k, v in cfg["conditions"].items() if v.get("train", True)}

    pairs: dict[str, dict[str, dict]] = {}
    for name, c in conds.items():
        built = _pairs_for(cfg, split, name, c, prompts, seed)
        if built is None:
            log.warning("%s: inputs missing for %s — excluded", split, name)
            continue
        pairs[name] = built

    scaling = [c for c in cfg["data_sizes"]["scaling_conditions"] if c in pairs]
    groups = {"main": list(pairs), "scaling": scaling}
    common: dict[str, list[str]] = {}
    manifest = {"split": split, "groups": {}}
    for group, members in groups.items():
        if members:
            common[group] = [r["id"] for r in prompts
                             if all(r["id"] in pairs[m] for m in members)]
            manifest["groups"][group] = {
                "conditions": members, "n_prompts": len(common[group]),
                "per_condition_valid": {m: len(pairs[m]) for m in members}}

    # Check BEFORE writing: a different condition set changes the shared prompt set,
    # which would silently invalidate models already trained on the old files.
    out_dir = pairs_dir(cfg, split)
    manifest_path = out_dir / "manifest.json"
    prev = read_json_safe(manifest_path) if manifest_path.exists() else None
    if prev and not force and \
            prev["groups"].get("main", {}).get("conditions") != groups["main"]:
        raise RuntimeError("Condition set changed since pairs were last built — this "
                           "changes which prompts every condition trains on. Re-run with "
                           "--force only if no model has been trained yet.")

    for group, ids in common.items():
        suffix = "" if group == "main" else ".scaling"
        for m in groups[group]:
            write_jsonl(out_dir / f"{m}{suffix}.jsonl", (pairs[m][pid] for pid in ids))
    write_json(manifest_path, manifest)
    return manifest


def _pairs_for(cfg, split, name, c, prompts, seed) -> dict[str, dict] | None:
    src, lab = c["candidates"], c["labels"]
    out = {}
    if lab == "human":
        for r in prompts:
            if (p := label.human_pair(r, name)) is not None:
                out[r["id"]] = p.model_dump()
        return out
    cands = _by_id(candidates_path(cfg, split, src))
    if not cands:
        return None
    judgements = {}
    if lab in JUDGED:
        judgements = _by_id(judgements_path(cfg, split, src, lab))
        if not judgements:
            return None
    for r in prompts:
        cand = cands.get(r["id"])
        if cand is None:
            continue
        if lab == "random":
            p = label.random_pair(r, cand, name, seed)
        elif lab == "longer":
            p = label.longer_pair(r, cand, name)
        else:
            jd = judgements.get(r["id"])
            p = label.judged_pair(r, cand, jd, name) if jd else None
        if p is not None:
            out[r["id"]] = p.model_dump()
    return out


def read_json_safe(path: Path) -> dict | None:
    import json
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
