"""Build every prompt split once, with a fixed seed, and write them to data/prompts/.

Design:
  * eval + label_agreement come from HH-RLHF's official *test* split; train + dev from
    its *train* split. Separate source splits are the first line of defence against leakage.
  * Within each pool, prompts are deduplicated (HH-RLHF repeats prompts).
  * Train/dev candidates that are near-duplicates (embedding cosine ≥ threshold) of any
    eval/label prompt are dropped — the second line of defence.
  * helpful-base and harmless-base are sampled 50/50 so results can be reported per subset.
  * Every filtering step is counted in stats.json, which becomes the data section's table.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np

from prefagent.data import gsm8k
from prefagent.data.hh_rlhf import normalize, parse_single_turn
from prefagent.data.schema import PromptRecord
from prefagent.utils.io import write_json, write_jsonl

log = logging.getLogger(__name__)

# How many extra candidates to embed for decontamination, relative to what we need.
# Embedding the whole train split (~80k rows) on CPU is slow and unnecessary.
OVERSAMPLE = 3


def load_hh_pool(cfg: dict, subset: str, hf_split: str, tokenizer, stats: Counter) -> list[dict]:
    """Parse, filter and dedupe one (subset, split) of HH-RLHF."""
    from datasets import load_dataset

    hh = cfg["hh_rlhf"]
    tag = f"{subset}/{hf_split}"
    log.info("loading HH-RLHF %s", tag)
    ds = load_dataset(hh["dataset"], data_dir=subset, split=hf_split)
    stats[f"{tag}:raw"] = len(ds)

    rows = []
    for i, row in enumerate(ds):
        ex = parse_single_turn(row["chosen"], row["rejected"])
        if ex is None:
            continue
        rows.append({"id": f"hh-{subset}-{hf_split}-{i:06d}", "prompt": ex.prompt,
                     "human_chosen": ex.chosen, "human_rejected": ex.rejected,
                     "subset": subset})
    stats[f"{tag}:single_turn_valid"] = len(rows)

    lengths = tokenizer([r["prompt"] for r in rows], add_special_tokens=False)["input_ids"]
    lo, hi = hh["min_prompt_tokens"], hh["max_prompt_tokens"]
    for r, ids in zip(rows, lengths):
        r["prompt_tokens"] = len(ids)
    rows = [r for r in rows if lo <= r["prompt_tokens"] <= hi]
    stats[f"{tag}:length_ok"] = len(rows)

    seen, unique = set(), []
    for r in rows:
        key = normalize(r["prompt"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    stats[f"{tag}:deduped"] = len(unique)
    log.info("%s: %d raw -> %d single-turn -> %d length-ok -> %d unique", tag, len(ds),
             stats[f"{tag}:single_turn_valid"], len(rows), len(unique))
    return unique


def stratified_take(pools: dict[str, list[dict]], n: int, rng: random.Random,
                    exclude: set[str] = frozenset()) -> list[dict]:
    """Take n rows split as evenly as possible across pools, skipping excluded ids.
    Pools are shuffled in place once by the caller, so repeated calls draw disjoint rows."""
    names = sorted(pools)
    quotas = {name: n // len(names) for name in names}
    for name in names[: n % len(names)]:
        quotas[name] += 1
    out = []
    for name in names:
        avail = [r for r in pools[name] if r["id"] not in exclude]
        if len(avail) < quotas[name]:
            raise ValueError(f"{name}: need {quotas[name]}, only {len(avail)} available")
        out.extend(avail[: quotas[name]])
    rng.shuffle(out)
    return out


def near_duplicates(candidates: list[dict], references: list[dict], embedder_name: str,
                    threshold: float) -> set[str]:
    """IDs of candidates whose prompt has cosine ≥ threshold to any reference prompt."""
    from sentence_transformers import SentenceTransformer

    log.info("decontamination: embedding %d candidates vs %d held-out prompts",
             len(candidates), len(references))
    model = SentenceTransformer(embedder_name, device="cpu")
    enc = lambda rows: model.encode([r["prompt"] for r in rows], batch_size=128,
                                    normalize_embeddings=True, show_progress_bar=False)
    sims = enc(candidates) @ enc(references).T          # cosine, since rows are unit-norm
    flagged = np.where(sims.max(axis=1) >= threshold)[0]
    return {candidates[i]["id"] for i in flagged}


def build_gsm8k(cfg: dict, tokenizer, rng: random.Random, stats: Counter) -> list[dict]:
    """Label-accuracy problems from GSM8K *train*; lm-eval's GSM8K benchmark uses *test*,
    so the two never overlap."""
    from datasets import load_dataset

    g = cfg["gsm8k"]
    log.info("loading GSM8K")
    ds = load_dataset(g["dataset"], g["config"], split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[: cfg["splits"]["gsm8k_label_problems"]]:
        q = ds[i]["question"]
        out.append({"id": f"gsm8k-train-{i:05d}", "prompt": q, "subset": "gsm8k",
                    "split": "gsm8k_label", "gold_answer": gsm8k.gold_answer(ds[i]["answer"]),
                    "prompt_tokens": len(tokenizer(q, add_special_tokens=False)["input_ids"])})
    stats["gsm8k:selected"] = len(out)
    return out


def prepare(cfg: dict) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["policy_tokenizer"])
    rng = random.Random(cfg["seeds"]["data"])
    stats: Counter = Counter()
    sp = cfg["splits"]
    subsets = cfg["hh_rlhf"]["subsets"]

    test_pools = {s: load_hh_pool(cfg, s, "test", tokenizer, stats) for s in subsets}
    train_pools = {s: load_hh_pool(cfg, s, "train", tokenizer, stats) for s in subsets}
    for pool in (*test_pools.values(), *train_pools.values()):
        rng.shuffle(pool)

    # 1) Held-out sets first, from the test split, disjoint from each other.
    eval_rows = stratified_take(test_pools, sp["eval_prompts"], rng)
    label_rows = stratified_take(test_pools, sp["label_agreement_pairs"], rng,
                                 exclude={r["id"] for r in eval_rows})
    held_out = eval_rows + label_rows

    # 2) Decontaminate an oversampled slice of the train pools against the held-out sets.
    need = sp["train_prompts"] + sp["dev_pairs"]
    per_pool = OVERSAMPLE * need // len(subsets)
    candidates = [r for s in subsets for r in train_pools[s][:per_pool]]
    dec = cfg["decontamination"]
    leaked = near_duplicates(candidates, held_out, dec["embedder"], dec["near_dup_cosine"])
    exact = {normalize(r["prompt"]) for r in held_out}
    leaked |= {r["id"] for r in candidates if normalize(r["prompt"]) in exact}
    stats["decontam:candidates"] = len(candidates)
    stats["decontam:removed"] = len(leaked)
    clean_pools = {s: [r for r in train_pools[s][:per_pool] if r["id"] not in leaked]
                   for s in subsets}

    # 3) Train and dev from the clean train pools, disjoint.
    train_rows = stratified_take(clean_pools, sp["train_prompts"], rng)
    dev_rows = stratified_take(clean_pools, sp["dev_pairs"], rng,
                               exclude={r["id"] for r in train_rows})

    out_dir = Path(cfg["paths"]["data_dir"]) / "prompts"
    counts = {}
    for split, rows in [("train", train_rows), ("dev", dev_rows), ("eval", eval_rows),
                        ("label_agreement", label_rows)]:
        records = [PromptRecord(split=split, **r).model_dump() for r in rows]
        counts[split] = write_jsonl(out_dir / f"{split}.jsonl", records)
        stats.update({f"final:{split}:{k}": v
                      for k, v in Counter(r["subset"] for r in rows).items()})

    if cfg["gsm8k"]["enabled"]:
        records = [PromptRecord(**r).model_dump() for r in build_gsm8k(cfg, tokenizer, rng, stats)]
        counts["gsm8k_label"] = write_jsonl(out_dir / "gsm8k_label.jsonl", records)

    write_json(out_dir / "stats.json", {"seed": cfg["seeds"]["data"], "counts": counts,
                                        "filtering": dict(sorted(stats.items()))})
    return counts
