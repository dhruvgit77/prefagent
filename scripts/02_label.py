"""Stage 2: agent pipeline — generate candidates, judge them, build preference pairs.

    python scripts/02_label.py budget                      # API calls/days per model, no calls
    python scripts/02_label.py pilot --limit 20            # small end-to-end run + spot-check
    python scripts/02_label.py run                         # the full labelling plan (resumable)
    python scripts/02_label.py generate --split train --source multi_persona --limit 50
    python scripts/02_label.py judge --split train --source multi_persona --method panel
    python scripts/02_label.py pairs --split train         # build data/pairs/train/*.jsonl
    python scripts/02_label.py usage                       # tokens spent so far, per model

Config overrides go after `--set`, e.g. `--set generation.temperature=0.7`.
"""

import argparse
import json
import logging
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefagent.agents import pipeline  # noqa: E402
from prefagent.config import load_config, save_snapshot  # noqa: E402
from prefagent.llm.client import LLMClient  # noqa: E402
from prefagent.utils.io import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["budget", "pilot", "run", "generate", "judge",
                                        "pairs", "usage"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--source", default="multi_persona")
    ap.add_argument("--method", default="panel", choices=pipeline.JUDGED)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", default=None, help="e.g. demo (configs/profiles/demo.yaml)")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides, profile=args.profile)
    logging.basicConfig(level=cfg["logging"]["level"], format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "budget":
        print_budget(cfg, pipeline.label_plan(cfg))
        return
    if args.command == "pairs":
        print(json.dumps(pipeline.build_pairs(cfg, args.split, force=args.force), indent=2))
        return

    client = LLMClient(cfg)
    if args.command == "usage":
        print(json.dumps(client.cache.usage_by_model(), indent=2))
        return

    save_snapshot(cfg, Path(cfg["paths"]["data_dir"]) / "candidates")
    if args.command == "generate":
        pipeline.run_generate(cfg, client, args.split, args.source, args.limit, args.workers)
    elif args.command == "judge":
        pipeline.run_judge(cfg, client, args.split, args.source, args.method, args.limit,
                           args.workers)
    elif args.command == "pilot":
        limit = args.limit or 20
        pipeline.run_generate(cfg, client, "train", "multi_persona", limit, args.workers)
        pipeline.run_judge(cfg, client, "train", "multi_persona", "panel", limit, args.workers)
        spot_check(cfg, limit)
    elif args.command == "run":
        for step in pipeline.label_plan(cfg):
            n = min(step["n"], args.limit) if args.limit else step["n"]
            if step["method"] is None:
                if step["source"] != "human_pair":
                    pipeline.run_generate(cfg, client, step["split"], step["source"], n,
                                          args.workers)
            else:
                pipeline.run_judge(cfg, client, step["split"], step["source"],
                                   step["method"], n, args.workers)
    print(json.dumps(client.cache.usage_by_model(), indent=2))


def print_budget(cfg: dict, plan: list[dict]) -> None:
    print("Labelling plan:")
    for s in plan:
        print(f"  {s['split']:>16}  {s['source']:<15} {s['method'] or 'generate':<13} "
              f"{s['n']:>5} prompts")
    print("\nAPI calls per model (upper bound, excluding parse retries):")
    for model, b in pipeline.budget(cfg, plan).items():
        days = f"{b['days']} days at {b['rpd']}/day" if b["days"] else "no rpd set"
        print(f"  {model:<12} {b['calls']:>7} calls   ≈ {days}")


def spot_check(cfg: dict, limit: int, show: int = 3) -> None:
    """The implementation guide's 'manually spot-check quality before scaling up'."""
    jpath = pipeline.judgements_path(cfg, "train", "multi_persona", "panel")
    cands = {r["prompt_id"]: r for r in read_jsonl(
        pipeline.candidates_path(cfg, "train", "multi_persona"))}
    judged = read_jsonl(jpath)[:limit]
    ok = [j for j in judged if j["aggregate"]]
    print(f"\n{len(ok)}/{len(judged)} prompts produced a pair; "
          f"failed judge calls: {sum(j['failed_calls'] for j in judged)}")
    if ok:
        agree = sum(j["aggregate"]["agreement"] for j in ok) / len(ok)
        print(f"mean panel agreement: {agree:.2f}")
    for j in random.Random(0).sample(ok, min(show, len(ok))):
        texts = [c["text"] for c in cands[j["prompt_id"]]["candidates"]]
        agg = j["aggregate"]
        print("\n" + "=" * 80)
        print(f"{j['prompt_id']}  scores={[round(s, 2) for s in agg['mean_scores']]}  "
              f"agreement={agg['agreement']:.2f}")
        for tag, idx in (("CHOSEN", agg["chosen"]), ("REJECTED", agg["rejected"])):
            print(f"--- {tag} (candidate {idx}):")
            print(textwrap.shorten(texts[idx], 600))


if __name__ == "__main__":
    main()
