"""Stage 3: DPO-train a LoRA adapter for one condition.

    python scripts/03_train.py --condition human --profile demo
    python scripts/03_train.py --condition multi_panel --model Qwen/Qwen2.5-1.5B-Instruct --seed 1

Writes outputs/runs/<condition>__<model>__n<N>__s<seed>/ with the adapter, the training
log (rewards/margins, rewards/accuracies, logps) and a config snapshot.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefagent.config import load_config  # noqa: E402
from prefagent.train.dpo import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--model", default=None, help="default: policies.primary")
    ap.add_argument("--seed", type=int, default=None, help="default: first of seeds.train")
    ap.add_argument("--n-pairs", type=int, default=None, help="default: data_sizes.main")
    ap.add_argument("--scaling", action="store_true", help="use the .scaling pairs file")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides, profile=args.profile)
    logging.basicConfig(level=cfg["logging"]["level"], format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    adapter = train(cfg, args.condition,
                    model_name=args.model or cfg["policies"]["primary"],
                    seed=args.seed if args.seed is not None else cfg["seeds"]["train"][0],
                    n_pairs=args.n_pairs or default_n_pairs(cfg, args.scaling),
                    scaling=args.scaling)
    print(f"adapter saved to {adapter}")


def default_n_pairs(cfg: dict, scaling: bool) -> int:
    """data_sizes.main, capped by how many common prompts survived labelling (judge ties
    and truncated candidates remove a few), so every condition uses the same count."""
    manifest = Path(cfg["paths"]["data_dir"]) / "pairs" / "train" / "manifest.json"
    group = "scaling" if scaling else "main"
    available = json.loads(manifest.read_text())["groups"][group]["n_prompts"]
    want = max(cfg["data_sizes"]["scaling"]) if scaling else cfg["data_sizes"]["main"]
    return min(want, available)


if __name__ == "__main__":
    main()
