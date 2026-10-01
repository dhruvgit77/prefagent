"""Stage 1: build all prompt splits (train / dev / eval / label_agreement / gsm8k_label).

    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py splits.train_prompts=500   # quick pilot

Runs locally on CPU in a few minutes. Outputs data/prompts/*.jsonl and stats.json.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefagent.config import load_config, save_snapshot  # noqa: E402
from prefagent.data.prepare import prepare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="config overrides, key.path=value")
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    logging.basicConfig(level=cfg["logging"]["level"], format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-request HF logs
    counts = prepare(cfg)
    save_snapshot(cfg, Path(cfg["paths"]["data_dir"]) / "prompts")
    for split, n in counts.items():
        print(f"{split:>16}: {n}")


if __name__ == "__main__":
    main()
