"""Label quality: how often do agent judges agree with HH-RLHF human labels?

    python scripts/02_label.py judge --split label_agreement --source human_pair --limit 50
    python scripts/07_label_agreement.py

Reads data/judgements/label_agreement/human_pair__<method>.jsonl (candidate 0 is always
the human-chosen response; judges saw a shuffled order). No API calls. Reports, per
judging method: agreement with humans (bootstrap CI), by subset, by panel confidence,
per judge and round (before vs. after deliberation), and position bias. Writes
outputs/label_agreement/agreement.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefagent.agents import pipeline  # noqa: E402
from prefagent.config import load_config  # noqa: E402
from prefagent.eval.agreement import summarize  # noqa: E402
from prefagent.utils.io import read_jsonl, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/label_agreement/agreement.json"))
    args = ap.parse_args()

    cfg = load_config()
    subset_of = {r["id"]: r["subset"] for r in read_jsonl(
        pipeline.prompts_path(cfg, "label_agreement"))}
    report = {}
    for method in pipeline.JUDGED:
        path = pipeline.judgements_path(cfg, "label_agreement", "human_pair", method)
        if path.exists():
            report[method] = summarize(read_jsonl(path), subset_of)
    if not report:
        sys.exit("no label_agreement judgements found; run 02_label.py judge first")
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
