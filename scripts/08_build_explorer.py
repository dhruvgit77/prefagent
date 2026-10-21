"""Build the interactive project explorer from the real output files.

    python scripts/08_build_explorer.py
    open outputs/explorer/index.html

Reads data/prompts, data/candidates, data/judgements, outputs/explain/reward_model.json
and outputs/label_agreement (computed here if missing). Writes one self-contained HTML
file that works offline. Every number on the page comes from those files. The "try it"
scorer is a text-features-only pairwise model fitted here, per HH-RLHF subset.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefagent.agents import pipeline  # noqa: E402
from prefagent.config import load_config  # noqa: E402
from prefagent.eval.agreement import summarize  # noqa: E402
from prefagent.explain.reward import (  # noqa: E402
    HEDGE_TERMS, REFUSAL_TERMS, TEXT_FEATURES, PairwiseRewardModel, preference_pairs,
    text_features)
from prefagent.utils.io import read_jsonl  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "templates" / "explorer.html"
PERSONAS = ["Concise", "Step-by-step", "Domain expert", "Careful sceptic"]


def _by_id(path: Path, key: str = "prompt_id") -> dict[str, dict]:
    return {r[key]: r for r in read_jsonl(path)} if path.exists() else {}


def walkthrough(cfg: dict) -> list[dict]:
    prompts = _by_id(pipeline.prompts_path(cfg, "train"), "id")
    cands = _by_id(pipeline.candidates_path(cfg, "train", "multi_persona"))
    out = []
    for j in read_jsonl(pipeline.judgements_path(cfg, "train", "multi_persona", "panel")):
        c = cands.get(j["prompt_id"])
        if c is None:
            continue
        out.append({
            "id": j["prompt_id"],
            "subset": prompts[j["prompt_id"]]["subset"],
            "prompt": prompts[j["prompt_id"]]["prompt"],
            "candidates": [{"persona": PERSONAS[x["persona"]] if isinstance(x["persona"], int)
                            else x["persona"], "text": x["text"]} for x in c["candidates"]],
            "dropped": c.get("dropped", 0),
            "calls": [{k: call[k] for k in ("judge", "round", "scores", "best", "worst",
                                            "rationale")} for call in j["calls"]],
            "aggregate": j["aggregate"],
        })
    return out


def agreement(cfg: dict) -> dict | None:
    path = pipeline.judgements_path(cfg, "label_agreement", "human_pair", "panel")
    if not path.exists():
        return None
    prompts = _by_id(pipeline.prompts_path(cfg, "label_agreement"), "id")
    judged = read_jsonl(path)
    summary = summarize(judged, {k: v["subset"] for k, v in prompts.items()})
    rows = []
    for j in judged:
        p = prompts[j["prompt_id"]]
        agg = j["aggregate"]
        rows.append({
            "id": j["prompt_id"], "subset": p["subset"], "prompt": p["prompt"],
            "human_chosen": p["human_chosen"], "human_rejected": p["human_rejected"],
            "panel_pick": None if agg is None else agg["chosen"],
            "panel_agreement": None if agg is None else agg["agreement"],
            "scores": None if agg is None else agg["mean_scores"],
            "rationales": [{"judge": c["judge"], "round": c["round"], "best": c["best"],
                            "rationale": c["rationale"]} for c in j["calls"]],
        })
    return {"summary": summary, "rows": rows}


def scorer(cfg: dict) -> dict:
    """Text-only pairwise models (pooled and per subset) the page can run in JS."""
    rows = read_jsonl(pipeline.prompts_path(cfg, "train"))
    models = {}
    for name in ["all"] + sorted({r["subset"] for r in rows}):
        sub = [r for r in rows if name == "all" or r["subset"] == name]
        pairs = preference_pairs(sub)
        chosen = text_features([p["chosen"] for p in pairs])
        rejected = text_features([p["rejected"] for p in pairs])
        m = PairwiseRewardModel(seed=0).fit(chosen, rejected)
        acc = float(np.mean(m.margins(chosen, rejected) > 0))
        models[name] = {"coef": m.coefficients.tolist(), "scale": m.scaler.scale_.tolist(),
                        "n_pairs": len(pairs), "train_accuracy": acc}
    return {"features": list(TEXT_FEATURES), "hedge_terms": list(HEDGE_TERMS),
            "refusal_terms": list(REFUSAL_TERMS), "models": models}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("outputs/explorer/index.html"))
    ap.add_argument("--reward", type=Path, default=Path("outputs/explain/reward_model.json"))
    args = ap.parse_args()

    cfg = load_config()
    stats = json.loads(pipeline.prompts_path(cfg, "train").with_name("stats.json").read_text())
    data = {
        "stats": stats,
        "walkthrough": walkthrough(cfg),
        "agreement": agreement(cfg),
        "reward": json.loads(args.reward.read_text()) if args.reward.exists() else None,
        "scorer": scorer(cfg),
        "panel_judges": cfg["judging"]["panel"]["judges"],
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", blob)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"wrote {args.output} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
