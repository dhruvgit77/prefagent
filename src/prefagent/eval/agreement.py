"""Agreement between agent judges and HH-RLHF human labels (label-quality check)."""

from __future__ import annotations

from collections import defaultdict

from prefagent.eval.stats import bootstrap_ci

HUMAN_CHOSEN = 0   # human_pair candidates are stored [human_chosen, human_rejected]


def rate(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "rate": None, "ci95": None}
    lo, hi = bootstrap_ci(values)
    return {"n": len(values), "rate": sum(values) / len(values), "ci95": [lo, hi]}


def summarize(judgements: list[dict], subset_of: dict[str, str]) -> dict:
    # A pair the judges scored as a tie counts 0.5: it neither agrees nor disagrees.
    overall, by_subset, by_conf = [], defaultdict(list), defaultdict(list)
    calls_by = defaultdict(list)
    first_shown = []
    for j in judgements:
        agg = j["aggregate"]
        hit = 0.5 if agg is None else float(agg["chosen"] == HUMAN_CHOSEN)
        overall.append(hit)
        by_subset[subset_of.get(j["prompt_id"], "unknown")].append(hit)
        if agg is not None:
            by_conf["unanimous" if agg["agreement"] == 1.0 else "split"].append(hit)
        for c in j["calls"]:
            calls_by[(c["judge"], c["round"])].append(float(c["best"] == HUMAN_CHOSEN))
            first_shown.append(float(c["permutation"].index(c["best"]) == 0))
    return {
        "n_prompts": len(judgements),
        "no_preference": sum(j["aggregate"] is None for j in judgements),
        "agreement_with_humans": rate(overall),
        "by_subset": {k: rate(v) for k, v in sorted(by_subset.items())},
        "by_panel_confidence": {k: rate(v) for k, v in sorted(by_conf.items())},
        "per_judge_round": {f"{judge} r{rnd}": rate(v)
                            for (judge, rnd), v in sorted(calls_by.items())},
        "picked_first_shown_rate": rate(first_shown),
    }
