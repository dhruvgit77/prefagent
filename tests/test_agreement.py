"""Label-agreement summary (no network)."""

from prefagent.eval.agreement import summarize


def judgement(pid, chosen, agreement=1.0, calls=()):
    agg = None if chosen is None else {"chosen": chosen, "rejected": 1 - chosen,
                                       "agreement": agreement}
    return {"prompt_id": pid, "aggregate": agg, "calls": list(calls)}


def call(judge, rnd, best, permutation):
    return {"judge": judge, "round": rnd, "best": best, "permutation": permutation}


def test_agreement_counts_ties_as_half_and_splits_by_subset_and_confidence():
    js = [judgement("a", 0), judgement("b", 1, agreement=2 / 3), judgement("c", None)]
    r = summarize(js, {"a": "helpful", "b": "harmless", "c": "helpful"})
    assert r["agreement_with_humans"]["rate"] == 0.5
    assert r["no_preference"] == 1
    assert r["by_subset"]["helpful"]["rate"] == 0.75
    assert r["by_panel_confidence"] == {
        "split": {"n": 1, "rate": 0.0, "ci95": [0.0, 0.0]},
        "unanimous": {"n": 1, "rate": 1.0, "ci95": [1.0, 1.0]},
    }


def test_per_judge_rounds_and_position_bias_use_original_indices():
    calls = [call("x", 1, 0, [1, 0]), call("x", 2, 1, [0, 1])]
    r = summarize([judgement("a", 0, calls=calls)], {})
    assert r["per_judge_round"]["x r1"]["rate"] == 1.0
    assert r["per_judge_round"]["x r2"]["rate"] == 0.0
    # r1 picked the second shown response, r2 the second shown again → 0% first-shown
    assert r["picked_first_shown_rate"]["rate"] == 0.0
