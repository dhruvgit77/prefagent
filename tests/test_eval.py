"""Evaluation logic tests (no network): win-rate order mapping and statistics."""

import json

import pytest

from prefagent.config import load_config
from prefagent.eval import stats
from prefagent.eval.winrate import compare, juror_verdict
from prefagent.llm.client import ChatResponse


class Juror:
    """Prefers the response containing GOOD; optionally always says 'A' (position bias)."""

    def __init__(self, always_a: bool = False):
        self.always_a = always_a

    def chat(self, req):
        user = req.messages[-1]["content"]
        a = user.split("## Response A\n")[1].split("\n\n## Response B")[0]
        b = user.split("## Response B\n")[1]
        if self.always_a:
            w = "A"
        else:
            w = "A" if "GOOD" in a else "B" if "GOOD" in b else "tie"
        return ChatResponse(text=json.dumps({"winner": w}), model=req.model, cached=False)


def test_consistent_juror_maps_back_to_the_right_model():
    assert juror_verdict(Juror(), "j", "sys", "q", "GOOD x", "y", 0.0)["winner"] == "x"
    assert juror_verdict(Juror(), "j", "sys", "q", "x", "GOOD y", 0.0)["winner"] == "y"


def test_position_biased_juror_counts_as_tie():
    # Always picking "A" means it picks x in one order and y in the other → tie.
    assert juror_verdict(Juror(always_a=True), "j", "sys", "q", "GOOD x", "y", 0.0)["winner"] == "tie"


def test_jury_needs_strict_majority():
    cfg = load_config(profile="demo")

    class Split:
        def chat(self, req):
            biased = req.model == "command"
            return (Juror(always_a=True) if biased else Juror()).chat(req)

    # gemini says x, command (position-biased) says tie → 1 of 2 is not a majority → tie
    assert compare(Split(), cfg, "q", "GOOD x", "y")["outcome"] == 0.5
    assert compare(Juror(), cfg, "q", "GOOD x", "y")["outcome"] == 1.0
    assert compare(Juror(), cfg, "q", "x", "GOOD y")["outcome"] == 0.0


def test_bootstrap_ci_brackets_mean():
    vals = [1.0] * 60 + [0.0] * 40
    lo, hi = stats.bootstrap_ci(vals)
    assert lo < 0.6 < hi and hi - lo < 0.25


def test_length_controlled_removes_pure_length_effect():
    # The longer answer wins 80% of the time whichever model wrote it, so once length is
    # controlled neither model is better: the expected win-rate is ~50%.
    outcomes, lx, ly = [], [], []
    for i in range(200):
        longer = i % 2 == 0
        lx.append(300 if longer else 100 + i % 7)
        ly.append(100 + i % 5 if longer else 300)
        x_wins = longer if (i // 2) % 5 else not longer
        outcomes.append(1.0 if x_wins else 0.0)
    raw = sum(outcomes) / len(outcomes)
    assert raw == pytest.approx(0.5)
    lc = stats.length_controlled_win_rate(outcomes, lx, ly)
    assert lc is not None and 0.35 < lc < 0.65
