"""Agent-pipeline tests against a fake LLM: no network, no quota.

The fake judge scores a response 5 if it contains "GOOD", 1 if it contains "BAD", else 3.
Because judges see candidates in a shuffled order, these tests check that every verdict
is mapped back to the right ORIGINAL candidate — the easiest bug to make here.
"""

import json
import random
import re

import pytest

from prefagent.agents import label, pipeline
from prefagent.agents.generate import generate
from prefagent.agents.judge import Judge, Verdict, aggregate
from prefagent.config import load_config
from prefagent.llm.client import ChatResponse
from prefagent.llm.parsing import extract_json
from prefagent.utils.io import read_jsonl, write_jsonl

RESPONSE_RE = re.compile(r"## Response ([A-Z])\n(.*?)(?=\n\n## |\Z)", re.S)


class FakeClient:
    def __init__(self, garbage_first: bool = False):
        self.calls = []
        self.garbage_first = garbage_first

    def chat(self, req):
        self.calls.append(req)
        system = req.messages[0]["content"]
        user = req.messages[-1]["content"]
        if "impartial expert judge" not in system:          # generation call
            return ChatResponse(text=f"answer from [{system[:20]}] #{req.sample_idx}",
                                model=req.model, cached=False, finish_reason="stop")
        if self.garbage_first and req.attempt == 0:
            return ChatResponse(text="Sure! Here is my verdict...", model=req.model,
                                cached=False, finish_reason="stop")
        shown = dict(RESPONSE_RE.findall(user))
        score = {k: 5 if "GOOD" in v else 1 if "BAD" in v else 3 for k, v in shown.items()}
        best = max(score, key=score.get)
        worst = min(score, key=score.get)
        if best == worst:
            worst = next(k for k in score if k != best)
        body = {"scores": {k: {"helpfulness": s, "correctness": s, "harmlessness": 5}
                           for k, s in score.items()},
                "best": best, "worst": worst, "rationale": "fake"}
        return ChatResponse(text=f"```json\n{json.dumps(body)}\n```", model=req.model,
                            cached=False, finish_reason="stop")


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides=[f"paths.data_dir={tmp_path}"])


CANDS = ["meh one", "a GOOD answer", "meh two", "a BAD answer"]


# --- judging ------------------------------------------------------------------

def test_single_judge_maps_shuffled_labels_back(cfg):
    client = FakeClient()
    res = Judge(client, cfg).single_judge("p1", "q?", CANDS)
    assert len(client.calls) == 6                       # compute: 6 calls
    perms = {tuple(c["permutation"]) for c in res["calls"]}
    assert len(perms) > 1                               # orders really were shuffled
    assert all(c["best"] == 1 and c["worst"] == 3 for c in res["calls"])
    assert res["aggregate"]["chosen"] == 1 and res["aggregate"]["rejected"] == 3
    assert res["aggregate"]["agreement"] == 1.0


def test_panel_is_compute_matched_and_debates(cfg):
    client = FakeClient()
    res = Judge(client, cfg).panel("p1", "q?", CANDS)
    assert len(client.calls) == 6                       # 3 judges × 2 rounds
    r2 = [c for c in client.calls if "Round-1 verdicts" in c.messages[-1]["content"]]
    assert len(r2) == 3
    assert "(you)" in r2[0].messages[-1]["content"]     # each judge sees its own view
    assert len(res["final"]) == 3 and all(res["calls"][i]["round"] == 2 for i in res["final"])
    assert res["aggregate"]["chosen"] == 1


def test_peer_views_are_relabelled_into_current_order(cfg):
    client = FakeClient()
    Judge(client, cfg).panel("p1", "q?", CANDS)
    for req in client.calls[3:]:
        user = req.messages[-1]["content"]
        good_label = next(k for k, v in RESPONSE_RE.findall(user) if "GOOD" in v)
        # every peer view must name the GOOD response (in THIS call's labels) as best
        assert user.count(f"best={good_label}") == 3


def test_devils_advocate_targets_lowest_rated(cfg):
    cfg["judging"]["panel"]["devils_advocate"] = True
    client = FakeClient()
    Judge(client, cfg).panel("p1", "q?", CANDS)
    user = client.calls[3].messages[-1]["content"]
    bad_label = next(k for k, v in RESPONSE_RE.findall(user) if "BAD" in v)
    assert f"case FOR Response {bad_label}" in user
    assert "case FOR" not in client.calls[4].messages[-1]["content"]


def test_unparseable_reply_is_retried_with_new_cache_key(cfg):
    client = FakeClient(garbage_first=True)
    res = Judge(client, cfg).single_judge("p1", "q?", CANDS)
    assert all(c["attempts"] == 2 for c in res["calls"])
    assert len(client.calls) == 12


def test_aggregate_tie_gives_no_pair():
    v = Verdict("j", 2, 0, [0, 1], [{"h": 3}, {"h": 3}], 0, 1, "", 1)
    assert aggregate([v], 2, random.Random(0)) is None


def test_aggregate_agreement_counts_dissent():
    up = Verdict("j", 2, 0, [0, 1], [{"h": 5}, {"h": 1}], 0, 1, "", 1)
    down = Verdict("k", 2, 0, [0, 1], [{"h": 2}, {"h": 3}], 1, 0, "", 1)
    agg = aggregate([up, up, down], 2, random.Random(0))
    assert agg["chosen"] == 0 and agg["agreement"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("text", [
    '{"a": 1}', 'Here you go:\n```json\n{"a": 1}\n```', 'prefix {"a": 1} suffix'])
def test_extract_json(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_failure():
    assert extract_json("no json here") is None


# --- generation ---------------------------------------------------------------

def test_multi_persona_uses_each_persona_once(cfg):
    client = FakeClient()
    out = generate(client, cfg, {"id": "p", "prompt": "q"}, "multi_persona")
    systems = [c.messages[0]["content"] for c in client.calls]
    assert systems == cfg["generation"]["multi_persona"]
    assert len(out["candidates"]) == 4


def test_single_persona_same_prompt_different_samples(cfg):
    client = FakeClient()
    generate(client, cfg, {"id": "p", "prompt": "q"}, "single_persona")
    assert len({c.messages[0]["content"] for c in client.calls}) == 1
    assert [c.sample_idx for c in client.calls] == [0, 1, 2, 3]
    assert {c.temperature for c in client.calls} == {cfg["generation"]["temperature"]}


def test_truncated_candidates_dropped(cfg):
    class Truncating(FakeClient):
        def chat(self, req):
            r = super().chat(req)
            if req.sample_idx == 0:
                r.finish_reason = "length"
            return r
    out = generate(Truncating(), cfg, {"id": "p", "prompt": "q"}, "single_persona")
    assert out["dropped"] == 1 and len(out["candidates"]) == 3


# --- labels and pairs ---------------------------------------------------------

def test_longer_baseline_and_random_baseline():
    rec = {"id": "p", "prompt": "q"}
    cands = {"candidates": [{"text": "a b"}, {"text": "a b c d e"}, {"text": "a"}]}
    p = label.longer_pair(rec, cands, "length")
    assert (p.chosen, p.rejected) == ("a b c d e", "a")
    r1 = label.random_pair(rec, cands, "random", seed=1)
    r2 = label.random_pair(rec, cands, "random", seed=1)
    assert (r1.chosen, r1.rejected) == (r2.chosen, r2.rejected)     # seeded


def test_build_pairs_uses_common_prompts_and_guards_changes(cfg, tmp_path):
    prompts = [{"id": f"p{i}", "prompt": f"q{i}", "subset": "helpful-base", "split": "train",
                "prompt_tokens": 2, "human_chosen": f"good {i}", "human_rejected": f"bad {i}"}
               for i in range(4)]
    write_jsonl(tmp_path / "prompts" / "train.jsonl", prompts)
    for src in ("single_persona", "multi_persona"):
        write_jsonl(pipeline.candidates_path(cfg, "train", src),
                    [{"prompt_id": p["id"], "source": src, "dropped": 0,
                      "candidates": [{"text": f"x {p['id']}"}, {"text": f"y y {p['id']}"}]}
                     for p in prompts])
        for method in ("single_judge", "panel"):
            rows = []
            for i, p in enumerate(prompts):
                # p2 is a judge tie under multi_persona/panel only → must vanish everywhere
                tie = src == "multi_persona" and method == "panel" and i == 2
                agg = None if tie else {"chosen": 1, "rejected": 0, "agreement": 1.0,
                                        "mean_scores": [1, 5], "n_final": 3}
                rows.append({"prompt_id": p["id"], "method": method, "aggregate": agg})
            write_jsonl(pipeline.judgements_path(cfg, "train", src, method), rows)

    manifest = pipeline.build_pairs(cfg, "train")
    main = manifest["groups"]["main"]
    assert "onpolicy_panel" not in main["conditions"]   # no policy samples yet
    assert main["n_prompts"] == 3
    for cond in main["conditions"]:
        ids = [r["prompt_id"] for r in read_jsonl(tmp_path / "pairs" / "train" / f"{cond}.jsonl")]
        assert ids == ["p0", "p1", "p3"]

    cfg["conditions"].pop("random")
    with pytest.raises(RuntimeError, match="Condition set changed"):
        pipeline.build_pairs(cfg, "train")


def test_budget_counts_compute_matched_calls(cfg):
    plan = [{"split": "train", "source": "multi_persona", "method": m, "n": 10}
            for m in (None, "single_judge", "panel")]
    b = pipeline.budget(cfg, plan)
    assert b["gptoss120b"]["calls"] == 10 * 6 + 10 * 2        # single judge + panel seat
    assert b["llama70b"]["calls"] == 10 * 4 + 10 * 2          # generation + panel seat
