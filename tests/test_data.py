import random

import pytest
from pydantic import ValidationError

from prefagent.data import gsm8k
from prefagent.data.hh_rlhf import normalize, parse_single_turn
from prefagent.data.prepare import stratified_take
from prefagent.data.schema import PreferencePair


# --- HH-RLHF parsing --------------------------------------------------------

def hh(prompt: str, reply: str) -> str:
    return f"\n\nHuman: {prompt}\n\nAssistant: {reply}"


def test_single_turn_parsed():
    ex = parse_single_turn(hh("Why is the sky blue?", "Rayleigh scattering."),
                           hh("Why is the sky blue?", "No idea."))
    assert ex.prompt == "Why is the sky blue?"
    assert ex.chosen == "Rayleigh scattering."
    assert ex.rejected == "No idea."


def test_multi_turn_rejected():
    two_turns = hh("Hi", "Hello!") + hh("Tell me a joke", "Knock knock.")
    assert parse_single_turn(two_turns, two_turns.replace("Knock", "Bang")) is None


def test_mismatched_prefix_rejected():
    assert parse_single_turn(hh("A?", "x"), hh("B?", "y")) is None


def test_identical_or_empty_replies_rejected():
    assert parse_single_turn(hh("A?", "same"), hh("A?", "same")) is None
    assert parse_single_turn(hh("A?", ""), hh("A?", "y")) is None


def test_normalize_collapses_trivial_variants():
    assert normalize("  What's   UP? ") == normalize("whats up")


# --- splitting --------------------------------------------------------------

def test_stratified_take_balanced_and_disjoint():
    pools = {s: [{"id": f"{s}-{i}"} for i in range(10)] for s in ("a", "b")}
    rng = random.Random(0)
    first = stratified_take(pools, 6, rng)
    second = stratified_take(pools, 6, rng, exclude={r["id"] for r in first})
    assert sum(r["id"].startswith("a") for r in first) == 3
    assert not {r["id"] for r in first} & {r["id"] for r in second}


def test_stratified_take_fails_loudly_when_short():
    with pytest.raises(ValueError):
        stratified_take({"a": [{"id": "1"}], "b": [{"id": "2"}]}, 4, random.Random(0))


# --- GSM8K ------------------------------------------------------------------

def test_gold_answer():
    assert gsm8k.gold_answer("She has 3 + 4 = 7 apples.\n#### 7") == "7"
    assert gsm8k.gold_answer("...\n#### 1,200") == "1200"


@pytest.mark.parametrize("text,expected", [
    ("So the total is 42.", "42"),
    ("The answer is $1,250.", "1250"),
    ("First 3 then 5, so \\boxed{8}", "8"),
    ("#### 12.50", "12.5"),
    ("I cannot solve this.", None),
])
def test_extract_answer(text, expected):
    assert gsm8k.extract_answer(text) == expected


def test_is_correct_uses_final_answer_not_intermediate():
    assert gsm8k.is_correct("3 + 4 = 7, times 2 is 14. The answer is 14.", "14")
    assert not gsm8k.is_correct("3 + 4 = 7. The answer is 7.", "14")


# --- schema -----------------------------------------------------------------

def test_pair_rejects_identical_responses():
    with pytest.raises(ValidationError):
        PreferencePair(prompt_id="p", prompt="q", chosen="same", rejected=" same ",
                       condition="human")


def test_pair_agreement_bounds():
    with pytest.raises(ValidationError):
        PreferencePair(prompt_id="p", prompt="q", chosen="a", rejected="b",
                       condition="multi_panel", agreement=1.5)
