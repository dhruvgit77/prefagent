"""Configured lm-evaluation-harness integration (no model download required)."""

import json
import sys
from types import SimpleNamespace

import pytest

from prefagent.eval.benchmarks import (
    evaluate_halueval_qa,
    evaluate_task,
    halueval_qa_metrics,
    halueval_qa_prompt,
    load_halueval_qa,
    parse_halueval_judgement,
    select_halueval_qa_examples,
)


def test_evaluate_task_passes_adapter_task_options_and_safety_flag(monkeypatch, tmp_path):
    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return {"results": {"ifeval": {"score": 0.75}}}

    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=fake_evaluate))
    result = evaluate_task(
        "Qwen/model", tmp_path / "adapter", "ifeval", {"limit": 10}, "auto", True)

    assert result == {"ifeval": {"score": 0.75}}
    assert calls == [{
        "model": "hf",
        "model_args": {
            "pretrained": "Qwen/model",
            "dtype": "auto",
            "peft": str(tmp_path / "adapter"),
        },
        "tasks": ["ifeval"],
        "batch_size": "auto",
        "limit": 10,
        "apply_chat_template": True,
        "confirm_run_unsafe_code": True,
    }]


def test_evaluate_task_maps_per_subject_limit_for_task_groups(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "lm_eval",
        SimpleNamespace(simple_evaluate=lambda **kwargs: (
            calls.append(kwargs) or {"results": {"mmlu_subject_a": {"acc": 0.5}}})))

    result = evaluate_task(
        "Qwen/model", None, "mmlu", {"limit_per_subject": 20}, 4, False)

    assert calls[0]["limit"] == 20
    assert calls[0]["confirm_run_unsafe_code"] is False
    assert result == {"mmlu_subject_a": {"acc": 0.5}}


def test_evaluate_task_rejects_unsupported_or_conflicting_limits():
    with pytest.raises(ValueError, match="unsupported options"):
        evaluate_task("model", None, "task", {"other": 1}, 1, False)
    with pytest.raises(ValueError, match="both limit"):
        evaluate_task(
            "model", None, "task", {"limit": 10, "limit_per_subject": 2}, 1, False)
    with pytest.raises(ValueError, match="evaluate_halueval_qa"):
        evaluate_task("model", None, "halueval_qa", {"seed": 1234}, 1, False)


def test_halueval_subset_is_deterministic_balanced_and_unique():
    rows = [{
        "question": f"question {index}",
        "right_answer": f"correct {index}",
        "hallucinated_answer": f"hallucination {index}",
    } for index in range(10)]

    examples = select_halueval_qa_examples(rows, 8, 1234)

    assert examples == select_halueval_qa_examples(rows, 8, 1234)
    assert len({example["id"] for example in examples}) == 8
    assert sum(example["label"] == "yes" for example in examples) == 4
    for example in examples:
        expected = "hallucination" if example["label"] == "yes" else "correct"
        assert example["answer"].startswith(expected)


def test_halueval_subset_requires_enough_rows_and_valid_fields():
    with pytest.raises(ValueError, match="requested"):
        select_halueval_qa_examples([], 1, 0)
    with pytest.raises(ValueError, match="missing string fields"):
        select_halueval_qa_examples(
            [{"question": "q", "right_answer": "a", "hallucinated_answer": 1}], 1, 0)


def test_halueval_dataset_is_downloaded_once_and_read_from_cache(monkeypatch, tmp_path):
    rows = [{
        "question": f"question {index}",
        "right_answer": f"correct {index}",
        "hallucinated_answer": f"hallucination {index}",
    } for index in range(2)]
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return content

    monkeypatch.setattr(
        "prefagent.eval.benchmarks.urllib.request.urlopen",
        lambda request, timeout: (calls.append(request.full_url) or Response()))

    first = load_halueval_qa(tmp_path, 2, 1)
    second = load_halueval_qa(tmp_path, 2, 1)

    assert first == second
    assert len(calls) == 1


def test_halueval_prompt_and_judgement_parser_are_unambiguous():
    prompt = halueval_qa_prompt({
        "question": "Where is the office?",
        "answer": "It is in Delhi.",
    })
    assert "#Question#: Where is the office?" in prompt
    assert "#Answer#: It is in Delhi." in prompt
    assert parse_halueval_judgement(" YES.") == "yes"
    assert parse_halueval_judgement("No, this is factual.") == "no"
    assert parse_halueval_judgement("The answer is yes.") is None
    assert parse_halueval_judgement("Yes and no.") is None


def test_halueval_metrics_include_invalid_outputs_and_wilson_interval():
    examples = [{"label": label} for label in ("yes", "no", "yes", "no")]

    metrics = halueval_qa_metrics(examples, ["Yes", "No.", "unknown", "Yes"])

    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["valid_rate"] == 0.75
    assert metrics["invalid_predictions"] == 1
    assert metrics["accuracy_ci95_low"] < 0.5 < metrics["accuracy_ci95_high"]


def test_halueval_metrics_reject_mismatched_and_empty_inputs():
    with pytest.raises(ValueError, match="predictions"):
        halueval_qa_metrics([{"label": "yes"}], [])
    with pytest.raises(ValueError, match="empty"):
        halueval_qa_metrics([], [])


def test_halueval_eval_validates_task_options_before_model_load(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="unsupported HaluEval options"):
        evaluate_halueval_qa("model", None, {"limit": 1, "other": 1}, tmp_path, 1)
