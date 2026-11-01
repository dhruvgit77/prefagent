"""Run configured lm-evaluation-harness and HaluEval tasks."""

from __future__ import annotations

import math
import random
import re
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prefagent.utils.io import read_jsonl

HALUEVAL_QA_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/"
    "b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json"
)


def evaluate_task(
    model_name: str,
    adapter: Path | None,
    task_name: str,
    task_options: dict[str, Any],
    batch_size: int | str,
    use_chat_template: bool,
) -> dict[str, dict[str, Any]]:
    """Evaluate one configured task (or task group) and return its metric results."""
    if task_name == "halueval_qa":
        raise ValueError("halueval_qa uses evaluate_halueval_qa, not lm-evaluation-harness")
    unsupported = set(task_options) - {"limit", "limit_per_subject"}
    if unsupported:
        raise ValueError(f"unsupported options for benchmark {task_name}: {sorted(unsupported)}")
    if "limit" in task_options and "limit_per_subject" in task_options:
        raise ValueError(f"{task_name} cannot set both limit and limit_per_subject")
    try:
        from lm_eval import simple_evaluate
    except ModuleNotFoundError as exc:
        if exc.name != "lm_eval":
            raise
        raise RuntimeError(
            "Benchmark evaluation requires lm-evaluation-harness; "
            "install the GPU environment with requirements-train.txt"
        ) from exc

    model_args = {"pretrained": model_name, "dtype": "auto"}
    if adapter is not None:
        model_args["peft"] = str(adapter)
    limit = task_options.get("limit", task_options.get("limit_per_subject"))
    result = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=[task_name],
        batch_size=batch_size,
        limit=limit,
        apply_chat_template=use_chat_template,
        confirm_run_unsafe_code=task_name == "ifeval",
    )
    task_results = result.get("results") if isinstance(result, Mapping) else None
    if not isinstance(task_results, Mapping) or not task_results:
        raise ValueError(f"lm-evaluation-harness returned no results for {task_name}")
    return {name: _jsonable(metrics) for name, metrics in task_results.items()}


def load_halueval_qa(cache_dir: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    """Load a stable subset of official HaluEval QA examples, cached locally."""
    if limit <= 0:
        raise ValueError(f"HaluEval limit must be positive, got {limit}")
    path = cache_dir / "halueval" / "qa_data.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            HALUEVAL_QA_URL, headers={"User-Agent": "prefagent-benchmark/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        if not content:
            raise ValueError(f"downloaded empty HaluEval QA dataset from {HALUEVAL_QA_URL}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

    records = read_jsonl(path)
    return select_halueval_qa_examples(records, limit, seed)


def select_halueval_qa_examples(
    records: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Sample unique QA rows and balance right/hallucinated answer labels."""
    if limit <= 0:
        raise ValueError(f"HaluEval limit must be positive, got {limit}")
    if len(records) < limit:
        raise ValueError(f"HaluEval has {len(records)} QA rows, but {limit} were requested")

    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), limit)
    hallucinated_count = limit // 2
    labels = [True] * hallucinated_count + [False] * (limit - hallucinated_count)
    rng.shuffle(labels)
    examples = []
    for index, hallucinated in zip(indices, labels):
        row = records[index]
        required = ("question", "right_answer", "hallucinated_answer")
        missing = [key for key in required if not isinstance(row.get(key), str)]
        if missing:
            raise ValueError(f"HaluEval row {index} is missing string fields {missing}")
        examples.append({
            "id": index,
            "question": row["question"],
            "answer": row["hallucinated_answer"] if hallucinated else row["right_answer"],
            "label": "yes" if hallucinated else "no",
        })
    return examples


def halueval_qa_prompt(example: dict[str, Any]) -> str:
    return (
        "You are a hallucination detector. Determine whether the answer contains "
        "hallucinated or factually incorrect information for the question. Use your "
        "world knowledge. Reply with exactly Yes or No.\n\n"
        f"#Question#: {example['question']}\n"
        f"#Answer#: {example['answer']}\n"
        "#Your Judgement#:"
    )


def parse_halueval_judgement(text: str) -> str | None:
    """Accept an unambiguous leading Yes/No token; reject malformed judgements."""
    match = re.match(r"\s*(yes|no)\b", text, flags=re.IGNORECASE)
    labels = {label.lower() for label in re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)}
    return match.group(1).lower() if match and len(labels) == 1 else None


def halueval_qa_metrics(
    examples: list[dict[str, Any]], predictions: list[str]
) -> dict[str, int | float]:
    if len(examples) != len(predictions):
        raise ValueError(
            f"received {len(predictions)} HaluEval predictions for {len(examples)} examples")
    if not examples:
        raise ValueError("cannot score an empty HaluEval subset")
    judgements = [parse_halueval_judgement(text) for text in predictions]
    correct = sum(judgement == example["label"]
                  for example, judgement in zip(examples, judgements))
    valid = sum(judgement is not None for judgement in judgements)
    n = len(examples)
    accuracy = correct / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (accuracy + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / denominator
    return {
        "n": n,
        "accuracy": accuracy,
        "accuracy_ci95_low": max(0.0, center - radius),
        "accuracy_ci95_high": min(1.0, center + radius),
        "valid_rate": valid / n,
        "invalid_predictions": n - valid,
    }


def evaluate_halueval_qa(
    model_name: str,
    adapter: Path | None,
    task_options: dict[str, Any],
    cache_dir: Path,
    batch_size: int | str,
) -> dict[str, dict[str, int | float]]:
    unsupported = set(task_options) - {"limit", "seed"}
    if unsupported:
        raise ValueError(f"unsupported HaluEval options: {sorted(unsupported)}")
    examples = load_halueval_qa(
        cache_dir, task_options.get("limit", 200), task_options.get("seed", 1234))
    from prefagent.eval.policy import generate, load_policy

    model, tokenizer = load_policy(model_name, adapter)
    predictions = generate(
        model, tokenizer, [halueval_qa_prompt(row) for row in examples],
        max_new_tokens=8, batch_size=8 if batch_size == "auto" else int(batch_size))
    return {"halueval_qa": halueval_qa_metrics(examples, predictions)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("benchmark results must not contain non-finite numeric values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    raise TypeError(f"benchmark result is not JSON serializable: {type(value).__name__}")
