"""Train a small reward model and explain its preference signals with SHAP.

    python scripts/06_explain_preferences.py
    python scripts/06_explain_preferences.py --input data/pairs/train/multi_panel.jsonl

Uses prepared HH-RLHF human labels when no pair file is supplied. No LLM API is
called. Outputs a held-out pairwise accuracy summary and a SHAP bar plot.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefagent.explain.reward import preference_pairs, response_features  # noqa: E402
from prefagent.utils.io import read_jsonl  # noqa: E402


def analyze(input_path: Path, output_dir: Path, limit: int | None,
            test_size: float, seed: int) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from sentence_transformers import SentenceTransformer

    pairs = preference_pairs(read_jsonl(input_path), limit)
    if len(pairs) < 4:
        raise ValueError(f"at least 4 preference pairs are required; found {len(pairs)}")
    if not 0 < test_size < 0.5:
        raise ValueError("--test-size must be greater than 0 and less than 0.5")

    responses = [text for pair in pairs for text in (pair["chosen"], pair["rejected"])]
    logging.info("Embedding %d responses on CPU", len(responses))
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embeddings = embedder.encode(
        responses, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    x, feature_names = response_features(np.asarray(embeddings), responses)
    y = np.tile(np.array([1, 0], dtype=np.int8), len(pairs))

    train_pairs, test_pairs = train_test_split(
        np.arange(len(pairs)), test_size=test_size, random_state=seed
    )
    train_rows = np.ravel(np.column_stack((2 * train_pairs, 2 * train_pairs + 1)))
    test_rows = np.ravel(np.column_stack((2 * test_pairs, 2 * test_pairs + 1)))
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(x[train_rows], y[train_rows])

    chosen_rows, rejected_rows = 2 * test_pairs, 2 * test_pairs + 1
    margins = model.decision_function(x[chosen_rows]) - model.decision_function(x[rejected_rows])
    pairwise_accuracy = float(np.mean((margins > 0) + 0.5 * (margins == 0)))

    explainer = shap.Explainer(model, x[train_rows])
    explanations = explainer(x[test_rows])
    shap_values = np.asarray(explanations.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    importance = np.abs(shap_values).mean(axis=0)
    top = np.argsort(importance)[::-1][:15]

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "shap_summary.png"
    shap.summary_plot(
        shap_values,
        x[test_rows],
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    result = {
        "input": str(input_path),
        "label_source": pairs[0]["source"],
        "n_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "pairwise_accuracy": pairwise_accuracy,
        "mean_chosen_minus_rejected_score": float(margins.mean()),
        "feature_count": len(feature_names),
        "top_shap_features": [
            {"feature": feature_names[i], "mean_absolute_shap": float(importance[i])}
            for i in top
        ],
        "shap_plot": str(plot_path),
        "interpretation_note": (
            "Exploratory reward model only. Accuracy is measured on held-out pairs; "
            "small smoke-test datasets are not research evidence."
        ),
    }
    with (output_dir / "reward_model.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="JSONL pair file; defaults to prepared HH-RLHF train prompts")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explain"))
    parser.add_argument("--limit", type=int, default=None, help="maximum number of pairs")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    default_pairs = Path("data/pairs/train/multi_panel.jsonl")
    default_prompts = Path("data/prompts/train.jsonl")
    input_path = args.input or (default_pairs if default_pairs.exists() else default_prompts)
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = analyze(input_path, args.output_dir, args.limit, args.test_size, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
