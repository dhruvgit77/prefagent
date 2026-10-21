"""Train a pairwise reward model and explain its preference signals with SHAP.

    python scripts/06_explain_preferences.py
    python scripts/06_explain_preferences.py --limit 500
    python scripts/06_explain_preferences.py --input data/pairs/train/multi_panel.jsonl

Uses prepared HH-RLHF human labels when no pair file is supplied. No LLM API is
called. Accuracy is measured out-of-fold with k-fold cross-validation (with a
bootstrap CI) and compared to a "longer response wins" baseline. Writes
reward_model.json, shap_summary.png and coefficients.png to --output-dir.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefagent.eval.stats import bootstrap_ci  # noqa: E402
from prefagent.explain.reward import (  # noqa: E402
    TEXT_FEATURES, PairwiseRewardModel, label_source, pairwise_accuracy, preference_pairs,
    response_features)
from prefagent.utils.io import read_jsonl  # noqa: E402

MIN_PAIRS_PER_FOLD = 2


def featurize(embeddings: np.ndarray, responses: list[str], fit_idx: np.ndarray,
              components: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """PCA is fitted on the training responses only, then applied to every response."""
    if components == 0:
        reduced = np.zeros((len(responses), 0))
    else:
        k = min(components, len(fit_idx) - 1, embeddings.shape[1])
        pca = PCA(n_components=k, random_state=seed).fit(embeddings[fit_idx])
        reduced = pca.transform(embeddings)
    return response_features(reduced, responses)


def rows_of(pair_idx: np.ndarray) -> np.ndarray:
    """Response rows for pairs (responses are stored chosen, rejected, chosen, ...)."""
    return np.ravel(np.column_stack((2 * pair_idx, 2 * pair_idx + 1)))


def cross_validate(embeddings: np.ndarray, responses: list[str], folds: int,
                   components: int, seed: int) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold per-pair correctness and per-fold accuracies."""
    pair_idx = np.arange(len(responses) // 2)
    oof = np.empty(len(pair_idx))
    fold_acc = []
    for train_p, test_p in KFold(folds, shuffle=True, random_state=seed).split(pair_idx):
        x, _ = featurize(embeddings, responses, rows_of(train_p), components, seed)
        model = PairwiseRewardModel(seed).fit(x[2 * train_p], x[2 * train_p + 1])
        oof[test_p] = model.margins(x[2 * test_p], x[2 * test_p + 1])
        fold_acc.append(float(pairwise_accuracy(oof[test_p]).mean()))
    return pairwise_accuracy(oof), fold_acc


def fit_full(embeddings: np.ndarray, responses: list[str], components: int,
             seed: int) -> tuple[PairwiseRewardModel, np.ndarray, list[str]]:
    x, names = featurize(embeddings, responses, np.arange(len(responses)), components, seed)
    return PairwiseRewardModel(seed).fit(x[0::2], x[1::2]), x, names


def length_baseline(responses: list[str]) -> np.ndarray:
    """Per-pair correctness of "the longer response is the chosen one"."""
    words = np.array([len(t.split()) for t in responses])
    return pairwise_accuracy(words[0::2] - words[1::2])


def pc_examples(x: np.ndarray, names: list[str], responses: list[str],
                k: int = 3, width: int = 160) -> dict:
    """Responses scoring highest/lowest on each embedding PC, so the axes can be named."""
    out = {}
    for j, name in enumerate(names):
        if name.startswith("embedding_pc"):
            order = np.argsort(x[:, j])
            clip = lambda i: " ".join(responses[i].split())[:width]  # noqa: E731
            out[name] = {"high": [clip(i) for i in order[::-1][:k]],
                         "low": [clip(i) for i in order[:k]]}
    return out


def analyze(input_path: Path, output_dir: Path, limit: int | None, folds: int,
            components: int, seed: int) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    pairs = preference_pairs(read_jsonl(input_path), limit)
    if len(pairs) < folds * MIN_PAIRS_PER_FOLD:
        raise ValueError(f"{folds}-fold CV needs at least {folds * MIN_PAIRS_PER_FOLD} "
                         f"pairs; found {len(pairs)}")

    responses = [text for pair in pairs for text in (pair["chosen"], pair["rejected"])]
    if components > 0:
        from sentence_transformers import SentenceTransformer

        logging.info("Embedding %d responses on CPU", len(responses))
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        embeddings = np.asarray(embedder.encode(
            responses, batch_size=64, normalize_embeddings=True, show_progress_bar=False))
    else:
        embeddings = np.zeros((len(responses), 0))

    correct, fold_acc = cross_validate(embeddings, responses, folds, components, seed)
    lo, hi = bootstrap_ci(correct.tolist(), seed=seed)
    length_correct = length_baseline(responses)

    # --- per-subset models (e.g. HH-RLHF helpful vs harmless) ------------------
    by_subset = {}
    subsets = np.array([p["subset"] or "" for p in pairs])
    for name in sorted(set(subsets) - {""}):
        idx = np.flatnonzero(subsets == name)
        if len(idx) < folds * MIN_PAIRS_PER_FOLD:
            continue
        rows = rows_of(idx)
        sub_emb, sub_resp = embeddings[rows], [responses[i] for i in rows]
        sub_correct, _ = cross_validate(sub_emb, sub_resp, folds, components, seed)
        sub_model, _, sub_names = fit_full(sub_emb, sub_resp, components, seed)
        by_subset[name] = {
            "n_pairs": len(idx),
            "pairwise_accuracy": float(sub_correct.mean()),
            "pairwise_accuracy_ci95": list(bootstrap_ci(sub_correct.tolist(), seed=seed)),
            "baseline_longer_response_wins": float(length_baseline(sub_resp).mean()),
            "text_feature_coefficients": {
                n: float(c) for n, c in zip(sub_names, sub_model.coefficients)
                if n in TEXT_FEATURES},
        }

    # --- final model on all pairs, explained with SHAP -------------------------
    model, x, names = fit_full(embeddings, responses, components, seed)
    diffs = model.transform(x[0::2], x[1::2])
    explainer = shap.LinearExplainer(model.model, diffs)
    shap_values = np.asarray(explainer(diffs).values)
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    shap_path = output_dir / "shap_summary.png"
    shap.summary_plot(shap_values, diffs, feature_names=names, max_display=15, show=False)
    plt.title("SHAP: feature difference (chosen − rejected) → preference margin")
    plt.tight_layout()
    plt.savefig(shap_path, dpi=160, bbox_inches="tight")
    plt.close()

    coef = model.coefficients
    coef_path = output_dir / "coefficients.png"
    shown = order[:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(shown) + 1.2))
    ax.barh([names[i] for i in shown], coef[shown],
            color=["#2a7f62" if coef[i] > 0 else "#b5473a" for i in shown])
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("weight on standardized difference  (→ preferred,  ← dispreferred)")
    ax.set_title("What the reward model rewards")
    fig.tight_layout()
    fig.savefig(coef_path, dpi=160)
    plt.close(fig)

    subset_path = None
    if len(by_subset) > 1:
        subset_path = output_dir / "subset_coefficients.png"
        fig, ax = plt.subplots(figsize=(7, 3.6))
        h = 0.8 / len(by_subset)
        ypos = np.arange(len(TEXT_FEATURES))
        for k, (name, sub) in enumerate(by_subset.items()):
            vals = [sub["text_feature_coefficients"][f] for f in TEXT_FEATURES]
            label = f"{name} (acc {sub['pairwise_accuracy']:.1%}, n={sub['n_pairs']})"
            ax.barh(ypos + (k - (len(by_subset) - 1) / 2) * h, vals, height=h, label=label)
        ax.set_yticks(ypos, TEXT_FEATURES)
        ax.invert_yaxis()
        ax.axvline(0, color="#444", linewidth=0.8)
        ax.set_xlabel("weight on standardized difference  (→ preferred,  ← dispreferred)")
        ax.set_title("Same signal, different subset: what gets rewarded")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(subset_path, dpi=160)
        plt.close(fig)

    result = {
        "input": str(input_path),
        "label_source": label_source(pairs),
        "n_pairs": len(pairs),
        "model": "pairwise Bradley-Terry logistic regression on standardized feature differences",
        "cv_folds": folds,
        "pairwise_accuracy": float(correct.mean()),
        "pairwise_accuracy_ci95": [lo, hi],
        "fold_accuracies": fold_acc,
        "baseline_longer_response_wins": float(length_correct.mean()),
        "embedding_pca_components": sum(n.startswith("embedding_pc") for n in names),
        "features": [
            {"feature": names[i], "coefficient": float(coef[i]),
             "mean_absolute_shap": float(importance[i]),
             "direction": "preferred" if coef[i] > 0 else "dispreferred"}
            for i in order
        ],
        "by_subset": by_subset,
        "embedding_pc_examples": pc_examples(x, names, responses),
        "shap_plot": str(shap_path),
        "coefficient_plot": str(coef_path),
        "subset_coefficient_plot": str(subset_path) if subset_path else None,
        "interpretation_note": (
            "Exploratory reward model. Accuracy is out-of-fold; check the CI width "
            "before drawing conclusions. SHAP and coefficients describe this model, "
            "not causal effects on human preference."
        ),
    }
    with (output_dir / "reward_model.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None,
                        help="JSONL pair file; defaults to agent pairs if present, "
                             "else prepared HH-RLHF train prompts")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explain"))
    parser.add_argument("--limit", type=int, default=None, help="maximum number of pairs")
    parser.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    parser.add_argument("--embedding-components", type=int, default=8,
                        help="PCA components of the sentence embedding (0 = text features only)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.embedding_components < 0:
        parser.error("--embedding-components must be non-negative")

    default_pairs = Path("data/pairs/train/multi_panel.jsonl")
    default_prompts = Path("data/prompts/train.jsonl")
    input_path = args.input or (default_pairs if default_pairs.exists() else default_prompts)
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = analyze(input_path, args.output_dir, args.limit, args.folds,
                     args.embedding_components, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
