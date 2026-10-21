"""Train and explain a lightweight pairwise (Bradley-Terry) reward model.

The model scores a response as r(x) = w·φ(x) and predicts P(chosen ≻ rejected) =
σ(r(chosen) − r(rejected)), so it is fitted on feature *differences*. Features are a
handful of named text signals plus a few PCA components of the sentence embedding,
which keeps the SHAP plot readable instead of ranking hundreds of raw embedding axes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HEDGE_TERMS = ("might", "possibly", "perhaps", "maybe", "probably", "not sure",
               "i think", "i believe", "it seems")
REFUSAL_TERMS = ("i can't", "i cannot", "i won't", "i'm not able", "i am not able",
                 "i'm sorry", "i apologize", "as an ai")


def _phrase_pattern(terms: tuple[str, ...]) -> re.Pattern:
    # Word boundaries so "might" does not match "mighty". Curly apostrophes are
    # normalized before matching (HH-RLHF uses both).
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b")


_HEDGE_RE = _phrase_pattern(HEDGE_TERMS)
_REFUSAL_RE = _phrase_pattern(REFUSAL_TERMS)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)

TEXT_FEATURES = ("log_word_count", "hedge_count", "refusal_count", "question_count",
                 "list_item_count")


def preference_pairs(rows: Iterable[dict], limit: int | None = None) -> list[dict]:
    """Normalize HH-RLHF prompt rows or DPO preference-pair rows."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    pairs = []
    for line_no, row in enumerate(rows, 1):
        if "chosen" in row and "rejected" in row:
            chosen, rejected = row["chosen"], row["rejected"]
            source = "preference_pairs"
        else:
            chosen, rejected = row.get("human_chosen"), row.get("human_rejected")
            source = "human_labels"
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            raise ValueError(f"record {line_no}: missing string chosen/rejected responses")
        if not chosen.strip() or not rejected.strip():
            raise ValueError(f"record {line_no}: chosen/rejected responses must not be empty")
        if chosen.strip() == rejected.strip():
            raise ValueError(f"record {line_no}: chosen and rejected responses are identical")
        pairs.append({"chosen": chosen, "rejected": rejected, "source": source,
                      "subset": row.get("subset")})
        if limit is not None and len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError("input contains no preference pairs")
    return pairs


def label_source(pairs: list[dict]) -> str:
    """The single label source, or "mixed:a+b" when the file combines formats."""
    sources = sorted({p["source"] for p in pairs})
    return sources[0] if len(sources) == 1 else "mixed:" + "+".join(sources)


def text_features(responses: list[str]) -> np.ndarray:
    """Named, human-readable response signals, one row per response (see TEXT_FEATURES)."""
    rows = []
    for text in responses:
        low = text.lower().replace("’", "'")
        rows.append((
            np.log1p(len(text.split())),
            len(_HEDGE_RE.findall(low)),
            len(_REFUSAL_RE.findall(low)),
            text.count("?"),
            len(_LIST_ITEM_RE.findall(text)),
        ))
    return np.asarray(rows, dtype=np.float64).reshape(len(responses), len(TEXT_FEATURES))


def response_features(embeddings: np.ndarray, responses: list[str],
                      embedding_prefix: str = "embedding_pc") -> tuple[np.ndarray, list[str]]:
    """Combine (already reduced) embedding columns with the named text features.

    `embeddings` may have zero columns, giving a text-features-only model.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if len(embeddings) != len(responses):
        raise ValueError("embedding count must match response count")
    features = np.column_stack((text_features(responses), embeddings))
    names = list(TEXT_FEATURES)
    names.extend(f"{embedding_prefix}{i + 1}" for i in range(embeddings.shape[1]))
    return features, names


class PairwiseRewardModel:
    """Bradley-Terry reward model: logistic regression on φ(chosen) − φ(rejected).

    No intercept and no centring, so score(a, b) == −score(b, a). Training pairs are
    randomly sign-flipped so both classes are present.
    """

    def __init__(self, seed: int = 0, C: float = 1.0):
        self.seed = seed
        self.scaler = StandardScaler(with_mean=False)
        self.model = LogisticRegression(fit_intercept=False, C=C, max_iter=1000)

    def fit(self, chosen: np.ndarray, rejected: np.ndarray) -> "PairwiseRewardModel":
        diffs = chosen - rejected
        flip = np.random.default_rng(self.seed).random(len(diffs)) < 0.5
        if flip.all() or not flip.any():  # guarantee both classes for tiny inputs
            flip[0] = not flip[0]
        signs = np.where(flip, -1.0, 1.0)
        x = self.scaler.fit_transform(diffs * signs[:, None])
        self.model.fit(x, (~flip).astype(int))
        return self

    def transform(self, chosen: np.ndarray, rejected: np.ndarray) -> np.ndarray:
        return self.scaler.transform(chosen - rejected)

    def margins(self, chosen: np.ndarray, rejected: np.ndarray) -> np.ndarray:
        """r(chosen) − r(rejected) in standardized units; > 0 means chosen is preferred."""
        return self.model.decision_function(self.transform(chosen, rejected))

    @property
    def coefficients(self) -> np.ndarray:
        """Weights on standardized feature differences (positive → feature is preferred)."""
        return self.model.coef_[0]


def pairwise_accuracy(margins: np.ndarray) -> np.ndarray:
    """Per-pair correctness: 1 if chosen scored higher, 0.5 on an exact tie, else 0."""
    return (margins > 0) + 0.5 * (margins == 0)
