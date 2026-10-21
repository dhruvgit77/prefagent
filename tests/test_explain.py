import numpy as np
import pytest

from prefagent.explain.reward import (PairwiseRewardModel, label_source, pairwise_accuracy,
                                      preference_pairs, response_features)


def test_preference_pairs_accepts_prepared_human_labels_and_limit():
    rows = [
        {"human_chosen": "Good answer.", "human_rejected": "Bad answer."},
        {"human_chosen": "Clear response.", "human_rejected": "Unclear response."},
    ]
    assert preference_pairs(rows, limit=1) == [
        {"chosen": "Good answer.", "rejected": "Bad answer.", "source": "human_labels",
         "subset": None}
    ]


def test_preference_pairs_accepts_dpo_rows():
    rows = [{"chosen": "A good answer.", "rejected": "A poor answer."}]
    assert preference_pairs(rows)[0]["source"] == "preference_pairs"


@pytest.mark.parametrize("row, message", [
    ({"chosen": "yes"}, "missing string"),
    ({"chosen": "", "rejected": "no"}, "must not be empty"),
    ({"chosen": "same", "rejected": " same "}, "identical"),
])
def test_preference_pairs_rejects_invalid_rows(row, message):
    with pytest.raises(ValueError, match=message):
        preference_pairs([row])


def test_preference_pairs_requires_positive_limit():
    with pytest.raises(ValueError, match="limit must be positive"):
        preference_pairs([{"chosen": "yes", "rejected": "no"}], limit=0)


def test_response_features_names_and_values():
    texts = ["This might work, possibly. Maybe?", "I'm sorry, I can’t help.\n- one\n2. two"]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    features, names = response_features(embeddings, texts)
    assert names == ["log_word_count", "hedge_count", "refusal_count", "question_count",
                     "list_item_count", "embedding_pc1", "embedding_pc2"]
    assert features.shape == (2, 7)
    assert features[:, 0].tolist() == pytest.approx(np.log1p([5, 9]).tolist())
    assert features[:, 1:5].tolist() == [[3, 0, 1, 0], [0, 2, 0, 2]]
    np.testing.assert_allclose(features[:, 5:], embeddings)


def test_hedge_terms_match_whole_words_only():
    features, _ = response_features(np.zeros((1, 0)), ["A mighty, maybelline answer."])
    assert features[0, 1] == 0


def test_response_features_allow_text_only():
    features, names = response_features(np.zeros((2, 0)), ["a", "b"])
    assert features.shape == (2, 5) and "embedding_pc1" not in names


def test_label_source_reports_mixed_files():
    rows = [{"chosen": "a", "rejected": "b"}, {"human_chosen": "c", "human_rejected": "d"}]
    assert label_source(preference_pairs(rows)) == "mixed:human_labels+preference_pairs"
    assert label_source(preference_pairs(rows[:1])) == "preference_pairs"


def test_pairwise_model_learns_signal_and_is_antisymmetric():
    rng = np.random.default_rng(0)
    rejected = rng.normal(size=(200, 3))
    chosen = rejected + rng.normal(size=(200, 3)) * 0.1
    chosen[:, 0] += 1.0  # chosen is always higher on feature 0
    model = PairwiseRewardModel(seed=0).fit(chosen, rejected)
    margins = model.margins(chosen, rejected)
    assert pairwise_accuracy(margins).mean() > 0.95
    assert np.argmax(np.abs(model.coefficients)) == 0 and model.coefficients[0] > 0
    assert model.margins(rejected, chosen) == pytest.approx(-margins)


def test_pairwise_accuracy_counts_ties_as_half():
    assert pairwise_accuracy(np.array([1.0, -1.0, 0.0])).tolist() == [1.0, 0.0, 0.5]


def test_response_features_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError, match="embedding count"):
        response_features(np.zeros((1, 2)), ["first", "second"])


def test_response_features_rejects_non_matrix_embeddings():
    with pytest.raises(ValueError, match="two-dimensional"):
        response_features(np.zeros(2), ["first", "second"])
