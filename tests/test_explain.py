import numpy as np
import pytest

from prefagent.explain.reward import preference_pairs, response_features


def test_preference_pairs_accepts_prepared_human_labels_and_limit():
    rows = [
        {"human_chosen": "Good answer.", "human_rejected": "Bad answer."},
        {"human_chosen": "Clear response.", "human_rejected": "Unclear response."},
    ]
    assert preference_pairs(rows, limit=1) == [
        {"chosen": "Good answer.", "rejected": "Bad answer.", "source": "human_labels"}
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


def test_response_features_include_length_and_hedging():
    texts = ["This might work, possibly.", "A direct answer."]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    features, names = response_features(embeddings, texts)
    assert names == ["sentence_embedding_0", "sentence_embedding_1", "word_count",
                     "hedge_count"]
    assert features.shape == (2, 4)
    assert features[:, -2].tolist() == [4, 3]
    assert features[:, -1].tolist() == [2, 0]


def test_response_features_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError, match="embedding count"):
        response_features(np.zeros((1, 2)), ["first", "second"])


def test_response_features_rejects_non_matrix_embeddings():
    with pytest.raises(ValueError, match="two-dimensional"):
        response_features(np.zeros(2), ["first", "second"])
