import pytest

from tinytransformer.autograd import Value
from tinytransformer.attention import attention_scores, weighted_sum

def test_attention_scores_returns_one_score_per_key():
    query = [Value(1.0), Value(2.0)]
    keys = [
        [Value(1.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(1.0)],
    ]

    scores = attention_scores(query, keys)

    assert [score.data for score in scores] == [1.0, 2.0, 3.0]

def test_weighted_sum_returns_vector():
    weights = [
        Value(0.2),
        Value(0.8),
    ]

    vectors = [
        [Value(1.0), Value(2.0)],
        [Value(3.0), Value(4.0)]
    ]

    result = weighted_sum(weights, vectors)

    assert [v.data for v in result] == pytest.approx([2.6, 3.6])