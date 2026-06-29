import pytest

from tinytransformer.autograd import Value
from tinytransformer.attention import (
    attention_scores, weighted_sum, 
    AttentionHead, MultiHeadAttention
)

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

def test_attention_head_returns_one_vector_per_input_vector():
    head = AttentionHead(embedding_size=2)

    vectors = [
        [Value(1.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(1.0)],
    ]

    outputs = head(vectors)

    assert len(outputs) == 3
    assert all(len(output) == 2 for output in outputs)


def test_attention_head_parameters_include_qkv_projections():
    head = AttentionHead(embedding_size=2)

    assert len(head.parameters()) == 18


def test_attention_head_returns_contextualized_vectors():
    head = AttentionHead(embedding_size=2)

    vectors = [
        [Value(1.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(1.0)],
    ]

    outputs = head(vectors)

    assert len(outputs) == len(vectors)
    assert all(len(output) == 2 for output in outputs)


def test_multi_head_attention_preserves_sequence_shape():
    mha = MultiHeadAttention(embedding_size=4, num_heads=2)

    vectors = [
        [Value(1.0), Value(0.0), Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0), Value(0.0), Value(0.0)],
        [Value(0.0), Value(0.0), Value(1.0), Value(0.0)],
    ]

    outputs = mha(vectors)

    assert len(outputs) == 3
    assert all(len(output) == 4 for output in outputs)