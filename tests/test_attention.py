from tinytransformer.autograd import Value
from tinytransformer.attention import attention_scores

def test_attention_scores_returns_one_score_per_key():
    query = [Value(1.0), Value(2.0)]
    keys = [
        [Value(1.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(1.0)],
    ]

    scores = attention_scores(query, keys)

    assert [score.data for score in scores] == [1.0, 2.0, 3.0]