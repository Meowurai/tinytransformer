from tinytransformer.autograd import Value
from tinytransformer.activations import softmax

def test_softmax_outputs_sum_to_one():
    values = [Value(1.0), Value(2.0), Value(3.0)]

    probabilities = softmax(values)

    assert abs(sum(p.data for p in probabilities) - 1.0) < 1e-9