from tinytransformer.autograd import Value 
from tinytransformer.layers import LayerNorm 

def test_layer_norm_preserves_vector_size():
    layer_norm = LayerNorm()

    vector = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
        Value(4.0),
    ]

    output = layer_norm(vector)

    assert len(output) == 4

def test_layer_norm_centers_output_around_zero():
    layer_norm = LayerNorm()

    vector = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]

    output = layer_norm(vector)

    mean = sum(value.data for value in output) / len(output)

    assert abs(mean) < 1e-6