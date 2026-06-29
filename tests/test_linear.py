from tinytransformer.autograd import Value 
from tinytransformer.layers import Linear

def test_linear_returns_output_vector():
    layer = Linear(input_size=3, output_size=2)

    inputs = [Value(1.0), Value(2.0), Value(3.0)]

    outputs = layer(inputs)

    assert len(outputs) == 2
    assert all(isinstance(output, Value) for output in outputs)

def test_linear_parameters_include_weights_and_bias():
    layer = Linear(input_size=3, output_size=2)

    assert len(layer.parameters()) == 8