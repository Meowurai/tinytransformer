from tinytransformer.autograd import Value
from tinytransformer.ops import dot, add_vectors

def test_dot_product_returns_single_value():
    a = [Value(1.0), Value(2.0), Value(3.0)]
    b = [Value(4.0), Value(5.0), Value(6.0)]

    result = dot(a, b)

    assert result.data == 32.0

def test_add_vectors_adds_elementwise():
    a = [Value(1.0), Value(2.0)]
    b = [Value(3.0), Value(4.0)]

    result = add_vectors(a, b)

    assert [v.data for v in result] == [4.0, 6.0]