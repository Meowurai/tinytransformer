from tinytransformer.autograd import Value
from tinytransformer.ops import dot

def test_dot_product_returns_single_value():
    a = [Value(1.0), Value(2.0), Value(3.0)]
    b = [Value(4.0), Value(5.0), Value(6.0)]

    result = dot(a, b)

    assert result.data == 32.0