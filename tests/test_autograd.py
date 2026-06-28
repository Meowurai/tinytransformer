from tinytransformer.autograd import Value


def test_value_stores_data_and_grad():
    x = Value(2.0)

    assert x.data == 2.0
    assert x.grad == 0.0


def test_value_repr():
    x = Value(2.0)

    assert repr(x) == "Value(data=2.0, grad=0.0)"