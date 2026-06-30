from tinytransformer.autograd import Value
from tinytransformer.loss import cross_entropy_loss


def test_cross_entropy_loss_returns_value():
    logits = [
        [Value(1.0), Value(2.0), Value(3.0)],
        [Value(3.0), Value(2.0), Value(1.0)],
    ]

    target_ids = [2, 0]

    loss = cross_entropy_loss(logits, target_ids)

    assert isinstance(loss, Value)


def test_cross_entropy_loss_is_positive():
    logits = [
        [Value(1.0), Value(2.0), Value(3.0)],
        [Value(3.0), Value(2.0), Value(1.0)],
    ]

    target_ids = [2, 0]

    loss = cross_entropy_loss(logits, target_ids)

    assert loss.data > 0