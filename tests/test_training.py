from tinytransformer.autograd import Value
from tinytransformer.training import zero_grad, step 
from tinytransformer.dataset import build_example
from tinytransformer.loss import cross_entropy_loss
from tinytransformer.transformer import Transformer

class FakeModel:
    def __init__(self):
        self.weight = Value(1.0)

    def parameters(self):
        return [self.weight]
    

def test_zero_grad_resets_parameter_gradients():
    model = FakeModel()
    model.weight.grad = 3.0

    zero_grad(model)

    assert model.weight.grad == 0.0 

def test_step_updates_parameters_using_gradient_descent():
    model = FakeModel()
    model.weight.grad = 2.0

    step(model, learning_rate=0.1)

    assert model.weight.data == 0.8


def test_one_training_step_runs_without_errors():
    token_ids = [1, 2, 3, 4, 5]

    x, y = build_example(
        token_ids,
        start_index=0,
        block_size=4
    )

    model = Transformer(
        vocabulary_size=6,
        embedding_size=4,
        num_heads=2,
        num_blocks=1,
        block_size=4,
    )

    logits = model(x)
    loss = cross_entropy_loss(logits, y)

    zero_grad(model)
    loss.backward()
    step(model, learning_rate=0.01)

    assert loss.data > 0