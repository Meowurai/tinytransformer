# src/tinytransformer/activations.py

from tinytransformer.autograd import Value

def softmax(values: list[Value]) -> list[Value]:
    max_value = max(value.data for value in values)
    weights = [value.exp() for value in values]
    total_weight = weights[0]
    for weight in weights[1:]:
        total_weight += weight

    return [weight / total_weight for weight in weights]