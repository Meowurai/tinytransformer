# src/tinytransfomer/loss.py

from tinytransformer.autograd import Value
from tinytransformer.activations import softmax

def cross_entropy(probabilities: list[Value], target: int) -> Value:
    return Value(-1.0) * probabilities[target].log()

def cross_entropy_loss(logits: list[list[Value]], target_ids: list[int]):
    assert len(logits) == len(target_ids)
    
    total = Value(0.0)

    for position_logits, target_id in zip(logits, target_ids):
        probabilities = softmax(position_logits)
        total += cross_entropy(probabilities, target_id)

    return total / Value(len(target_ids))
