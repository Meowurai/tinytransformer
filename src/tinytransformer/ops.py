# src/tinytransformer/ops.py

from tinytransformer.autograd import Value

# Small reusable vector operations used by transformer components.

def dot(a: list[Value], b: list[Value]) -> Value:
    total = Value(0.0)
    for a_value, b_value in zip(a, b):
        total += a_value * b_value

    return total