# src/tinytransformer/ops.py

from tinytransformer.autograd import Value

# Small reusable vector operations used by transformer components.

def dot(a: list[Value], b: list[Value]) -> Value:
    total = Value(0.0)
    for a_value, b_value in zip(a, b):
        total += a_value * b_value

    return total


def add_vectors(a: list[Value], b: list[Value]) -> list[Value]:
    """Element-wise vector addition"""
    return [
        a_value + b_value
        for a_value, b_value in zip(a, b)
    ]

def add_sequences(a: list[list[Value]], b: list[list[Value]]) -> list[list[Value]]:
    return [
        add_vectors(a_vector, b_vector)
        for a_vector, b_vector in zip(a, b)
    ]


