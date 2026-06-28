# src/tinytransformer/attention.py

from tinytransformer.autograd import Value
from tinytransformer.ops import dot

def attention_scores(query: list[Value], keys: list[list[Value]]) -> list[Value]:
    scores = []
    for key in keys:
        score = dot(query, key)
        scores.append(score)

    return scores

def weighted_sum(weights: list[Value], vectors: list[list[Value]]) -> list[Value]:
    result = [Value(0.0) for _ in vectors[0]]

    for vector_idx, vector in enumerate(vectors):
        for value_idx, value in enumerate(vector):
            result[value_idx] += weights[vector_idx] * value

    return result
