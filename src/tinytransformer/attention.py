# src/tinytransformer/attention.py

from tinytransformer.autograd import Value
from tinytransformer.ops import dot

def attention_scores(query: list[Value], keys: list[list[Value]]) -> list[Value]:
    scores = []
    for key in keys:
        score = dot(query, key)
        scores.append(score)

    return scores
