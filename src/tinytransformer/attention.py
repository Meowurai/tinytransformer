# src/tinytransformer/attention.py

from tinytransformer.autograd import Value
from tinytransformer.ops import dot
from tinytransformer.activations import softmax
from tinytransformer.layers import Linear

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

class AttentionHead:
    def __init__(self, embedding_size: int) -> None:
        self.query_projection = Linear(embedding_size, embedding_size)
        self.key_projection = Linear(embedding_size, embedding_size)
        self.value_projection = Linear(embedding_size, embedding_size)

    def __call__(self, vectors: list[list[Value]]) -> list[list[Value]]:
        queries = [self.query_projection(vector) for vector in vectors]
        keys = [self.key_projection(vector) for vector in vectors]
        values = [self.value_projection(vector) for vector in vectors]

        outputs = []
        for query in queries:
            scores = attention_scores(query, keys) 
            weights = softmax(scores)
            output = weighted_sum(weights, values) 
            outputs.append(output)

        return outputs
    
    def parameters(self) -> list[Value]:
        return (
            self.query_projection.parameters() 
            + self.key_projection.parameters()
            + self.value_projection.parameters()
        )


class MultiHeadAttention:
    def __init__(self, embedding_size: int, num_heads: int) -> None:
        self.attention_heads = [AttentionHead(embedding_size) for _ in range(num_heads)]
        self.output_projection = Linear(embedding_size * num_heads, embedding_size)
    
    def __call__(self, vectors: list[list[Value]]) -> list[list[Value]]:
        head_outputs = [head(vectors) for head in self.attention_heads]

        outputs = []
        for token_index in range(len(vectors)):
            concatenated = []
            for head_output in head_outputs:
                concatenated += head_output[token_index]

            output = self.output_projection(concatenated)
            outputs.append(output)

        return outputs
    
    def parameters(self) -> list[Value]:
        parameters = []
        for head in self.attention_heads:
            params = head.parameters()
            for param in params:
                parameters.append(param)

        parameters += self.output_projection.parameters()

        return parameters