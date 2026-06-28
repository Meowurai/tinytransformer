# src/tinytransformer/embeddings.py

import random

from tinytransformer.autograd import Value

class EmbeddingTable:
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
    ) -> None:
        self.vectors = [
            [
                Value(random.uniform(-0.1, 0.1))
                for _ in range(embedding_size)
            ]
            for _ in range(vocabulary_size)
        ]

    def __call__(self, token_ids: list[int]) -> list[list[Value]]:
        return [self.lookup(token_id) for token_id in token_ids]

    def parameters(self) -> list[Value]:
        return [vector for row in self.vectors for vector in row]
    
    def lookup(self, token_id: int) -> list[Value]:
        return self.vectors[token_id]