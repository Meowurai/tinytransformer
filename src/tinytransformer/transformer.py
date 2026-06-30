# src/tinytransformer/transformer.py

from tinytransformer.autograd import Value
from tinytransformer.embeddings import EmbeddingTable
from tinytransformer.block import TransformerBlock
from tinytransformer.layers import Linear

class Transformer:
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        num_heads: int,
        num_blocks: int,
        block_size: int            
    ) -> None:
        self.token_embeddings = EmbeddingTable(vocabulary_size, embedding_size)
        self.position_embeddings = EmbeddingTable(block_size, embedding_size)

        self.blocks = [TransformerBlock(embedding_size, num_heads) for _ in range(num_blocks)]
        self.output_projection = Linear(embedding_size, vocabulary_size)

    def __call__(self, token_ids: list[int]) -> list[list[Value]]:
        token_vectors = self.token_embeddings(token_ids)
        position_vectors = self.position_embeddings(list(range(len(token_ids))))

        vectors = []
        for token_vector, position_vector in zip(token_vectors, position_vectors):
            vectors.append([
                token_value + position_value
                for token_value, position_value in zip(token_vector, position_vector)
            ])

        for block in self.blocks:
            vectors = block(vectors)

        return [self.output_projection(vector) for vector in vectors]