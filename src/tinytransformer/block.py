# src/tinytransformer/block.py

from tinytransformer.autograd import Value
from tinytransformer.layers import MLP, LayerNorm
from tinytransformer.attention import MultiHeadAttention
from tinytransformer.ops import add_sequences

class TransformerBlock:
    def __init__(self, embedding_size: int, num_heads: int) -> None:
        self.attention = MultiHeadAttention(embedding_size, num_heads)
        self.norm1 = LayerNorm()
        self.mlp = MLP(embedding_size)
        self.norm2 = LayerNorm()

    def __call__(self, vectors: list[list[Value]]) -> list[list[Value]]:
        attention_output = self.attention(vectors)
        vectors = add_sequences(vectors, attention_output)
        vectors = [self.norm1(vector) for vector in vectors]

        mlp_output = [self.mlp(vector) for vector in vectors]
        vectors = add_sequences(vectors, mlp_output)
        vectors = [self.norm2(vector) for vector in vectors]

        return vectors
    
    def parameters(self) -> list[Value]:
        return self.attention.parameters() + self.mlp.parameters()

