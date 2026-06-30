from tinytransformer.autograd import Value
from tinytransformer.block import TransformerBlock

def test_transformer_block_preserves_sequence_shape():
    block = TransformerBlock(embedding_size=4, num_heads=2)

    vectors = [
        [Value(1.0), Value(0.0), Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0), Value(0.0), Value(0.0)],
        [Value(0.0), Value(0.0), Value(1.0), Value(0.0)],
    ]

    outputs = block(vectors)

    assert len(outputs) == 3 
    assert all(len(output) == 4 for output in outputs)