from tinytransformer.transformer import Transformer

def test_transformer_returns_one_logit_vector_per_input_token():
    transformer = Transformer(
        vocabulary_size=10,
        embedding_size=4,
        num_heads=2,
        num_blocks=2,
        block_size=4,
    )  

    token_ids = [1, 2, 3]

    logits = transformer(token_ids)

    assert len(logits) == 3
    assert all(len(logit) == 10 for logit in logits)