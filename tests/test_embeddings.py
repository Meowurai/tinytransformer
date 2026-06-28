from tinytransformer.embeddings import EmbeddingTable

def test_token_and_position_embeddings_can_be_added():
    token_embeddings = EmbeddingTable(vocabulary_size=5, embedding_size=3)
    position_embeddings = EmbeddingTable(vocabulary_size=4, embedding_size=3)

    token_ids = [1, 2, 3, 4]
    token_vectors = token_embeddings(token_ids)
    position_vectors = position_embeddings([0, 1, 2, 3])

    combined = []
    for token_vector, position_vector in zip(token_vectors, position_vectors):
        combined.append([
            token_value + position_value
            for token_value, position_value in zip(token_vector, position_vector)
        ])

    assert len(combined) == 4
    assert len(combined[0]) == 3