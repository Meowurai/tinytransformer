
from tinytransformer.tokenizer import Tokenizer


def test_tokenizer_builds_vocabulary_from_text():
    tokenizer = Tokenizer.from_text("banana")

    assert tokenizer.vocabulary_size == 3

def test_tokenizer_encodes_text_to_ids():
    tokenizer = Tokenizer.from_text("abc")
    ids = tokenizer.encode("cab")

    assert ids == [
        tokenizer.token_to_id["c"],
        tokenizer.token_to_id["a"],
        tokenizer.token_to_id["b"],
    ]

def test_tokenizer_decodes_ids_to_text():
    tokenizer = Tokenizer.from_text("abc")
    ids = tokenizer.encode("cab")

    text = tokenizer.decode(ids)

    assert text == "cab"

def test_tokenizer_round_trips_text():
    tokenizer = Tokenizer.from_text("hello world")

    text = "hello"

    assert tokenizer.decode(tokenizer.encode(text)) == text