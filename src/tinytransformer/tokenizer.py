# src/tinytransformer/tokenizer.py


class Tokenizer:
    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {
            id: token
            for token, id in token_to_id.items()
        }

    @classmethod
    def from_text(cls, text: str) -> "Tokenizer":
        tokens = [*sorted(set(text))]

        token_to_id = {
            token: token_id
            for token_id, token in enumerate(tokens)
        }

        return cls(token_to_id)
    
    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id)
    
    def encode(self, text: str) -> list[int]:
        try:
            return [self.token_to_id[token] for token in text]
        except KeyError as error:
            raise ValueError(f"Unrecognized token {error.args[0]!r}") from error
        
    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.id_to_token[id] for id in ids)
        except KeyError as error:
            raise ValueError(f"Unrecognized id {error.args[0]}") from error
        
    