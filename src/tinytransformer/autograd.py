# src/tinytransformer/autograd.py

class Value:
    def __init__(
        self,
        data: float,
        _parents=(),
        _op=""
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._prev = set(_parents)
        self._op = _op
        self._backward = lambda: None 

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"