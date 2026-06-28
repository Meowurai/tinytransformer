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
    
    def __add__(self, other: Value) -> Value:
        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += out.grad
            out.grad += self.grad

        out._backward = _backward

        return out 