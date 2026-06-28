# src/tinytransformer/autograd.py

import math

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
    
    def __mul__(self, other: Value) -> Value:
        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.grad * out.grad 
            other.grad += self.grad * out.grad 

        out._backward = _backward

        return out
    
    def __truediv__(self, other: Value) -> Value:
        out = Value(
            self.data / other.data,
            (self, other),
            "/"
        )

        def _backward():
            self.grad += 1 / other.data * out.grad
            other.grad += -self.data / (other.data ** 2) * out.grad
        
        out._backward = _backward

        return out
    
    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out