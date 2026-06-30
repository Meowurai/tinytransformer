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
            other.grad += out.grad

        out._backward = _backward

        return out 
    
    def __mul__(self, other: Value) -> Value:
        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 

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
    
    def __neg__(self):
        return self * Value(-1.0)
    
    def __sub__(self, other: Value):
        return self + (-other)
    
    def __pow__(self, exponent: int | float) -> Value:
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out
    
    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out
    
    def tanh(self) -> Value:
        x = self.data
        t = math.tanh(x)

        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward

        return out
    
    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out
    

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for parent in v._prev:
                    build_topo(parent)
                
                topo.append(v)
            
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()