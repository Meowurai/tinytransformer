# src/tinytransformer/linear.py

import random 

from tinytransformer.autograd import Value

class Linear:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = [
            [Value(random.uniform(-0.1, 0.1))
            for _ in range(input_size)]
            for _ in range(output_size)
        ]

        self.bias = [
            Value(random.uniform(-0.1, 0.1))
            for _ in range(output_size)
        ]

    def __call__(self, inputs: list[Value]) -> list[Value]:
        outputs = []
        for output_index, weights in enumerate(self.weights):
            weighted_sum = Value(0.0)
            for input_value, weight in zip(inputs, weights):
                weighted_sum += weight * input_value 
            
            output = weighted_sum + self.bias[output_index]
            outputs.append(output)
        
        return outputs

    def parameters(self) -> list[Value]:
        return [
            weight
            for row in self.weights
            for weight in row
        ] + self.bias
    

class LayerNorm:
    def __call__(self, vector: list[Value]) -> list[Value]:
        average = sum(vector, start=Value(0.0)) / Value(len(vector))
        centered = [value - average for value in vector]
        squared = [value ** 2 for value in centered]
        avg_squared = sum(squared, start=Value(0.0)) / Value(len(squared))
        std = avg_squared ** 0.5
        normalized = [value / std for value in centered]
        
        return normalized


class MLP:
    def __init__(self, embedding_size: int) -> None:
        self.up = Linear(embedding_size, embedding_size * 4)
        self.down = Linear(embedding_size * 4, embedding_size)

    def __call__(self, vector: list[Value]) -> list[Value]:
        up_vector = self.up(vector)
        activations = [value.tanh() for value in up_vector]
        return self.down(activations)
    
    def parameters(self) -> list[Value]:
        return (
            self.up.parameters()
            + self.down.parameters()
        )


