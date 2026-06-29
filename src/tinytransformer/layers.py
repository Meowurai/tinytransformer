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