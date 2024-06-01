import random
from engine import Value

class Neuron:
    
    def __init__(self, n_inputs) -> None:
        self.weights = [Value(random.uniform(-1, 1)) for _ in n_inputs]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        out = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        out = out.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]