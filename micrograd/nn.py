import random
from .engine import Value

class LengthMismatchError(Exception):
    """Exception raised for errors in the input length."""
    def __init__(self, message="The lengths of x and self.weights do not match."):
        self.message = message
        super().__init__(self.message)

class Neuron:
    
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        if len(x) != len(self.weights):
            raise LengthMismatchError 
        out = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        out = out.relu()
        return out

    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:

    def __init__(self, n_outputs, n_inputs):
        self.n_outputs = n_outputs
        self.layer = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.layer]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.layer for p in n.parameters()]
    
class MLP:

    def __init__(self, n_inputs, n_outputs):
        sz = [n_inputs] + n_outputs
        self.layers = [Layer(sz[i+1], sz[i]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [layer.parameters() for layer in self.layers]