import random
from .engine import Value

class Neuron:
    
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        out = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        out = out.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:

    def __init__(self, n_outputs, n_inputs):
        self.n_outputs = n_outputs
        self.layer = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.layer]
        return outs
    
    def parameters(self):
        return [p for n in self.layer for p in n.parameters()]
    
class MLP:

    def __init__(self, n_layers, n_inputs, n_outputs):
        self.layers = [Layer(n_outputs, n_inputs) for _ in range(n_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [layer.parameters() for layer in self.layers]