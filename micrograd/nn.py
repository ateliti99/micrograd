import random
import math
from .engine import Value

class LengthMismatchError(Exception):
    """Exception raised for errors in the input length."""
    def __init__(self, message="The lengths of x and self.weights do not match."):
        self.message = message
        super().__init__(self.message)

class Module:
    
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def update(self, learning_rate):
        for p in self.parameters():
            p.data -= learning_rate * p.grad

class Neuron(Module):
    
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)] # to be reinitialize when generating layer or MLP
        self.bias = Value(random.uniform(-1, 1)) # to be reinitialize when generating layer or MLP

    def __call__(self, x, activation_function="tanh"):
        if len(x) != len(self.weights):
            raise LengthMismatchError 
        out = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        
        activation_functions = {
            "tanh": self.tanh,
            "relu": self.relu
        }

        out = activation_functions[activation_function](out)

        return out
    
    def relu(self, value):
        return value.relu()
    
    def tanh(self, value):
        return value.tanh()

    def parameters(self):
        return self.weights + [self.bias]
    
    def he_initialization(self, n_inputs):
        a = math.sqrt(6 / n_inputs)
        for weight in self.weights:
            weight.data = random.uniform(-a, a)

class Layer(Module):

    def __init__(self, n_outputs, n_inputs):
        self.layer = [Neuron(n_inputs) for _ in range(n_outputs)]
        # He initialization
        for neuron in self.layer:
            neuron.he_initialization(n_inputs)

    def __call__(self, x, **kargs):
        outs = [neuron(x, **kargs) for neuron in self.layer]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.layer for p in n.parameters()]
    
    def he_initialization(self, n_inputs, n_outputs):
        a, b = math.sqrt(6/n_inputs), math.sqrt(6/n_outputs)
        he_parameters = [Value(random.uniform(a,b)) for _ in range(len(self.parameters))]
        for p, he_p in zip(self.parameters(), he_parameters):
            p = he_p    
    
class MLP(Module):

    def __init__(self, n_inputs, n_outputs):
        sz = [n_inputs] + n_outputs
        self.layers = [Layer(sz[i+1], sz[i]) for i in range(len(n_outputs))]

    def __call__(self, x, **kargs):
        for layer in self.layers:
            x = layer(x, **kargs)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]