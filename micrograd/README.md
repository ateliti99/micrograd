## Overview

This repository contains an implementation of a simple automatic differentiation engine and a neural network library built on top of it. The code is divided into two main components:
1. **Engine**: A module for automatic differentiation using reverse mode.
2. **Neural Network Library**: A module for creating and training neural networks, including classes for neurons, layers, and multi-layer perceptrons (MLPs).

## Files

### Engine

The `engine.py` file contains the implementation of the `Value` class, which represents a scalar value and its gradient for automatic differentiation.

#### Key Features:
- **Basic Operations**: Addition, subtraction, multiplication, division, and power operations with gradient calculation.
- **Activation Functions**: Tanh and ReLU activation functions.
- **Backward Propagation**: A method to perform backward propagation to compute gradients.

### Neural Network Library

The `nn.py` file contains the implementation of a simple neural network library, including classes for neurons, layers, and MLPs.

#### Key Features:
- **Neuron**: A class representing a single neuron with support for different weight initialization methods.
- **Layer**: A class representing a layer of neurons.
- **MLP (Multi-Layer Perceptron)**: A class representing a multi-layer perceptron neural network.

## Usage

### Engine

To use the `Value` class from the engine:

```python
from engine import Value

# Create Value objects
a = Value(2.0)
b = Value(3.0)

# Perform operations
c = a + b
d = a * b
e = a - b
f = a ** 2
g = a / b

# Activation functions
h = a.tanh()
i = a.relu()

# Perform backward propagation
g.backward()

# Print gradients
print(a.grad)
print(b.grad)
```

### Neural Network Library

To create and use a neural network:

```python
from nn import MLP, Value

# Define the architecture: 2 inputs, 1 hidden layer with 4 neurons, 1 output
n_inputs = 2
n_outputs = [4, 1]
mlp = MLP(n_inputs, n_outputs)

# Create input data
x = [Value(1.0), Value(2.0)]

# Perform a forward pass
output = mlp(x)

# Perform backward propagation
output.backward()

# Print gradients of the parameters
for param in mlp.parameters():
    print(param, param.grad)
```

## Classes and Methods

### Engine

#### `Value`
- **Attributes**:
  - `data`: The scalar value.
  - `grad`: The gradient of the value.
- **Methods**:
  - `__add__`: Addition operation.
  - `__mul__`: Multiplication operation.
  - `__sub__`: Subtraction operation.
  - `__pow__`: Power operation.
  - `__truediv__`: Division operation.
  - `tanh`: Tanh activation function.
  - `relu`: ReLU activation function.
  - `backward`: Perform backward propagation.

### Neural Network Library

#### `Neuron`
- **Attributes**:
  - `weights`: The weights of the neuron.
  - `bias`: The bias of the neuron.
- **Methods**:
  - `__call__`: Perform a forward pass.
  - `parameters`: Return the parameters (weights and bias) of the neuron.
  - `initialize_weights_uniform`: Initialize weights uniformly.
  - `initialize_bias_uniform`: Initialize bias uniformly.
  - `initialize_weights_xavier`: Initialize weights using Xavier initialization.
  - `initialize_weights_he`: Initialize weights using He initialization.

#### `Layer`
- **Attributes**:
  - `layer`: A list of neurons in the layer.
- **Methods**:
  - `__call__`: Perform a forward pass for all neurons in the layer.
  - `parameters`: Return the parameters of all neurons in the layer.

#### `MLP`
- **Attributes**:
  - `layers`: A list of layers in the MLP.
- **Methods**:
  - `__call__`: Perform a forward pass for all layers in the MLP.
  - `parameters`: Return the parameters of all layers in the MLP.
  - `zero_grad`: Reset the gradients of all parameters to zero.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributions

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## Contact

For any questions or inquiries, please open an issue in the repository.