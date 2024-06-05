import unittest
import math
from unittest.mock import patch
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP, LengthMismatchError

# Unit tests
class TestNeuralNetwork(unittest.TestCase):

    @patch('random.uniform', lambda a, b: 0.5)
    def test_neuron_initialization(self):
        n = Neuron(4)
        self.assertEqual(len(n.weights), 4)
        print(n.bias.data)
        self.assertEqual(n.bias.data, 0.5)

    @patch('random.uniform', lambda a, b: 0.5)
    def test_neuron_call(self):
        n = Neuron(3)
        x = [Value(1), Value(2), Value(3)]
        result = n(x, activation_function=Value.relu)
        self.assertEqual(result.data, max(0, 3.5))  # relu(0.5 * 1 + 0.5 * 2 + 0.5 * 3 + 0.5)

    def test_neuron_length_mismatch(self):
        n = Neuron(3)
        x = [Value(1), Value(2)]
        with self.assertRaises(LengthMismatchError):
            n(x)

    def test_layer_initialization(self):
        l = Layer(2, 3)
        self.assertEqual(len(l.layer), 2)
        self.assertEqual(l.layer[0].n_inputs, 3)

    def test_mlp_initialization(self):
        mlp = MLP(3, [2, 2])
        self.assertEqual(len(mlp.layers), 2)
        self.assertEqual(len(mlp.layers[0].layer), 2)
        self.assertEqual(len(mlp.layers[1].layer), 2)

    @patch('random.uniform', lambda a, b: 0.5)
    def test_mlp_call(self):
        mlp = MLP(3, [2, 2])
        x = [Value(1), Value(2), Value(3)]
        result = mlp(x)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    @patch('random.uniform', lambda a, b: 0.5)
    def test_zero_grad(self):
        mlp = MLP(3, [2, 2])
        mlp.zero_grad()
        for p in mlp.parameters():
            self.assertEqual(p.grad, 0)

    @patch('random.uniform', lambda a, b: 0.5)
    def test_update(self):
        mlp = MLP(3, [2, 2])
        for p in mlp.parameters():
            p.grad = 1
        mlp.update(0.1)
        for p in mlp.parameters():
            self.assertEqual(p.data, 0.4)

    def test_backward_propagation(self):
        # Test addition backward
        a = Value(3)
        b = Value(4)
        c = a + b
        c.backward()
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)

        # Test multiplication backward
        x = Value(2)
        y = Value(3)
        z = x * y
        z.backward()
        self.assertEqual(x.grad, 3)
        self.assertEqual(y.grad, 2)

        # Test subtraction backward
        m = Value(5)
        n = Value(2)
        p = m - n
        p.backward()
        self.assertEqual(m.grad, 1)
        self.assertEqual(n.grad, -1)

        # Test power backward
        u = Value(2)
        v = u ** 3
        v.backward()
        self.assertEqual(u.grad, 12)  # derivative of u^3 with respect to u is 3*u^2 = 3*2^2 = 12

    @patch('random.uniform', lambda a, b: 0.5)  # Mocking random.uniform
    def test_gradient_update(self):
        # Create a simple MLP
        mlp = MLP(2, [3, 1])

        # Set gradients manually for testing
        for parameter in mlp.parameters():
            parameter.grad = 0.1

        # Update gradients
        mlp.update(learning_rate=0.01)

        # Check if gradients are updated correctly
        for parameter in mlp.parameters():
            self.assertAlmostEqual(parameter.data, 0.5 - 0.01 * 0.1, places=5)

    def test_mlp_learns_sin_function(self):
        # Create dataset
        def sin_function(x):
            return math.sin(x)

        # Generate training data
        data = [(Value(x), Value(sin_function(x))) for x in range(-5, 6)]
        
        # Initialize MLP
        mlp = MLP(1, [10, 1])
        
        # Training parameters
        epochs = 1000
        learning_rate = 0.01
        
        for epoch in range(epochs):
            total_loss = Value(0.0)
            for x, y in data:
                # Forward pass
                y_pred = mlp([x])
                
                # Loss calculation (Mean Squared Error)
                loss = (y_pred - y) ** 2
                total_loss += loss
                
                # Zero gradients
                mlp.zero_grad()
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                mlp.update(learning_rate)
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss.data}')
        
        # Test if the network learned the function
        for x, y in data:
            y_pred = mlp([x])
            self.assertAlmostEqual(y_pred.data, y.data, delta=0.2)

if __name__ == '__main__':
    unittest.main()