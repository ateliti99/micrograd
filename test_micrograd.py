import unittest
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

class TestMicrograd(unittest.TestCase):

    def test_value_operations(self):
        # Generate Value objects
        a = Value(data=10)
        b = Value(data=2)
        
        # Test basic arithmetic operations
        self.assertEqual((a + b).data, 12)
        self.assertEqual((a - b).data, 8)
        self.assertEqual((a * b).data, 20)
        self.assertAlmostEqual((a / b).data, 5.0)
        
        # Check children and operation for multiplication
        c = a * b
        self.assertEqual(c.data, 20)
        self.assertIn(a, c._children)
        self.assertIn(b, c._children)
        self.assertEqual(c._op, "*")
        
        # Test nested operations
        d = Value(4)
        e = c - d + 1
        self.assertEqual(e.data, 17)
        self.assertIn(c, e._children)
        self.assertIn(d, e._children)
        self.assertEqual(e._op, "+")

        # Perform backpropagation
        e.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertIsNotNone(c.grad)
        self.assertIsNotNone(d.grad)
        self.assertIsNotNone(e.grad)

    def test_neuron(self):
        # Test Neuron with 2 inputs
        n1 = Neuron(2)
        x = [1, 1]
        output = n1(x)
        self.assertIsInstance(output, Value)
        self.assertEqual(len(n1.parameters()), 3)

    def test_layer(self):
        # Test Layer with 3 neurons each taking 2 inputs
        l1 = Layer(2, 3)
        x = [1, 1]
        output = l1(x)
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), 3)
        self.assertEqual(len(l1.parameters()), 3)

    def test_mlp(self):
        # Test MLP with 3 inputs and layers of sizes 2, 3, and 1
        nn1 = MLP(3, [2, 3, 1])
        x = [1, 1, 1]
        output = nn1(x)
        self.assertIsInstance(output, Value)
        self.assertEqual(len(nn1.parameters()), 3)

    def test_mlp_backward(self):
        # Test forward and backward propagation for MLP
        nn = MLP(3, [2, 3, 1])
        x = [1, 1, 1]
        out = nn(x)
        self.assertIsInstance(out, Value)
        
        # Perform backpropagation
        out.backward()
        
        # Check gradients
        for params in nn.parameters():
            for param in params:
                self.assertIsNotNone(param.grad)

    def test_training(self):
        # Hyperparameters
        epochs = 500
        lr = 0.01

        # Initialize MLP with 3 inputs and layers of sizes 2, 3, and 1
        nn = MLP(3, [2, 3, 1])
        x = [1, 1, 1]

        # Training loop
        for _ in range(epochs):
            # Forward pass
            out = nn(x)

            # Compute loss (mean squared error)
            loss = (out - 34) ** 2
            
            # Zero gradients
            for layer_params in nn.parameters():
                for param in layer_params:
                    param.grad = 0
            
            # Backward pass
            loss.backward()
            
            # Update weights and biases
            for layer_params in nn.parameters():
                for param in layer_params:
                    param.data -= lr * param.grad

        self.assertAlmostEqual(out.data, 34, places=1)

if __name__ == "__main__":
    unittest.main()
