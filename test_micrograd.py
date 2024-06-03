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
        self.assertEqual(c._op, "*")
        
        # Test nested operations
        d = Value(4)
        e = c - d + 1
        self.assertEqual(e.data, 17)
        self.assertEqual(e._op, "+")

        # Perform backpropagation
        e.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertIsNotNone(c.grad)
        self.assertIsNotNone(d.grad)
        self.assertIsNotNone(e.grad)

    def test_mlp(self):
        # Test MLP with 3 inputs and layers of sizes 2, 3, and 1
        nn1 = MLP(3, [2, 3, 1])
        x = [1, 1, 1]
        output = nn1(x)
        self.assertIsInstance(output, Value)

    def test_mlp_backward(self):
        # Test forward and backward propagation for MLP
        nn = MLP(3, [2, 3, 1])
        x = [1, 1, 1]
        out = nn(x)
        self.assertIsInstance(out, Value)
        
        # Perform backpropagation
        out.backward()
        
        # Check gradients
        for p in nn.parameters():
                self.assertIsNotNone(p.grad)

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
            nn.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Update weights and biases
            # for p in nn.parameters():
            #         p.data -= lr * p.grad
            nn.update()

        self.assertAlmostEqual(out.data, 34, places=1)

if __name__ == "__main__":
    unittest.main()
