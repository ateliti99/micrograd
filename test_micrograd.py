from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

def test1():
    # Generate objects
    a = Value(data=10)
    print(a)
    b = Value(data=2)
    print(b)
    print()

    # Test operations
    print(a+b)
    print(a-b)
    print(a*b)
    print(a/b)
    print()

    # Check children and operation
    c = a * b
    print(c)
    print(c._children)
    print(c._op)
    print()

    # Test nested case
    d = Value(4)
    e = c - d + 1
    print(e)
    print(e._children)
    print(e._op)
    print()

    e.backward()

def test2():
    n1 = Neuron(2)
    x = [1, 1]
    print(n1(x))
    print(n1.parameters())

def test3():
    l1 = Layer(2, 3)
    x = [1,1,1]
    print(l1(x))
    print(l1.parameters())

def test4():
    nn1 = MLP(3, [2, 3, 1])
    x = [1, 1, 1]
    print(nn1(x))
    print(nn1.parameters())

def test5():
    nn = MLP(3, [2, 3, 1])
    x = [1, 1, 1]
    out = nn(x)
    print(f"Output: {out}\n")
    grads = [p.grad for params in nn.parameters() for p in params]
    print(f"Grads: {grads}")
    out.backward()
    grads = [p.grad for params in nn.parameters() for p in params]
    print(f"Grads: {grads}")

def test6():
    epochs = 500
    lr = 0.01

    nn = MLP(3, [2, 3, 1])
    x = [1, 1, 1]

    for _ in range(epochs):
        # Forward
        out = nn(x)
        print(out)
        # Compute loss
        loss = (out - 34) ** 2
        # Zero grad
        for layer_parms in nn.parameters():
            for parms in layer_parms:
                parms.grad = 0
        # Backward
        loss.backward()
        # Update weights and bias
        for layer_parms in nn.parameters():
            for parms in layer_parms:
                parms.data = parms.data - lr * parms.grad

    print(out)

if __name__ == "__main__":
    test6()