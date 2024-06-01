from micrograd.engine import Value

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
    e = c - d
    print(e)
    print(e._children)
    print(e._op)
    print()

    e.backward()
    print(e.grad)
    print(a.grad)
    print(b.grad)
    print(c.grad)
    print(d.grad)



if __name__ == "__main__":
    test1()