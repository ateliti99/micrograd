import math

class Value():
    def __init__(self, data: float, _children = set(), _op = '') -> None:
        self.data = float(data)
        self._children = _children
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
         
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, exp):
        out = Value(self.data ** exp, (self, ), 'pow')

        def _backward():
            self.grad += ((exp) * (self.data ** (exp - 1))) * out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        x = self.data
        out = Value(0 if x <= 0 else x, (self, ), "relu")

        def _backward():
            self.grad += (0 if x <= 0 else 1) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo_sort = []; topo_sort.append(self)
        visited = set()
        def topo_sorting(node):
            for child in node._children:
                if child not in visited:
                    topo_sort.append(child)
                    visited.add(child)
            for child in node._children:
                topo_sorting(child)
            return
        topo_sorting(self)

        self.grad = 1.0

        for node in topo_sort:
            node._backward()