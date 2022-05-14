from ..core import Node, Operator, Variable, Placeholder, Constant, nodeWrapper, Session
import numpy as np

class Add(Operator):
    def __init__(self, x, y):
        
        super().__init__("O::Add::" + str(Session.getGraph().numAdd))
        Session.getGraph().numAdd += 1
        self.operands = [x, y]

    def forward(self):
        self.value = self.operands[0].value + self.operands[1].value
    
    def backward(self):
        result = [self.grad, self.grad]
        if ((isinstance(self.operands[0], Variable) or isinstance(self.operands[0], Placeholder))
        and self.operands[0].T is not None 
        and self.operands[1] == self.operands[0].T):
            return 2 * np.array(result, dtype=object)
        return result

class Multiply(Operator):
    def __init__(self, x, y):
        
        super().__init__("O::Mul::" + str(Session.getGraph().numMul))
        Session.getGraph().numMul += 1
        self.operands = [x, y]

    def forward(self):
        self.value = self.operands[0].value * self.operands[1].value
    
    def backward(self):
        result = [self.operands[1].value * self.grad, self.operands[0].value * self.grad]
        if ((isinstance(self.operands[0], Variable) or isinstance(self.operands[0], Placeholder))
        and self.operands[0].T is not None 
        and self.operands[1] == self.operands[0].T):
            return 2 * np.array(result, dtype=object)
        return result

class Matmul(Operator):
    def __init__(self, x, y):
        
        super().__init__("O::Matmul::" + str(Session.getGraph().numMatmul))
        Session.getGraph().numMatmul += 1
        self.operands = [x, y]

    def forward(self):
        self.value = np.array(self.operands[0].value) @ np.array(self.operands[1].value)
    
    def backward(self):
        result = [np.array(self.operands[1].value).T * np.array(self.grad), np.array(self.operands[0].value).T * np.array(self.grad)]
        if ((isinstance(self.operands[0], Variable) or isinstance(self.operands[0], Placeholder))
        and self.operands[0].T is not None 
        and self.operands[1] == self.operands[0].T):
            return 2 * np.array(result, dtype=object)
        return result

class Divide(Operator):
    def __init__(self, x, y):
        
        super().__init__("O::Div::" + str(Session.getGraph().numDiv))
        Session.getGraph().numDiv += 1
        self.operands = [x, y]

    def forward(self):
        self.value = self.operands[0].value / self.operands[1].value
    
    def backward(self):
        result = [self.grad / self.operands[1].value, -self.operands[0].value * self.grad / self.operands[1].value**2]
        if ((isinstance(self.operands[0], Variable) or isinstance(self.operands[0], Placeholder))
        and self.operands[0].T is not None 
        and self.operands[1] == self.operands[0].T):
            return 0 * np.array(result, dtype=object)
        return result

class Power(Operator):
    def __init__(self, x, y):
        
        super().__init__("O::Pow::" + str(Session.getGraph().numPow))
        Session.getGraph().numPow += 1
        self.operands = [x, y]

    def forward(self):
        self.value = self.operands[0].value ** self.operands[1].value
    
    def backward(self):
        if ((isinstance(self.operands[0], Variable) or isinstance(self.operands[0], Placeholder))
        and self.operands[0].T is not None 
        and self.operands[1] == self.operands[0].T):
            return [self.operands[0] ** self.operands[1].value * (np.log(np.abs(self.operands[0].value)) + 1), self.operands[0] ** self.operands[1].value * (np.log(np.abs(self.operands[0].value)) + 1)]
        return [self.operands[1].value * self.grad * (self.operands[0].value ** (self.operands[1].value - 1)), 
            (self.operands[0].value ** self.operands[1].value) * np.log(np.abs(self.operands[0].value)) * self.grad]

Node.__add__ = lambda self, other: Add(self, nodeWrapper(other))
Node.__truediv__ = lambda self, other: Divide(self, nodeWrapper(other))
Node.__sub__ = lambda self, other: Add(self, Constant(-1) * nodeWrapper(other))
Node.__mul__ = lambda self, other: Multiply(self, nodeWrapper(other))
Node.__matmul__ = lambda self, other: Matmul(self, nodeWrapper(other))
Node.__pow__ = lambda self, other: Power(self, nodeWrapper(other))
Node.__neg__ = lambda self: Multiply(self, Constant(-1))