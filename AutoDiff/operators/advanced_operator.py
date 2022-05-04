from ..core import Node, Operator, Session, Constant
from .basic_operator import Power
import numpy as np

class Log(Operator):
    def __init__(self, operand, base):
        
        super().__init__('O::Log::' + str(Session.getGraph().numLog))
        self.operands = [operand]
        self.base = base
        Session.getGraph().numLog += 1
    
    def forward(self):
        self.value = np.log(self.operands[0].value) / np.log(self.base)
    
    def backward(self):
        return self.grad * 1 / (self.operands[0].value * np.log(self.base))

class Sin(Operator):
    def __init__(self, operand):
        
        super().__init__('O::Sin::' + str(Session.getGraph().numSin))
        self.operands = [operand]
        Session.getGraph().numSin += 1
    
    def forward(self):
        self.value = np.sin(self.operands[0].value)
    
    def backward(self):
        return np.cos(self.operands[0].value) * self.grad

class Cos(Operator):
    def __init__(self, operand):
        
        super().__init__('O::Cos::' + str(Session.getGraph().numCos))
        self.operands = [operand]
        Session.getGraph().numCos += 1
    
    def forward(self):
        self.value = np.cos(self.operands[0].value)
    
    def backward(self):
        return -np.sin(self.operands[0].value) * self.grad
    
class Max(Operator):
    def __init__(self, operand1: Node, operand2: Node):
        
        super().__init__('O::Max::' + str(Session.getGraph().numMax))
        self.operands = [operand1, operand2]
        Session.getGraph().numMax += 1
    
    def forward(self):
        self.value = np.maximum(self.operands[0].value, self.operands[1].value)

    def backward(self):
        return [self.grad * (self.operands[0].value > self.operands[1].value), self.grad * (self.operands[1].value >= self.operands[0].value)]

class Min(Operator):
    def __init__(self, operand1: Node, operand2: Node):
        
        super().__init__('O::Min::' + str(Session.getGraph().numMin))
        self.operands = [operand1, operand2]
        Session.getGraph().numMin += 1
    
    def forward(self):
        self.value = np.minimum(self.operands[0].value, self.operands[1].value)

    def backward(self):
        return [self.grad * (self.operands[0].value < self.operands[1].value), self.grad * (self.operands[1].value <= self.operands[0].value)]

class Exp(Power):
    def __init__(self, operand):
        
        super().__init__(Constant(np.e, name='e'), operand)
        self.name = 'O::Exp::' + str(Session.getGraph().numExp)
        Session.getGraph().numExp += 1

class Tan(Operator):
    def __init__(self, operand):
        
        super().__init__('O::Tan::' + str(Session.getGraph().numTan))
        self.operands = [operand]
        Session.getGraph().numTan += 1
    
    def forward(self):
        self.value = np.tan(self.operands[0].value)
    
    def backward(self):
        return self.grad / (np.cos(self.operands[0].value) ** 2)
    
class Sqrt(Power):
    def __init__(self, operand):
        
        super().__init__(operand, Constant(0.5))
        self.name = 'O::Sqrt::' + str(Session.getGraph().numSqrt)
        Session.getGraph().numSqrt += 1

class Sinh(Operator):
    def __init__(self, operand):
        
        super().__init__('O::Sinh::' + str(Session.getGraph().numSinh))
        self.operands = [operand]
        Session.getGraph().numSinh += 1
    
    def forward(self):
        self.value = np.sinh(self.operands[0].value)
    
    def backward(self):
        return np.cosh(self.operands[0].value) * self.grad

class Cosh(Operator):
    def __init__(self, operand):
        
        super().__init__('O::Cosh::' + str(Session.getGraph().numCosh))
        self.operands = [operand]
        Session.getGraph().numCosh += 1
    
    def forward(self):
        self.value = np.cosh(self.operands[0].value)
    
    def backward(self):
        return np.sinh(self.operands[0].value) * self.grad

class Tanh(Operator):
    def __init__(self, operand):
        
        super().__init__('O::Tanh::' + str(Session.getGraph().numTanh))
        self.operands = [operand]
        Session.getGraph().numTanh += 1
    
    def forward(self):
        self.value = np.tanh(self.operands[0].value)
    
    def backward(self):
        return self.grad / (np.cosh(self.operands[0].value) ** 2)

# Class wrapper for advanced operators

def log(x: Node, base = np.e):
    return Log(x, base)

def log10(x: Node):
    return Log(x, 10)

def log2(x: Node):
    return Log(x, 2)

def sin(x: Node):
    return Sin(x)

def cos(x: Node):
    return Cos(x)

def tan(x: Node):
    return Tan(x)

def maximum(x: Node, y: Node):
    return Max(x, y)

def minimum(x: Node, y: Node):
    return Min(x, y)

def exp(x: Node):
    return Exp(x)

def sqrt(x: Node):
    return Sqrt(x)

def sinh(x: Node):
    return Sinh(x)

def cosh(x: Node):
    return Cosh(x)

def tanh(x: Node):
    return Tanh(x)