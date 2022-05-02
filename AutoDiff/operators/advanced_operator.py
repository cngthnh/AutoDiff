from ..core import Node, Operator, Session
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
        
        super().__init__('O::Sin::' + str(Session.getGraph().numLog))
        self.operands = [operand]
        Session.getGraph().numSin += 1
    
    def forward(self):
        self.value = np.sin(self.operands[0].value)
    
    def backward(self):
        return np.cos(self.operands[0].value) * self.grad

class Cos(Operator):
    def __init__(self, operand):
        
        super().__init__('O::Cos::' + str(Session.getGraph().numLog))
        self.operands = [operand]
        Session.getGraph().numSin += 1
    
    def forward(self):
        self.value = np.cos(self.operands[0].value)
    
    def backward(self):
        return -np.sin(self.operands[0].value) * self.grad

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