class Dual:
    def __init__(self, value, derivative):
        self.value = value
        self.derivative = derivative
    def __neg__(self):
        return Dual(-self.value, -self.derivative)
    def __str__(self):
        return f"({self.value}, {self.derivative})"

class Variable(Dual):
    def __init__(self, value, respective=False):
        super().__init__(value, int(respective))

def nodeWrapper(other):
    
    if isinstance(other, Dual):
        return other
    if isinstance(other, int) or isinstance(other, float):
        return Variable(other)
    raise TypeError('AutoDiff::Graph - Unsupported type: ' + str(type(other)))