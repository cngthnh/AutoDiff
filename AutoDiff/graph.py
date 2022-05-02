class Graph():
    
    def __init__(self):
        self.variables = set()
        self.placeholders = set()
        self.operators = set()
        self.constants = set()
        self.numAdd = 0
        self.numMul = 0
        self.numDiv = 0
        self.numPow = 0
        self.numMatmul = 0
        self.numLog = 0
        self.numSin = 0
        self.numCos = 0
    
    def clear(self):
        self.__init__()
    
    def destroy(self):
        for v in self.variables:
            del v
