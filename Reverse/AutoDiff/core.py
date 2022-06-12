from .session import Session
import numpy as np

class Node():
    """
    General class for all nodes in the graph.
    """

    def __init__(self, value, grad):
        
        if Session.getGraph() is None:
            raise ValueError("No graph found. Please create a session first.")
        self.value = value
        self.grad = grad
        self.name = None

    def __repr__(self):
        return type(self).__name__ + '(' + ', '.join([str(self.name), str(self.value), str(self.grad)]) + ')'
    
    def __str__(self):
        return str(self.__repr__())

    def compute(self, feed_dict=None):

        order = topoSort(self)
        fp = forwardPass(order, feed_dict)
        backwardPass(order)

        return fp

class Variable(Node):
    def __init__(self, value, name = None, _transpose = None):
        
        super().__init__(value, None)
        if isinstance(value, list):
            self.value = np.array(value)

        if isinstance(self.value, np.ndarray) and _transpose is None:
            self.T = Variable(np.transpose(self.value), _transpose=self)
        else:
            if isinstance(self.value, np.ndarray):
                self.T = _transpose
            else:
                self.T = None

        if name is None:
            self.name = 'V::' + str(Session.getGraph().variables.__len__())
        else:
            self.name = name
        Session.getGraph().variables.add(self)
    
    def updateValue(self, value):
        self.value = value
        if (self.T is not None):
            self.T.value = np.transpose(self.value)
    
class Constant(Node):
    def __init__(self, value, name = None):
        
        super().__init__(value, None)

        if name is None:
            self.name = 'C::' + str(Session.getGraph().constants.__len__())
        else:
            self.name = name
        Session.getGraph().constants.add(self)
    
class Placeholder(Node):
    def __init__(self, name):
        
        super().__init__(None, None)

        if name is None:
            self.name = 'P::' + str(Session.getGraph().placeholders.__len__())
        else:
            self.name = name
        Session.getGraph().placeholders.add(self)
    
    def setValue(self, value):
        if isinstance(value, list):
            self.value = np.array(value)
        else:
            self.value = value
        if isinstance(self.value, np.ndarray):
            self.T = Variable(self.name + '.T')
            self.T.value = np.transpose(self.value)
            self.T.T = self
        else:
            self.T = None

class Operator(Node):

    def __init__(self, name = None):
        
        super().__init__(None, None)
        self.operands = []
        
        if name is None:
            self.name = 'O::' + str(Session.getGraph().operators.__len__())
        else:
            self.name = name
        Session.getGraph().operators.add(self)
    
    def __repr__(self):
        return type(self).__name__ + '(' + str(self.name) + ', ' + str(self.operands) + ')'

def nodeWrapper(other):
    
    if isinstance(other, Node):
        return other
    if isinstance(other, int) or isinstance(other, float):
        return Constant(other)
    raise TypeError('AutoDiff::Graph - Unsupported type: ' + str(type(other)))

def topoSort(head):
    """
    This function is used to topologically sort the graph.
    """

    vis = set()
    ordering = []
    
    
    def _dfs(node):
        if node not in vis:
            vis.add(node)
            if isinstance(node, Operator):
                for operand in node.operands:
                    _dfs(operand)
            ordering.append(node)
            
    if head is None:
        for node in Session.getGraph().operators:
            _dfs(node)
    else:
        _dfs(head)
        
    return ordering

def forwardPass(order, feed_dict={}):
    """
    This function is used to compute the forward pass of the graph.
    """

    for node in order:
        
        if isinstance(node, Placeholder):
            node.setValue(feed_dict[node.name])
                    
        elif isinstance(node, Operator):
            node.forward()

    return order[-1].value

def backwardPass(order):
    """
    This function is used to compute the backward pass of the graph.
    """

    vis = set()
    order[-1].grad = 1
    for node in reversed(order):
        if isinstance(node, Operator):
            grads = node.backward()
            if not isinstance(grads, np.ndarray):
                grads = np.array(grads, ndmin=1, dtype=object)
            for operand, grad in zip(np.array(node.operands, ndmin=1, dtype=object), grads):
                if operand not in vis:
                    operand.grad = grad
                else:
                    operand.grad += grad
                vis.add(operand)