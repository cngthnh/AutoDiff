from .core import Dual, nodeWrapper
import numpy as np

def add(a: Dual, b: Dual):
    value = a.value + b.value
    derivative = a.derivative + b.derivative
    return Dual(value, derivative)

def mul(a: Dual, b: Dual):
    value = a.value * b.value
    derivative = a.derivative * b.value + a.value * b.derivative
    return Dual(value, derivative)

def div(a: Dual, b: Dual):
    value = a.value / b.value
    derivative = (b.value * a.derivative - a.value * b.derivative) / b.value ** 2
    return Dual(value, derivative)

def power(a: Dual, b: Dual):
    value = a.value ** b.value
    derivative = b.value * a.value ** (b.value - 1) * a.derivative
    return Dual(value, derivative)

Dual.__add__ = lambda self, other: add(self, nodeWrapper(other))
Dual.__sub__ = lambda self, other: add(self, -nodeWrapper(other))
Dual.__mul__ = lambda self, other: mul(self, nodeWrapper(other))
Dual.__truediv__ = lambda self, other: div(self, nodeWrapper(other))
Dual.__pow__ = lambda self, other: power(self, nodeWrapper(other))

def maximum(a, b):
    a = nodeWrapper(a)
    b = nodeWrapper(b)
    value = np.max([a.value, b.value])
    derivative = a.derivative if a.value > b.value else b.derivative
    return Dual(value, derivative)

def log(a):
    a = nodeWrapper(a)
    value = np.log(a.value)
    derivative = a.derivative / a.value
    return Dual(value, derivative)

def exp(a):
    a = nodeWrapper(a)
    value = np.exp(a.value)
    derivative = a.derivative * value
    return Dual(value, derivative)