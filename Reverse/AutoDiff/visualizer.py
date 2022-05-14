from graphviz import Digraph
from .core import *

NAME_DICT = {
    'O::Log::': 'log [',
    'O::Sin::': 'sin [',
    'O::Cos::': 'cos [',
    'O::Add::': '+ [',
    'O::Mul::': '* [',
    'O::Div::': '/ [',
    'O::Pow::': '^ [',
    'O::Matmul::': '@ [',
    'O::Max::': 'max [',
    'O::Min::': 'min [',
    'O::Exp::': 'exp [',
    'O::Tan::': 'tan [',
    'O::Sqrt::': 'sqrt [',
    'O::Cosh::': 'cosh [',
    'O::Tanh::': 'tanh [',
    'O::Sinh::': 'sinh [',
    '::': ' [',
}

def labelWrapper(label: str):
    for (key, value) in NAME_DICT.items():
        if key in label:
            return label.replace(key, value) + ']'
    return label

def drawGraph(node: Node):
    g = Digraph()
    g.attr(rankdir='LR', size='10, 8')
    g.attr('node', shape='circle')

    order = topoSort(node)

    for node in order:
        shape = 'box' if isinstance(node, Placeholder) else 'circle'
        label = labelWrapper(node.name)
        g.node(label, label=label, shape=shape)
    for node in order:
        if isinstance(node, Operator):
            for edge in node.operands:
                nodeLabel = labelWrapper(node.name)
                edgeLabel = labelWrapper(edge.name)
                g.edge(edgeLabel, nodeLabel, label=str("{:.4f}".format(edge.value)))
    return g