import onnx
from onnx import numpy_helper
from pathlib import Path




class Node:
    def __init__(self, op_type, inputs, outputs, attributes):
        self.op_type = op_type          # string, e.g. "Conv"
        self.inputs = inputs            # list of tensor names
        self.outputs = outputs          # list of tensor names
        self.attributes = attributes    # dict, e.g. {"strides": [1, 1]}


class Graph:
    def __init__(self, nodes, weights, inputs, outputs):
        self.nodes = nodes              # list of Node objects, in run order
        self.weights = weights          # dict: name -> numpy array
        self.inputs = inputs            # list of names, e.g. ["images"]
        self.outputs = outputs          # list of names, e.g. ["logits"]



def from_onnx(path):
    model = onnx.load(path)
    g = model.graph

    weights = {}
    for init in g.initializer:
        weights[init.name] = numpy_helper.to_array(init)


    nodes = []
    for n in g.node:
        attrs = {}
        for a in n.attribute:
            attrs[a.name] = onnx.helper.get_attribute_value(a)
        nodes.append(Node(n.op_type, list(n.input), list(n.output), attrs))


    inputs = [i.name for i in g.input if i.name not in weights]
    outputs = [o.name for o in g.output]


    return Graph(nodes, weights, inputs, outputs)







##TEST CASE

if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "models" / "mnist_cnn.onnx"
    graph = from_onnx(model_path)

    print("inputs:", graph.inputs)
    print("outputs:", graph.outputs)
    print("weights:", len(graph.weights))
    for node in graph.nodes:
        print(node.op_type, node.inputs, node.outputs, node.attributes)