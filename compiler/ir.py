import math
import onnx
from onnx import numpy_helper
from pathlib import Path


class Node:
    def __init__(self, op_type, inputs, outputs, attributes):
        self.op_type = op_type          # string, e.g. "Conv"
        self.inputs = inputs            # list of tensor names
        self.outputs = outputs          # list of tensor names
        self.attributes = attributes    # dict, e.g. {"strides": [1, 1]}
        self.shape = None               # output shape, filled in by infer_shapes


class Graph:
    def __init__(self, nodes, weights, inputs, outputs):
        self.nodes = nodes              # list of Node objects, in run order
        self.weights = weights          # dict: name -> numpy array
        self.inputs = inputs            # list of names, e.g. ["images"]
        self.outputs = outputs          # list of names, e.g. ["logits"]
        self.shapes = {}                # dict: tensor name -> shape tuple


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


# ---------- shape inference ----------

def window_out(size, k, pad_begin, pad_end, stride):
    # out = (in + padding - window) / stride + 1
    return (size + pad_begin + pad_end - k) // stride + 1


def infer_shapes(graph, input_shape):
    shapes = graph.shapes

    # known shapes to start: the image, and every weight
    shapes[graph.inputs[0]] = tuple(input_shape)
    for name, w in graph.weights.items():
        shapes[name] = tuple(w.shape)

    for node in graph.nodes:
        x = shapes[node.inputs[0]]
        attrs = node.attributes

        if node.op_type == "Conv":
            w = shapes[node.inputs[1]]               # (out_ch, in_ch, kh, kw)
            pads = attrs.get("pads", [0, 0, 0, 0])   # [top, left, bottom, right]
            strides = attrs.get("strides", [1, 1])
            h = window_out(x[2], w[2], pads[0], pads[2], strides[0])
            wd = window_out(x[3], w[3], pads[1], pads[3], strides[1])
            out = (x[0], w[0], h, wd)

        elif node.op_type == "Relu":
            out = x                                   # same shape

        elif node.op_type == "MaxPool":
            k = attrs["kernel_shape"]
            pads = attrs.get("pads", [0, 0, 0, 0])
            strides = attrs.get("strides", k)
            h = window_out(x[2], k[0], pads[0], pads[2], strides[0])
            wd = window_out(x[3], k[1], pads[1], pads[3], strides[1])
            out = (x[0], x[1], h, wd)

        elif node.op_type == "Reshape":
            target = [int(v) for v in graph.weights[node.inputs[1]]]
            total = math.prod(x)
            # 0 means "copy this dimension" (unless allowzero is set)
            if attrs.get("allowzero", 0) == 0:
                target = [x[i] if v == 0 else v for i, v in enumerate(target)]
            # -1 means "whatever is left over"
            if -1 in target:
                known = math.prod(v for v in target if v != -1)
                target[target.index(-1)] = total // known
            out = tuple(target)

        elif node.op_type == "Gemm":
            b = shapes[node.inputs[1]]
            m = x[1] if attrs.get("transA", 0) else x[0]
            n = b[0] if attrs.get("transB", 0) else b[1]
            out = (m, n)

        else:
            raise ValueError("No shape rule for op: " + node.op_type)

        node.shape = out
        shapes[node.outputs[0]] = out

    return graph


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "models" / "mnist_cnn.onnx"
    graph = from_onnx(model_path)
    infer_shapes(graph, (1, 1, 28, 28))

    for node in graph.nodes:
        print(node.op_type, "->", node.outputs[0], node.shape)