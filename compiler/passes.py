from ir import Node


def count_uses(graph, name):
    # how many nodes read this tensor (plus 1 if it's a graph output)
    uses = sum(name in n.inputs for n in graph.nodes)
    if name in graph.outputs:
        uses += 1
    return uses


def clip_bounds(node, graph):
    # Clip's min/max are inputs in new ONNX versions, attributes in old ones
    lo, hi = node.attributes.get("min", float("-inf")), node.attributes.get("max", float("inf"))
    if len(node.inputs) > 1 and node.inputs[1]:
        lo = float(graph.weights[node.inputs[1]])
    if len(node.inputs) > 2 and node.inputs[2]:
        hi = float(graph.weights[node.inputs[2]])
    return lo, hi


def fuse_conv_activation(graph):
    """Merge Conv + ReLU into ConvRelu, and Conv + Clip (ReLU6) into ConvClip."""
    nodes = graph.nodes
    new_nodes = []
    i = 0

    while i < len(nodes):
        node = nodes[i]
        nxt = nodes[i + 1] if i + 1 < len(nodes) else None

        can_fuse = (
            node.op_type == "Conv"
            and nxt is not None
            and nxt.op_type in ("Relu", "Clip")
            and nxt.inputs[0] == node.outputs[0]
            and count_uses(graph, node.outputs[0]) == 1   # only the activation uses it
        )

        if can_fuse:
            attrs = dict(node.attributes)
            if nxt.op_type == "Relu":
                op = "ConvRelu"
            else:
                op = "ConvClip"
                attrs["clip_min"], attrs["clip_max"] = clip_bounds(nxt, graph)
            fused = Node(op, node.inputs, nxt.outputs, attrs)
            fused.shape = nxt.shape
            new_nodes.append(fused)
            i += 2          # skip both the Conv and the activation
        else:
            new_nodes.append(node)
            i += 1

    graph.nodes = new_nodes
    return graph


# old name, so interpreter.py / memplan.py keep working
fuse_conv_relu = fuse_conv_activation