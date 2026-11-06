from ir import Node


def count_uses(graph, name):
    # how many nodes read this tensor (plus 1 if it's a graph output)
    uses = sum(name in n.inputs for n in graph.nodes)
    if name in graph.outputs:
        uses += 1
    return uses


def fuse_conv_relu(graph):
    nodes = graph.nodes
    new_nodes = []
    i = 0

    while i < len(nodes):
        node = nodes[i]
        nxt = nodes[i + 1] if i + 1 < len(nodes) else None

        can_fuse = (
            node.op_type == "Conv"
            and nxt is not None
            and nxt.op_type == "Relu"
            and nxt.inputs[0] == node.outputs[0]
            and count_uses(graph, node.outputs[0]) == 1   # only the ReLU uses it
        )

        if can_fuse:
            fused = Node("ConvRelu", node.inputs, nxt.outputs, node.attributes)
            fused.shape = nxt.shape
            new_nodes.append(fused)
            i += 2          # skip both the Conv and the ReLU
        else:
            new_nodes.append(node)
            i += 1

    graph.nodes = new_nodes
    return graph