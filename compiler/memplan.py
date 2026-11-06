import math
from pathlib import Path
from ir import from_onnx, infer_shapes
from passes import fuse_conv_relu

BYTES_PER_FLOAT = 4   # float32


def size_in_bytes(shape):
    return math.prod(shape) * BYTES_PER_FLOAT


def find_lifetimes(graph):
    # for each tensor the model creates: (first step it exists, last step it's needed)
    lifetimes = {}
    last = len(graph.nodes) - 1

    for name in graph.inputs:
        lifetimes[name] = [0, 0]                  # the image exists from the start

    for step, node in enumerate(graph.nodes):
        for name in node.inputs:
            if name in lifetimes:                 # skip weights, they live in their own memory
                lifetimes[name][1] = step
        for name in node.outputs:
            lifetimes[name] = [step, step]

    for name in graph.outputs:
        lifetimes[name][1] = last                 # final answer must survive to the end

    return lifetimes


def plan_memory(graph):
    lifetimes = find_lifetimes(graph)
    buffers = []          # buffers[i] = size in bytes of buffer i
    free = []             # indices of buffers nobody is using right now
    assignment = {}       # tensor name -> buffer index

    def get_buffer(nbytes):
        # reuse the smallest free buffer that's big enough; otherwise make a new one
        fits = [b for b in free if buffers[b] >= nbytes]
        if fits:
            best = min(fits, key=lambda b: buffers[b])
            free.remove(best)
            return best
        buffers.append(nbytes)
        return len(buffers) - 1

    for name in graph.inputs:
        assignment[name] = get_buffer(size_in_bytes(graph.shapes[name]))

    for step, node in enumerate(graph.nodes):
        # 1. give this step's output a buffer (inputs are still in use, so they can't be reused yet)
        for name in node.outputs:
            assignment[name] = get_buffer(size_in_bytes(graph.shapes[name]))

        # 2. anything whose last use was this step is now done: hand its buffer back
        for name, (start, end) in lifetimes.items():
            if end == step and name not in graph.outputs and name in assignment:
                free.append(assignment[name])

    return assignment, buffers, lifetimes


def check_plan(assignment, lifetimes):
    # safety check: two tensors that are alive at the same time must never share a buffer
    names = list(assignment)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if assignment[a] != assignment[b]:
                continue
            sa, ea = lifetimes[a]
            sb, eb = lifetimes[b]
            if sa <= eb and sb <= ea:
                raise AssertionError(f"{a} and {b} share buffer {assignment[a]} while both alive")


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "models" / "mnist_cnn.onnx"
    graph = from_onnx(model_path)
    infer_shapes(graph, (1, 1, 28, 28))
    fuse_conv_relu(graph)

    assignment, buffers, lifetimes = plan_memory(graph)
    check_plan(assignment, lifetimes)

    print(f"{'tensor':<14}{'shape':<18}{'bytes':>8}   alive   buffer")
    for name, buf in assignment.items():
        shape = graph.shapes[name]
        start, end = lifetimes[name]
        print(f"{name:<14}{str(shape):<18}{size_in_bytes(shape):>8}   {start}-{end}     B{buf}")

    naive = sum(size_in_bytes(graph.shapes[n]) for n in assignment)
    planned = sum(buffers)
    print()
    print("Buffers:", [f"B{i}: {b} bytes" for i, b in enumerate(buffers)])
    print(f"Without planning: {naive:,} bytes")
    print(f"With planning:    {planned:,} bytes")
    print(f"Saved:            {100 * (naive - planned) / naive:.0f}%")