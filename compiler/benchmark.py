import time
import subprocess
import numpy as np
from pathlib import Path
import onnxruntime as ort
from ops import relu, reshape, gemm, max_pool, conv
from codegenerator import build

C_RUNS = 2000        # the C versions are fast, so run them many times
ORT_RUNS = 2000
PY_RUNS = 5          # the Python interpreter is slow, so only a few runs


def run_python(graph, image):
    # same run loop as interpreter.py, as a function so we can time it
    t = dict(graph.weights)
    t[graph.inputs[0]] = image
    for node in graph.nodes:
        x = [t[n] for n in node.inputs]
        a = node.attributes
        if node.op_type == "ConvRelu":
            r = relu(conv(x[0], x[1], x[2], a["pads"][0], a["strides"][0]))
        elif node.op_type == "Conv":
            r = conv(x[0], x[1], x[2], a["pads"][0], a["strides"][0])
        elif node.op_type == "Relu":
            r = relu(x[0])
        elif node.op_type == "MaxPool":
            r = max_pool(x[0], a["kernel_shape"][0], a["strides"][0])
        elif node.op_type == "Reshape":
            r = reshape(x[0], x[1])
        elif node.op_type == "Gemm":
            r = gemm(x[0], x[1], x[2])
        t[node.outputs[0]] = r
    return t[graph.outputs[0]]


def c_scores(exe, input_path):
    out = subprocess.run([str(exe), str(input_path)], capture_output=True, text=True, check=True).stdout
    return np.array([float(v) for v in out.split()], dtype=np.float32)


def c_time_us(exe, input_path, runs):
    out = subprocess.run([str(exe), str(input_path), str(runs)], capture_output=True, text=True, check=True).stdout
    return float(out)


def load_image(root):
    from torchvision import datasets, transforms
    test_set = datasets.MNIST(root=root / "data", train=False, download=True, transform=transforms.ToTensor())
    image, label = test_set[0]
    return image.numpy().reshape(1, 1, 28, 28).astype(np.float32), label


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    model_path = root / "models" / "mnist_cnn.onnx"
    build_dir = root / "build"

    # build two versions of the C program: original Conv and fast Conv
    graph, exe_naive = build(model_path, build_dir, fast=False, exe_name="model_naive.exe")
    _, exe_fast = build(model_path, build_dir, fast=True, exe_name="model_fast.exe")

    image, label = load_image(root)
    input_path = build_dir / "input.bin"
    image.tofile(input_path)

    # ---- correctness first: every version must match ONNX Runtime ----
    session = ort.InferenceSession(str(model_path))
    ref = session.run(None, {graph.inputs[0]: image})[0].ravel()
    results = {
        "Python interpreter": run_python(graph, image).ravel(),
        "C, original Conv": c_scores(exe_naive, input_path),
        "C, fast Conv": c_scores(exe_fast, input_path),
    }
    print("Correctness check (biggest difference vs ONNX Runtime):")
    for name, scores in results.items():
        diff = np.max(np.abs(scores - ref))
        ok = "OK" if diff < 1e-4 else "WRONG"
        print(f"  {name:<20} predicts {int(np.argmax(scores))}   diff {diff:.1e}   {ok}")
    print()

    # ---- timing ----
    t0 = time.perf_counter()
    for _ in range(PY_RUNS):
        run_python(graph, image)
    py_us = (time.perf_counter() - t0) / PY_RUNS * 1e6

    for _ in range(50):
        session.run(None, {graph.inputs[0]: image})                 # warm up
    t0 = time.perf_counter()
    for _ in range(ORT_RUNS):
        session.run(None, {graph.inputs[0]: image})
    ort_us = (time.perf_counter() - t0) / ORT_RUNS * 1e6

    naive_us = c_time_us(exe_naive, input_path, C_RUNS)
    fast_us = c_time_us(exe_fast, input_path, C_RUNS)

    print(f"{'version':<22}{'time per image':>16}{'vs Python':>12}")
    for name, us in [("Python interpreter", py_us), ("ONNX Runtime", ort_us),
                     ("C, original Conv", naive_us), ("C, fast Conv", fast_us)]:
        print(f"{name:<22}{us:>13.1f} us{py_us / us:>11.0f}x")