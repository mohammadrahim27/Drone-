import sys
import numpy as np
from pathlib import Path
import onnxruntime as ort
from codegenerator import build
import fair_bench_comparison as fb

RUNS = 30            # MobileNetV2 takes milliseconds, not microseconds, so fewer runs


def make_input(root, shape):
    # a real photo would be nicer, but for checking the compiler any image works:
    # we compare our output against ONNX Runtime on the exact same input
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape).astype(np.float32)


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    name = sys.argv[1] if len(sys.argv) > 1 else "mobilenet_v2.onnx"
    model_path = root / "models" / name
    build_dir = root / "build"

    print(f"Compiling {name} ...")
    graph, exe = build(model_path, build_dir, exe_name="mobilenet.exe")
    print(f"  {len(graph.nodes)} nodes after fusion, {len(graph.weights):,} weight tensors")

    image = make_input(root, graph.shapes[graph.inputs[0]])
    input_path = build_dir / "mobilenet_input.bin"
    image.tofile(input_path)

    # ---- correctness: compare every one of the 1000 scores with ONNX Runtime ----
    fb.RUNS = RUNS
    ours_us, ours = fb.run_ours(exe, input_path)
    ref = ort.InferenceSession(str(model_path)).run(None, {graph.inputs[0]: image})[0].ravel()

    rel_err = np.max(np.abs(ours - ref)) / np.max(np.abs(ref))
    top5_ours, top5_ref = np.argsort(-ours)[:5], np.argsort(-ref)[:5]
    print()
    print(f"Biggest difference vs ONNX Runtime: {np.max(np.abs(ours - ref)):.2e} "
          f"({rel_err * 100:.4f}% of the largest score)")
    print(f"Top-5 classes, ours:         {top5_ours.tolist()}")
    print(f"Top-5 classes, ONNX Runtime: {top5_ref.tolist()}")
    print("Correct:", "YES" if rel_err < 1e-3 and (top5_ours == top5_ref).all() else "NO")

    # ---- speed: both timed in C, same stopwatch ----
    # laptops are noisy: run each 3 times and keep the best
    ort_exe = fb.build_ort_bench(graph, fb.get_onnxruntime(root / "tools"), build_dir)
    ours_us = min([ours_us] + [fb.run_ours(exe, input_path)[0] for _ in range(2)])
    ort1_us = min(fb.run_ort(ort_exe, model_path, input_path, graph, threads=1)[0] for _ in range(3))
    ortn_us = min(fb.run_ort(ort_exe, model_path, input_path, graph, threads=0)[0] for _ in range(3))

    print()
    print(f"{'version (best of 3)':<30}{'time per image':>16}")
    print(f"{'Our compiler (1 core)':<30}{ours_us / 1000:>13.2f} ms")
    print(f"{'ONNX Runtime (1 core)':<30}{ort1_us / 1000:>13.2f} ms")
    print(f"{'ONNX Runtime (all cores)':<30}{ortn_us / 1000:>13.2f} ms")
    r = ort1_us / ours_us
    print()
    print(f"Same number of cores: " + (f"our compiler is {r:.2f}x faster" if r >= 1
                                       else f"ONNX Runtime is {1 / r:.2f}x faster"))