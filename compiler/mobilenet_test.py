import os
import sys
import subprocess
import numpy as np
from pathlib import Path
import onnxruntime as ort
from codegenerator import build
import fair_bench_comparison as fb

RUNS = 30            # MobileNetV2 takes milliseconds, not microseconds, so fewer runs
THREADS = [1, 2, 4, 6, 8, 12]
SCHEDULES = ["static", "guided", "dynamic,4"]


def time_ours(exe, input_path, threads, schedule=None):
    # our program uses OpenMP: OMP_NUM_THREADS sets how many cores it may use,
    # OMP_SCHEDULE sets how the work is shared between them
    env = dict(os.environ)
    if schedule:
        env["OMP_SCHEDULE"] = schedule
    else:
        env.pop("OMP_SCHEDULE", None)
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    else:
        env.pop("OMP_NUM_THREADS", None)
    out = subprocess.run([str(exe), str(input_path), str(RUNS)], capture_output=True,
                         text=True, check=True, env=env).stdout
    return float(out)


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
    print()
    print("Thread sweep for our compiler, ms per image (best of 3 each):")
    print("  threads " + "".join(f"{sc:>12}" for sc in SCHEDULES))
    sweep = {}
    for t in THREADS:
        row = []
        for sc in (SCHEDULES if t > 1 else ["static"]):
            sweep[(t, sc)] = min(time_ours(exe, input_path, t, sc) for _ in range(3))
            row.append(f"{sweep[(t, sc)] / 1000:>12.2f}")
        print(f"  {t:>7} " + "".join(row))
    best = min(sweep, key=sweep.get)
    best_t = f"{best[0]} threads, {best[1]}"
    print(f"  fastest: {best_t}")

    # bake the winning setting into the program, then time it with NO settings at all
    print(f"\nRebuilding with {best_t} as the built-in default ...")
    graph, exe = build(model_path, build_dir, exe_name="mobilenet.exe", threads=best[0], schedule=best[1])
    tuned_us = min(time_ours(exe, input_path, 0, None) for _ in range(3))
    print(f"  tuned program, no settings: {tuned_us / 1000:.2f} ms")
    ours1_us, oursn_us = sweep[(1, "static")], tuned_us
    ort1_us = min(fb.run_ort(ort_exe, model_path, input_path, graph, threads=1)[0] for _ in range(3))
    ortn_us = min(fb.run_ort(ort_exe, model_path, input_path, graph, threads=0)[0] for _ in range(3))

    print()
    print(f"{'version (best of 3)':<30}{'1 core':>12}{'multi-core':>12}")
    print(f"{'Our compiler':<30}{ours1_us / 1000:>9.2f} ms{oursn_us / 1000:>9.2f} ms")
    print(f"{'ONNX Runtime':<30}{ort1_us / 1000:>9.2f} ms{ortn_us / 1000:>9.2f} ms")
    print()
    print(f"(multi-core: ours tuned to {best_t}, ONNX Runtime with its own choice)")
    for label, ours, theirs in [("1 core", ours1_us, ort1_us), ("multi-core", oursn_us, ortn_us)]:
        r = theirs / ours
        print(f"{label + ':':<12}" + (f"our compiler is {r:.2f}x faster" if r >= 1
                                     else f"ONNX Runtime is {1 / r:.2f}x faster"))