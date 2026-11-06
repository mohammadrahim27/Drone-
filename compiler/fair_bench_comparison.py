import os
import sys
import shutil
import tarfile
import zipfile
import subprocess
import urllib.request
import numpy as np
from pathlib import Path
from codegenerator import build

ORT_VERSION = "1.20.1"
RUNS = 5000

# A C program that runs the model through ONNX Runtime's C library and times it
# with the SAME stopwatch our generated code uses. No Python in the timed loop.
ORT_BENCH_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime_c_api.h"

#ifdef _WIN32
#include <windows.h>
static double now_us(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}
#endif

static const OrtApi *ort;

static void check(OrtStatus *s)
{
    if (s) { printf("ONNX Runtime error: %s\n", ort->GetErrorMessage(s)); exit(1); }
}

/* usage: ort_bench model.onnx input.bin input_name output_name runs threads */
int main(int argc, char **argv)
{
    if (argc < 7) { printf("usage: ort_bench model input in_name out_name runs threads\n"); return 1; }
    int runs = atoi(argv[5]), threads = atoi(argv[6]);

    static float input[INPUT_SIZE], output[OUTPUT_SIZE];
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(input, sizeof(float), INPUT_SIZE, f) != INPUT_SIZE) { printf("bad input file\n"); return 1; }
    fclose(f);

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv *env;
    check(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env));

    OrtSessionOptions *opts;
    check(ort->CreateSessionOptions(&opts));
    check(ort->SetIntraOpNumThreads(opts, threads));      /* 0 = use all cores */
    check(ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

    OrtSession *session;
#ifdef _WIN32
    wchar_t wpath[1024];
    mbstowcs(wpath, argv[1], 1024);
    check(ort->CreateSession(env, wpath, opts, &session));
#else
    check(ort->CreateSession(env, argv[1], opts, &session));
#endif

    /* input and output tensors point straight at our arrays (no copying each run) */
    OrtMemoryInfo *mem;
    check(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));
    int64_t in_shape[] = IN_SHAPE, out_shape[] = OUT_SHAPE;
    OrtValue *in_tensor = NULL, *out_tensor = NULL;
    check(ort->CreateTensorWithDataAsOrtValue(mem, input, sizeof(input), in_shape, IN_DIMS,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_tensor));
    check(ort->CreateTensorWithDataAsOrtValue(mem, output, sizeof(output), out_shape, OUT_DIMS,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &out_tensor));

    const char *in_names[] = { argv[3] }, *out_names[] = { argv[4] };

    for (int r = 0; r < 20; r++)                                   /* warm up */
        check(ort->Run(session, NULL, in_names, (const OrtValue *const *)&in_tensor, 1,
                       out_names, 1, &out_tensor));

    double t0 = now_us();
    for (int r = 0; r < runs; r++)
        check(ort->Run(session, NULL, in_names, (const OrtValue *const *)&in_tensor, 1,
                       out_names, 1, &out_tensor));
    double t1 = now_us();

    printf("%.3f\n", (t1 - t0) / runs);                           /* line 1: microseconds per run */
    for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.7f\n", output[i]);   /* then the scores */
    return 0;
}
"""


def get_onnxruntime(tools_dir):
    # download Microsoft's prebuilt ONNX Runtime C library once, into tools/
    if sys.platform == "win32":
        name, ext = f"onnxruntime-win-x64-{ORT_VERSION}", ".zip"
    else:
        name, ext = f"onnxruntime-linux-x64-{ORT_VERSION}", ".tgz"
    folder = tools_dir / name
    if folder.exists():
        return folder

    tools_dir.mkdir(exist_ok=True)
    url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{name}{ext}"
    archive = tools_dir / (name + ext)
    print("Downloading ONNX Runtime C library (one time only)...")
    urllib.request.urlretrieve(url, archive)
    if ext == ".zip":
        zipfile.ZipFile(archive).extractall(tools_dir)
    else:
        tarfile.open(archive).extractall(tools_dir, filter="data")
    archive.unlink()
    return folder


def build_ort_bench(graph, ort_dir, build_dir):
    in_shape = graph.shapes[graph.inputs[0]]
    out_shape = graph.shapes[graph.outputs[0]]
    src = (ORT_BENCH_C
           .replace("INPUT_SIZE", str(int(np.prod(in_shape))))
           .replace("OUTPUT_SIZE", str(int(np.prod(out_shape))))
           .replace("IN_SHAPE", "{" + ", ".join(map(str, in_shape)) + "}")
           .replace("OUT_SHAPE", "{" + ", ".join(map(str, out_shape)) + "}")
           .replace("IN_DIMS", str(len(in_shape)))
           .replace("OUT_DIMS", str(len(out_shape))))
    (build_dir / "ort_bench.c").write_text(src)

    exe = build_dir / "ort_bench.exe"
    if sys.platform == "win32":
        # link straight against the DLL, and copy it next to the exe so Windows finds it
        dll = ort_dir / "lib" / "onnxruntime.dll"
        shutil.copy(dll, build_dir / "onnxruntime.dll")
        cmd = ["gcc", "-O3", "-I", str(ort_dir / "include"), "-o", str(exe),
               str(build_dir / "ort_bench.c"), str(dll)]
    else:
        lib = ort_dir / "lib"
        cmd = ["gcc", "-O3", "-I", str(ort_dir / "include"), "-o", str(exe),
               str(build_dir / "ort_bench.c"), "-L", str(lib), "-lonnxruntime",
               f"-Wl,-rpath,{lib}"]
    subprocess.run(cmd, check=True)
    return exe


def run_ort(exe, model_path, input_path, graph, threads):
    out = subprocess.run([str(exe), str(model_path), str(input_path), graph.inputs[0],
                          graph.outputs[0], str(RUNS), str(threads)],
                         capture_output=True, text=True, check=True).stdout.split()
    return float(out[0]), np.array([float(v) for v in out[1:]], dtype=np.float32)


def run_ours(exe, input_path):
    t = float(subprocess.run([str(exe), str(input_path), str(RUNS)],
                             capture_output=True, text=True, check=True).stdout)
    scores = subprocess.run([str(exe), str(input_path)], capture_output=True, text=True, check=True).stdout
    return t, np.array([float(v) for v in scores.split()], dtype=np.float32)


def load_image(root):
    from torchvision import datasets, transforms
    test_set = datasets.MNIST(root=root / "data", train=False, download=True, transform=transforms.ToTensor())
    image, label = test_set[0]
    return image.numpy().reshape(1, 1, 28, 28).astype(np.float32)


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    model_path = root / "models" / "mnist_cnn.onnx"
    build_dir = root / "build"

    graph, our_exe = build(model_path, build_dir, fast=True, exe_name="model_fast.exe")
    ort_dir = get_onnxruntime(root / "tools")
    ort_exe = build_ort_bench(graph, ort_dir, build_dir)

    input_path = build_dir / "input.bin"
    load_image(root).tofile(input_path)

    ours_us, ours_scores = run_ours(our_exe, input_path)
    ort1_us, ort1_scores = run_ort(ort_exe, model_path, input_path, graph, threads=1)
    ortn_us, _ = run_ort(ort_exe, model_path, input_path, graph, threads=0)

    print()
    print("Both timed in C, same stopwatch, same image, "
          f"{RUNS:,} runs each (no Python in the timed loop)")
    print(f"Scores agree: biggest difference {np.max(np.abs(ours_scores - ort1_scores)):.1e}")
    print()
    print(f"{'version':<36}{'time per image':>15}")
    print(f"{'Our compiler (1 core)':<36}{ours_us:>12.1f} us")
    print(f"{'ONNX Runtime (1 core)':<36}{ort1_us:>12.1f} us")
    print(f"{'ONNX Runtime (all cores)':<36}{ortn_us:>12.1f} us")
    print()
    ratio = ort1_us / ours_us
    if ratio >= 1:
        print(f"Same number of cores: our compiler is {ratio:.2f}x faster than ONNX Runtime")
    else:
        print(f"Same number of cores: ONNX Runtime is {1 / ratio:.2f}x faster than our compiler")