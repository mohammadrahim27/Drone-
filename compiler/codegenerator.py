import math
import subprocess
import numpy as np
from pathlib import Path
from ir import from_onnx, infer_shapes
from passes import fuse_conv_relu
from memplan import plan_memory, check_plan


# ---------- the C kernels: one hand-written function per operation ----------

KERNELS = r"""
#include <string.h>
#include <float.h>

/* portable SIMD: gcc turns v8 math into AVX on x86 and NEON on ARM (Jetson) */
typedef float v8 __attribute__((vector_size(32)));
static inline v8 LOADV(const float *p) { v8 v; memcpy(&v, p, sizeof v); return v; }
static inline void STOREV(float *p, v8 v) { memcpy(p, &v, sizeof v); }
static inline v8 SPLAT(float f) { v8 v = {f, f, f, f, f, f, f, f}; return v; }

/* Conv (optionally with ReLU fused in).
   x: input  [C][H][W]      w: weights [K][C][kh][kw]      b: bias [K]
   y: output [K][OH][OW]                                                   */
static void conv2d(const float *x, int C, int H, int W,
                   const float *w, const float *b, int K, int kh, int kw,
                   int pad_t, int pad_l, int stride, int relu,
                   float *y, int OH, int OW)
{
    for (int k = 0; k < K; k++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                float sum = b[k];
                for (int c = 0; c < C; c++)
                    for (int i = 0; i < kh; i++) {
                        int ih = oh * stride + i - pad_t;
                        if (ih < 0 || ih >= H) continue;          /* padding = 0 */
                        for (int j = 0; j < kw; j++) {
                            int iw = ow * stride + j - pad_l;
                            if (iw < 0 || iw >= W) continue;
                            sum += x[(c * H + ih) * W + iw] *
                                   w[((k * C + c) * kh + i) * kw + j];
                        }
                    }
                if (relu && sum < 0.0f) sum = 0.0f;
                y[(k * OH + oh) * OW + ow] = sum;
            }
}

/* Faster Conv. Same math, different loop order:
   - for each filter weight, sweep it across a whole output row at once
   - the innermost loop walks memory in a straight line, so the CPU can use
     SIMD instructions (several multiply-adds in one instruction)
   - edge checks are worked out once per row instead of once per multiply   */
static void conv2d_fast(const float *x, int C, int H, int W,
                        const float *w, const float *b, int K, int kh, int kw,
                        int pad_t, int pad_l, int stride, int relu,
                        float *y, int OH, int OW)
{
    for (int k = 0; k < K; k++) {
        float *yk = y + k * OH * OW;
        for (int p = 0; p < OH * OW; p++) yk[p] = b[k];            /* start at bias */

        for (int c = 0; c < C; c++) {
            const float *xc = x + c * H * W;
            for (int i = 0; i < kh; i++)
                for (int j = 0; j < kw; j++) {
                    float wv = w[((k * C + c) * kh + i) * kw + j];
                    int off = j - pad_l;

                    /* only the columns where this weight lands inside the image */
                    int lo = 0, hi = OW;
                    while (lo < OW && lo * stride + off < 0) lo++;
                    while (hi > lo && (hi - 1) * stride + off >= W) hi--;

                    for (int oh = 0; oh < OH; oh++) {
                        int ih = oh * stride + i - pad_t;
                        if (ih < 0 || ih >= H) continue;
                        const float *xrow = xc + ih * W;
                        float *yrow = yk + oh * OW;
                        if (stride == 1)
                            for (int ow = lo; ow < hi; ow++) yrow[ow] += wv * xrow[ow + off];
                        else
                            for (int ow = lo; ow < hi; ow++) yrow[ow] += wv * xrow[ow * stride + off];
                    }
                }
        }
        if (relu)
            for (int p = 0; p < OH * OW; p++) if (yk[p] < 0.0f) yk[p] = 0.0f;
    }
}

/* pick which Conv the model uses: gcc -DFAST_CONV picks the fast one */
#ifdef FAST_CONV
#define CONV conv2d_fast
#else
#define CONV conv2d
#endif

/* MaxPool: keep the biggest value in each k x k window */
static void maxpool2d(const float *x, int C, int H, int W,
                      int k, int stride, float *y, int OH, int OW)
{
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                float m = -FLT_MAX;
                for (int i = 0; i < k; i++)
                    for (int j = 0; j < k; j++) {
                        float v = x[(c * H + oh * stride + i) * W + ow * stride + j];
                        if (v > m) m = v;
                    }
                y[(c * OH + oh) * OW + ow] = m;
            }
}

/* Gemm: y[n] = b[n] + sum over k of x[k] * W(n, k)
   transB = 1 means W is stored as [N][K], transB = 0 means [K][N]          */
static void gemm(const float *x, int K, const float *w, const float *b,
                 int N, int transB, float *y)
{
    for (int n = 0; n < N; n++) {
        float sum = b[n];
        for (int k = 0; k < K; k++)
            sum += x[k] * (transB ? w[n * K + k] : w[k * N + n]);
        y[n] = sum;
    }
}

/* ReLU on its own (only used if a ReLU wasn't fused into a Conv) */
static void relu(const float *x, int n, float *y)
{
    for (int i = 0; i < n; i++) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
"""


MAIN_C = r"""
#include <stdio.h>
#include <stdlib.h>

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

void model_run(const float *input, float *output);

/* usage:  model input.bin          -> prints the 10 scores
           model input.bin 1000     -> runs 1000 times, prints average microseconds per run */
int main(int argc, char **argv)
{
    if (argc < 2) { printf("usage: model input.bin [runs]\n"); return 1; }

    static float input[INPUT_SIZE], output[OUTPUT_SIZE];
    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("could not open %s\n", argv[1]); return 1; }
    if (fread(input, sizeof(float), INPUT_SIZE, f) != INPUT_SIZE) {
        printf("input file is the wrong size\n"); return 1;
    }
    fclose(f);

    if (argc >= 3) {
        int runs = atoi(argv[2]);
        for (int r = 0; r < 20; r++) model_run(input, output);      /* warm up */
        double t0 = now_us();
        for (int r = 0; r < runs; r++) model_run(input, output);
        double t1 = now_us();
        printf("%.3f\n", (t1 - t0) / runs);
        return 0;
    }

    model_run(input, output);
    for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.7f\n", output[i]);
    return 0;
}
"""


def cf(v):
    # a valid C float literal: 0.5 -> "0.5f", 0 -> "0.0f", 1e-05 -> "1e-05f"
    t = f"{float(v):.9g}"
    if "." not in t and "e" not in t:
        t += ".0"
    return t + "f"


def c_name(name):
    # turn "conv_stack.0.weight" into a valid C name: "W_conv_stack_0_weight"
    return "W_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def emit_weights(graph, names):
    lines = []
    for name in names:
        arr = graph.weights[name].astype(np.float32).ravel()
        values = ", ".join(cf(v) for v in arr)
        lines.append(f"static const float {c_name(name)}[{arr.size}] = {{{values}}};")
    return "\n".join(lines)


VEC = 8   # floats per SIMD vector (8 x 32-bit = 256 bits)


def emit_conv_special(idx, node, graph):
    """Write a Conv function made for ONE layer, fully unrolled by the compiler.

    The idea: for P neighbouring output pixels at once, keep all K running sums
    in CPU registers as SIMD vectors (8 filters per vector). For each filter
    tap we load the K weights once and reuse them for all P pixels.
    Everything is written out as separate named variables (a0_0, a0_1, ...)
    so gcc can't spill them back to memory.
    """
    x, w_name, b_name = node.inputs[0], node.inputs[1], node.inputs[2]
    _, C, H, W = graph.shapes[x]
    _, K, OH, OW = graph.shapes[node.outputs[0]]
    w = graph.weights[w_name].astype(np.float32)          # [K][C][kh][kw]
    _, _, KH, KW = w.shape
    a = node.attributes
    pt, pl, pb, pr = a.get("pads", [0, 0, 0, 0])
    S = a.get("strides", [1, 1])[0]
    HP, WP = H + pt + pb, W + pl + pr
    relu = node.op_type == "ConvRelu"

    V = (K + VEC - 1) // VEC          # vectors needed to hold K filters
    KP = V * VEC                      # K rounded up to a multiple of 8 (extra filters are zero)

    # weights rearranged to [C][kh][kw][KP] so the filters for one tap sit together
    packed = np.zeros((C, KH, KW, KP), np.float32)
    packed[..., :K] = w.transpose(1, 2, 3, 0)
    bias = np.zeros(KP, np.float32)
    bias[:K] = graph.weights[b_name]

    P = max(1, 12 // V)               # pixels per tile, so P*V sums fit in registers
    P = min(P, OW)

    def tile(n):
        # code that computes n output pixels starting at column ow0
        L = []
        for p in range(n):
            for v in range(V):
                L.append(f"            v8 a{p}_{v} = LOADV(&BIAS{idx}[{v * VEC}]);")
        L.append(f"            for (int t = 0; t < {C * KH * KW}; t++) {{")
        L.append(f"                const int c = t / {KH * KW}, i = (t / {KW}) % {KH}, j = t % {KW};")
        L.append(f"                const float *wk = &WP{idx}[t * {KP}];")
        L.append(f"                const float *xr = &PAD{idx}[(c * {HP} + oh * {S} + i) * {WP} + ow0 * {S} + j];")
        for v in range(V):
            L.append(f"                v8 w{v} = LOADV(wk + {v * VEC});")
        for p in range(n):
            L.append(f"                v8 x{p} = SPLAT(xr[{p * S}]);")
            for v in range(V):
                L.append(f"                a{p}_{v} += x{p} * w{v};")
        L.append("            }")
        for p in range(n):
            for v in range(V):
                L.append(f"            STOREV(tmp, a{p}_{v});")
                L.append(f"            for (int k = 0; k < {VEC}; k++) if ({v * VEC} + k < {K}) {{")
                L.append(f"                float r = tmp[k];" + (" if (r < 0.0f) r = 0.0f;" if relu else ""))
                L.append(f"                y[({v * VEC} + k) * {OH * OW} + oh * {OW} + ow0 + {p}] = r;")
                L.append("            }")
        return "\n".join(L)

    main_end = OW - OW % P
    tail = OW % P
    code = [f"""
/* {node.op_type} layer {idx}: {C}x{H}x{W} -> {K}x{OH}x{OW}, {KH}x{KW} filter, stride {S}
   tile = {P} output pixels x {KP} filters ({P * V} vector sums kept in registers) */
static float PAD{idx}[{C * HP * WP}];                           /* zero-padded copy of the input */
static const float WP{idx}[{packed.size}] = {{{", ".join(cf(v) for v in packed.ravel())}}};
static const float BIAS{idx}[{KP}] = {{{", ".join(cf(v) for v in bias)}}};

static void conv_layer{idx}(const float *x, float *y)
{{
    float tmp[{VEC}];
    for (int c = 0; c < {C}; c++)                                  /* copy input inside the border */
        for (int h = 0; h < {H}; h++)
            memcpy(&PAD{idx}[(c * {HP} + h + {pt}) * {WP} + {pl}], &x[(c * {H} + h) * {W}],
                   {W} * sizeof(float));

    for (int oh = 0; oh < {OH}; oh++) {{
        for (int ow0 = 0; ow0 < {main_end}; ow0 += {P}) {{
{tile(P)}
        }}"""]
    if tail:
        code.append(f"""        {{   /* last {tail} column(s) */
            const int ow0 = {main_end};
{tile(tail)}
        }}""")
    code.append("    }\n}\n")
    return "\n".join(code)



def emit_gemm_special(idx, node, graph):
    """Write a Gemm function for ONE layer. Weights are rearranged to [K][N] so
    that for each input value we update all N output scores at once, using
    SIMD vectors kept in registers (same trick as the Conv)."""
    x, w_name, b_name = node.inputs[0], node.inputs[1], node.inputs[2]
    K = graph.shapes[x][1]
    N = graph.shapes[node.outputs[0]][1]
    w = graph.weights[w_name].astype(np.float32)
    if not node.attributes.get("transB", 0):
        w = w.T                                            # make it [N][K]
    V = (N + VEC - 1) // VEC
    NP = V * VEC
    packed = np.zeros((K, NP), np.float32)
    packed[:, :N] = w.T                                    # -> [K][NP]
    bias = np.zeros(NP, np.float32)
    bias[:N] = graph.weights[b_name]

    L = [f"""
/* Gemm layer {idx}: {K} inputs -> {N} outputs, all {N} outputs updated together */
static const float GW{idx}[{packed.size}] = {{{", ".join(cf(v) for v in packed.ravel())}}};
static const float GB{idx}[{NP}] = {{{", ".join(cf(v) for v in bias)}}};

static void gemm_layer{idx}(const float *x, float *y)
{{
    float tmp[{NP}];"""]
    for v in range(V):
        L.append(f"    v8 a{v} = LOADV(&GB{idx}[{v * VEC}]);")
    L.append(f"    for (int k = 0; k < {K}; k++) {{")
    L.append("        v8 xk = SPLAT(x[k]);")
    for v in range(V):
        L.append(f"        a{v} += xk * LOADV(&GW{idx}[k * {NP} + {v * VEC}]);")
    L.append("    }")
    for v in range(V):
        L.append(f"    STOREV(&tmp[{v * VEC}], a{v});")
    L.append(f"    memcpy(y, tmp, {N} * sizeof(float));\n}}\n")
    return "\n".join(L)


def generate_c(graph, assignment, buffers, kernel="special"):
    numel = lambda name: math.prod(graph.shapes[name])
    buf = lambda name: f"B{assignment[name]}"

    # only Conv/Gemm weights get emitted (Reshape's shape constant isn't needed at runtime)
    used = []
    for node in graph.nodes:
        if node.op_type in ("Conv", "ConvRelu", "Gemm"):
            for name in node.inputs[1:]:
                if kernel == "special":
                    continue                      # written packed, inside the layer's own function
                if name not in used:
                    used.append(name)

    out = ["/* generated by codegenerator.py -- do not edit by hand */", KERNELS]
    out.append("/* ---------- weights ---------- */")
    out.append(emit_weights(graph, used))

    if kernel == "special":
        out.append("\n/* ---------- one specialized function per Conv / Gemm layer ---------- */")
        for idx, node in enumerate(graph.nodes):
            if node.op_type in ("Conv", "ConvRelu"):
                out.append(emit_conv_special(idx, node, graph))
            elif node.op_type == "Gemm":
                out.append(emit_gemm_special(idx, node, graph))

    out.append("\n/* ---------- buffers from the memory planner ---------- */")
    for i, nbytes in enumerate(buffers):
        out.append(f"static float B{i}[{nbytes // 4}];   /* {nbytes} bytes */")

    body = []
    in_name, out_name = graph.inputs[0], graph.outputs[0]
    body.append(f"    memcpy({buf(in_name)}, input, {numel(in_name)} * sizeof(float));")

    for node in graph.nodes:
        x, y = node.inputs[0], node.outputs[0]
        xs, ys = graph.shapes[x], graph.shapes[y]
        a = node.attributes
        body.append(f"\n    /* {node.op_type}: {x} {xs} -> {y} {ys} */")

        if node.op_type in ("Conv", "ConvRelu") and kernel == "special":
            idx = graph.nodes.index(node)
            body.append(f"    conv_layer{idx}({buf(x)}, {buf(y)});")

        elif node.op_type in ("Conv", "ConvRelu"):
            w, b = node.inputs[1], node.inputs[2]
            ws = graph.shapes[w]
            pads = a.get("pads", [0, 0, 0, 0])
            stride = a.get("strides", [1, 1])[0]
            fused = 1 if node.op_type == "ConvRelu" else 0
            body.append(f"    CONV({buf(x)}, {xs[1]}, {xs[2]}, {xs[3]}, "
                        f"{c_name(w)}, {c_name(b)}, {ws[0]}, {ws[2]}, {ws[3]}, "
                        f"{pads[0]}, {pads[1]}, {stride}, {fused}, "
                        f"{buf(y)}, {ys[2]}, {ys[3]});")

        elif node.op_type == "Relu":
            body.append(f"    relu({buf(x)}, {numel(x)}, {buf(y)});")

        elif node.op_type == "MaxPool":
            k = a["kernel_shape"][0]
            stride = a.get("strides", a["kernel_shape"])[0]
            body.append(f"    maxpool2d({buf(x)}, {xs[1]}, {xs[2]}, {xs[3]}, {k}, {stride}, "
                        f"{buf(y)}, {ys[2]}, {ys[3]});")

        elif node.op_type == "Reshape":
            # same numbers, new layout: just copy if the planner gave it a different buffer
            if buf(x) != buf(y):
                body.append(f"    memcpy({buf(y)}, {buf(x)}, {numel(x)} * sizeof(float));")

        elif node.op_type == "Gemm" and kernel == "special":
            idx = graph.nodes.index(node)
            body.append(f"    gemm_layer{idx}({buf(x)}, {buf(y)});")

        elif node.op_type == "Gemm":
            w, b = node.inputs[1], node.inputs[2]
            trans_b = a.get("transB", 0)
            body.append(f"    gemm({buf(x)}, {xs[1]}, {c_name(w)}, {c_name(b)}, "
                        f"{ys[1]}, {trans_b}, {buf(y)});")

        else:
            raise ValueError("No C code for op: " + node.op_type)

    body.append(f"\n    memcpy(output, {buf(out_name)}, {numel(out_name)} * sizeof(float));")

    out.append("\n/* ---------- the model, one step per node ---------- */")
    out.append("void model_run(const float *input, float *output)\n{")
    out.extend(body)
    out.append("}")
    return "\n".join(out) + "\n"


def build(model_path, build_dir, fast=True, exe_name="model.exe", kernel=None):
    # kernel: "naive" (original), "fast" (row-sweep), "special" (per-layer, all filters at once)
    if kernel is None:
        kernel = "special" if fast else "naive"
    graph = from_onnx(model_path)
    infer_shapes(graph, (1, 1, 28, 28))
    fuse_conv_relu(graph)
    assignment, buffers, lifetimes = plan_memory(graph)
    check_plan(assignment, lifetimes)

    build_dir.mkdir(exist_ok=True)
    in_size = math.prod(graph.shapes[graph.inputs[0]])
    out_size = math.prod(graph.shapes[graph.outputs[0]])

    (build_dir / "model.c").write_text(generate_c(graph, assignment, buffers, kernel))
    main = MAIN_C.replace("INPUT_SIZE", str(in_size)).replace("OUTPUT_SIZE", str(out_size))
    (build_dir / "main.c").write_text(main)

    exe = build_dir / exe_name
    flags = ["-O3", "-march=native"] + (["-DFAST_CONV"] if kernel == "fast" else [])
    subprocess.run(["gcc", *flags, "-o", str(exe),
                    str(build_dir / "model.c"), str(build_dir / "main.c")], check=True)
    return graph, exe


if __name__ == "__main__":
    from torchvision import datasets, transforms
    import onnxruntime as ort

    root = Path(__file__).parent.parent
    model_path = root / "models" / "mnist_cnn.onnx"
    build_dir = root / "build"

    # 1. generate C code and compile it
    graph, exe = build(model_path, build_dir)
    print("Wrote", build_dir / "model.c", "and compiled", exe.name)

    # 2. save the same test image as raw floats for the C program to read
    test_set = datasets.MNIST(root=root / "data", train=False, download=True,
                              transform=transforms.ToTensor())
    image, label = test_set[0]
    image = image.numpy().reshape(1, 1, 28, 28).astype(np.float32)
    image.tofile(build_dir / "input.bin")

    # 3. run the compiled C program
    result = subprocess.run([str(exe), str(build_dir / "input.bin")],
                            capture_output=True, text=True, check=True)
    c_logits = np.array([float(v) for v in result.stdout.split()], dtype=np.float32)

    # 4. compare against ONNX Runtime
    ort_logits = ort.InferenceSession(str(model_path)).run(None, {graph.inputs[0]: image})[0].ravel()

    print("C logits:", np.round(c_logits, 4))
    print("Predicted digit:", int(np.argmax(c_logits)), "  True digit:", label)
    print("Biggest difference vs ONNX Runtime:", np.max(np.abs(c_logits - ort_logits)))
    print("Match:", np.allclose(c_logits, ort_logits, atol=1e-4))