import math
import subprocess
import numpy as np
from pathlib import Path
from ir import from_onnx, infer_shapes
from passes import fuse_conv_activation, clip_bounds
from memplan import plan_memory, check_plan


# ---------- hand-written C kernels shared by every model ----------

KERNELS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* portable SIMD: a vector of VEC_WIDTH floats. gcc turns vf math into AVX / AVX-512
   on x86 and NEON on ARM (Jetson). The width is picked when the model is compiled. */
typedef float vf __attribute__((vector_size(VEC_WIDTH * 4)));
static inline vf LOADV(const float *p) { vf v; memcpy(&v, p, sizeof v); return v; }
static inline void STOREV(float *p, vf v) { memcpy(p, &v, sizeof v); }
SPLAT_DEFINITION
typedef int vfi __attribute__((vector_size(VEC_WIDTH * 4)));
static inline vf VMAX(vf a, vf b) { vfi m = a < b; return (vf)(((vfi)a & ~m) | ((vfi)b & m)); }
static inline vf VMIN(vf a, vf b) { vfi m = a > b; return (vf)(((vfi)a & ~m) | ((vfi)b & m)); }

/* Original Conv (kept for comparison).
   x: input [C][H][W]   w: weights [K][C][kh][kw]   b: bias [K]   y: output [K][OH][OW] */
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

/* Row-sweep Conv (kept for comparison) */
static void conv2d_fast(const float *x, int C, int H, int W,
                        const float *w, const float *b, int K, int kh, int kw,
                        int pad_t, int pad_l, int stride, int relu,
                        float *y, int OH, int OW)
{
    for (int k = 0; k < K; k++) {
        float *yk = y + k * OH * OW;
        for (int p = 0; p < OH * OW; p++) yk[p] = b[k];
        for (int c = 0; c < C; c++) {
            const float *xc = x + c * H * W;
            for (int i = 0; i < kh; i++)
                for (int j = 0; j < kw; j++) {
                    float wv = w[((k * C + c) * kh + i) * kw + j];
                    int off = j - pad_l;
                    int lo = 0, hi = OW;
                    while (lo < OW && lo * stride + off < 0) lo++;
                    while (hi > lo && (hi - 1) * stride + off >= W) hi--;
                    for (int oh = 0; oh < OH; oh++) {
                        int ih = oh * stride + i - pad_t;
                        if (ih < 0 || ih >= H) continue;
                        const float *xrow = xc + ih * W;
                        float *yrow = yk + oh * OW;
                        for (int ow = lo; ow < hi; ow++) yrow[ow] += wv * xrow[ow * stride + off];
                    }
                }
        }
        if (relu)
            for (int p = 0; p < OH * OW; p++) if (yk[p] < 0.0f) yk[p] = 0.0f;
    }
}

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

/* Original Gemm (kept for comparison) */
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

/* usage:  model input.bin          -> prints the scores
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
        for (int r = 0; r < 5; r++) model_run(input, output);       /* warm up (also loads weights) */
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


VEC = 8   # floats per SIMD vector: 8 (AVX2 / NEON) or 16 (AVX-512); set by detect_vector_width()


def detect_vector_width():
    """Ask gcc what this CPU supports. AVX-512 -> 16 floats per vector, otherwise 8."""
    import os
    out = subprocess.run(["gcc", "-march=native", "-dM", "-E", "-x", "c", os.devnull],
                         capture_output=True, text=True).stdout
    return 16 if "__AVX512F__" in out else 8


def c_name(name):
    # turn "conv_stack.0.weight" into a valid C name: "W_conv_stack_0_weight"
    return "W_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def cfloat(v):
    # a valid C float literal for any value, including infinity
    if v == float("inf"):
        return "FLT_MAX"
    if v == float("-inf"):
        return "-FLT_MAX"
    t = f"{float(v):.9g}"
    if "." not in t and "e" not in t:
        t += ".0"
    return t + "f"


class WeightStore:
    """Collects every weight array into one binary file (weights.bin).
    The C code gets a pointer into one big array instead of millions of numbers as text."""

    def __init__(self):
        self.chunks = []
        self.size = 0

    def add(self, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        offset = self.size
        pad = (-arr.size) % 16                     # keep every array 64-byte aligned
        self.chunks.append(arr)
        if pad:
            self.chunks.append(np.zeros(pad, np.float32))
        self.size += arr.size + pad
        return f"(WTS + {offset})"

    def write(self, path):
        data = np.concatenate(self.chunks) if self.chunks else np.zeros(1, np.float32)
        data.tofile(path)


def activation(node, var):
    # C code that applies this node's fused activation to the scalar `var`
    if node.op_type == "ConvRelu":
        return f"if ({var} < 0.0f) {var} = 0.0f;"
    if node.op_type == "ConvClip":
        lo, hi = node.attributes["clip_min"], node.attributes["clip_max"]
        code = ""
        if lo != float("-inf"):
            code += f"if ({var} < {cfloat(lo)}) {var} = {cfloat(lo)}; "
        if hi != float("inf"):
            code += f"if ({var} > {cfloat(hi)}) {var} = {cfloat(hi)};"
        return code
    return ""


def conv_params(node, graph):
    x = node.inputs[0]
    _, C, H, W = graph.shapes[x]
    _, K, OH, OW = graph.shapes[node.outputs[0]]
    w = graph.weights[node.inputs[1]].astype(np.float32)
    KH, KW = w.shape[2], w.shape[3]
    a = node.attributes
    pt, pl, pb, pr = a.get("pads", [0, 0, 0, 0])
    S = a.get("strides", [1, 1])[0]
    if graph.shapes[x][0] != 1:
        raise ValueError("only batch size 1 is supported")
    if len(node.inputs) > 2 and node.inputs[2]:
        b = graph.weights[node.inputs[2]].astype(np.float32)
    else:
        b = np.zeros(K, np.float32)
    return C, H, W, K, OH, OW, w, b, KH, KW, pt, pl, pb, pr, S, a.get("group", 1)


def padded_source(idx, C, H, W, pt, pl, pb, pr):
    """Code that gives the layer a zero-padded copy of its input (or the input itself if no padding)."""
    HP, WP = H + pt + pb, W + pl + pr
    if pt == pl == pb == pr == 0:
        return HP, WP, "", "    const float *src = x;"
    decl = f"static float PAD{idx}[{C * HP * WP}];   /* zero-padded copy of this layer's input */\n"
    body = f"""    for (int c = 0; c < {C}; c++)                       /* copy input inside the zero border */
        for (int h = 0; h < {H}; h++)
            memcpy(&PAD{idx}[(c * {HP} + h + {pt}) * {WP} + {pl}], &x[(c * {H} + h) * {W}],
                   {W} * sizeof(float));
    const float *src = PAD{idx};"""
    return HP, WP, decl, body


def emit_conv_dense(idx, node, graph, ws):
    """A Conv function for ONE normal (non-depthwise) layer.

    Filters are handled in blocks of KB (8 or 16). For each block, we compute a
    tile of P neighbouring output pixels x KB filters, keeping all P*KB running
    sums in CPU registers as SIMD vectors. Each weight loaded is reused P times.
    The code for a tile is written out fully unrolled by this Python code.
    """
    C, H, W, K, OH, OW, w, b, KH, KW, pt, pl, pb, pr, S, _ = conv_params(node, graph)
    HP, WP, pad_decl, pad_body = padded_source(idx, C, H, W, pt, pl, pb, pr)

    KB = VEC if K <= VEC else 2 * VEC         # filters per block
    V = KB // VEC                             # vectors per block
    NB = (K + KB - 1) // KB                   # number of filter blocks
    KP = NB * KB                              # K rounded up (extra filters are zero)
    T = C * KH * KW                           # filter taps

    packed = np.zeros((NB, KB, C, KH, KW), np.float32)
    packed.reshape(KP, C, KH, KW)[:K] = w
    packed = packed.transpose(0, 2, 3, 4, 1)  # -> [NB][C][KH][KW][KB]
    bias = np.zeros(KP, np.float32)
    bias[:K] = b
    W_ptr, B_ptr = ws.add(packed), ws.add(bias)

    P = min(max(1, 12 // V), OW)              # pixels per tile
    act = activation(node, "r")

    def tile(n):
        L = []
        for p in range(n):
            for v in range(V):
                L.append(f"            vf a{p}_{v} = LOADV(bias + {v * VEC});")
        L.append(f"            for (int t = 0; t < {T}; t++) {{")
        L.append(f"                const int c = t / {KH * KW}, i = (t / {KW}) % {KH}, j = t % {KW};")
        L.append(f"                const float *wk = wblk + t * {KB};")
        L.append(f"                const float *xr = &src[(c * {HP} + oh * {S} + i) * {WP} + ow0 * {S} + j];")
        for v in range(V):
            L.append(f"                vf w{v} = LOADV(wk + {v * VEC});")
        for p in range(n):
            L.append(f"                vf x{p} = SPLAT(xr[{p * S}]);")
            for v in range(V):
                L.append(f"                a{p}_{v} += x{p} * w{v};")
        L.append("            }")
        for p in range(n):
            for v in range(V):
                L.append(f"            STOREV(tmp, a{p}_{v});")
                L.append(f"            for (int k = 0; k < {VEC}; k++) {{")
                L.append(f"                int kk = nb * {KB} + {v * VEC} + k;")
                L.append(f"                if (kk >= {K}) break;")
                L.append(f"                float r = tmp[k]; {act}")
                L.append(f"                y[kk * {OH * OW} + oh * {OW} + ow0 + {p}] = r;")
                L.append("            }")
        return "\n".join(L)

    main_end, tail = OW - OW % P, OW % P
    code = f"""
/* {node.op_type} layer {idx}: {C}x{H}x{W} -> {K}x{OH}x{OW}, {KH}x{KW} filter, stride {S}
   {NB} block(s) of {KB} filters, tile = {P} pixels x {KB} filters ({P * V} vector sums in registers) */
{pad_decl}static void conv_layer{idx}(const float *x, float *y)
{{
    float tmp[{VEC}];
{pad_body}
    for (int nb = 0; nb < {NB}; nb++) {{
        const float *wblk = {W_ptr} + nb * {T * KB};
        const float *bias = {B_ptr} + nb * {KB};
        for (int oh = 0; oh < {OH}; oh++) {{
            for (int ow0 = 0; ow0 < {main_end}; ow0 += {P}) {{
{tile(P)}
            }}"""
    if tail:
        code += f"""
            {{   /* last {tail} column(s) */
            const int ow0 = {main_end};
{tile(tail)}
            }}"""
    code += "\n        }\n    }\n}\n"
    return code



def vector_activation(node, var):
    # C code that applies this node's fused activation to the whole vector `var`
    if node.op_type == "ConvRelu":
        return f"{var} = VMAX({var}, SPLAT(0.0f));"
    if node.op_type == "ConvClip":
        lo, hi = node.attributes["clip_min"], node.attributes["clip_max"]
        code = ""
        if lo != float("-inf"):
            code += f"{var} = VMAX({var}, SPLAT({cfloat(lo)})); "
        if hi != float("inf"):
            code += f"{var} = VMIN({var}, SPLAT({cfloat(hi)}));"
        return code
    return ""


MR = 6    # filters per micro-tile (the pixel side is 2 vectors)


def emit_conv_1x1(idx, node, graph, ws):
    """A 1x1 Conv is a matrix multiply: out[K][pixels] = W[K][C] x in[C][pixels].

    Micro-tile of MR=6 filters x 2 vectors of pixels: 12 vector sums in registers.
    For each input channel: load 2 vectors of pixels, broadcast 6 weights,
    12 multiply-adds. Results are written a whole vector at a time, with the
    activation applied to whole vectors.
    Leftover pixels (when the image size isn't a multiple of the tile) are copied
    into a small zero-padded block first, so they also use the fast tile."""
    C, H, W, K, OH, OW, w, b, KH, KW, pt, pl, pb, pr, S, _ = conv_params(node, graph)
    HW = H * W
    NR = 2 * VEC                                              # pixels per micro-tile
    NB = (K + MR - 1) // MR
    KP = NB * MR
    wpad = np.zeros((KP, C), np.float32)
    wpad[:K] = w.reshape(K, C)
    packed = wpad.reshape(NB, MR, C).transpose(0, 2, 1)       # -> [NB][C][MR]
    bias = np.zeros(KP, np.float32)
    bias[:K] = b
    W_ptr, B_ptr = ws.add(packed), ws.add(bias)
    act = vector_activation(node, "r")

    def micro(xbase, xstride, ybase, ystride, check_k=True):
        # MR filters x NR pixels. Input pixel q of channel c is xbase[c * xstride + q],
        # output pixel q of filter k goes to ybase[k * ystride + q].
        L = [f"            const float *wb = wall + nb * {C * MR};"]
        for m in range(MR):
            L.append(f"            vf a{m}_0 = SPLAT(ball[nb * {MR} + {m}]), a{m}_1 = a{m}_0;")
        L.append(f"            for (int c = 0; c < {C}; c++) {{")
        L.append(f"                const float *xc = {xbase} + c * {xstride};")
        L.append(f"                vf x0 = LOADV(xc), x1 = LOADV(xc + {VEC});")
        L.append(f"                const float *wc = wb + c * {MR};")
        for m in range(MR):
            L.append(f"                {{ vf wv = SPLAT(wc[{m}]); a{m}_0 += wv * x0; a{m}_1 += wv * x1; }}")
        L.append("            }")
        for m in range(MR):
            guard = f"if (nb * {MR} + {m} < {K}) " if (check_k and K % MR) else ""
            L.append(f"            {guard}{{ float *yk = {ybase} + (nb * {MR} + {m}) * {ystride};")
            L.append(f"                vf r = a{m}_0; {act} STOREV(yk, r);")
            L.append(f"                r = a{m}_1; {act} STOREV(yk + {VEC}, r); }}")
        return "\n".join(L)

    full, rest = HW - HW % NR, HW % NR
    tail_decl = ""
    if rest:
        tail_decl = (f"static float TX{idx}[{C * NR}];       /* leftover input pixels, zero-padded */\n"
                     f"static float TY{idx}[{KP * NR}];      /* leftover output pixels */\n")

    code = f"""
/* {node.op_type} 1x1 layer {idx}: {C}x{H}x{W} -> {K}x{OH}x{OW} (a {K}x{C} by {C}x{HW} matrix multiply)
   micro-tile = {MR} filters x {NR} pixels, {VEC}-wide vectors */
{tail_decl}static void conv_layer{idx}(const float *x, float *y)
{{
    const float *wall = {W_ptr}, *ball = {B_ptr};
    for (int p = 0; p < {full}; p += {NR})
        for (int nb = 0; nb < {NB}; nb++) {{
{micro("x + p", HW, "y + p", HW)}
        }}
"""
    if rest:
        code += f"""
    /* last {rest} pixel(s): copy into a zero-padded block, run the same tile, copy back */
    for (int c = 0; c < {C}; c++)
        memcpy(TX{idx} + c * {NR}, x + c * {HW} + {full}, {rest} * sizeof(float));
    for (int nb = 0; nb < {NB}; nb++) {{
{micro(f"TX{idx}", NR, f"TY{idx}", NR, check_k=False)}
    }}
    for (int k = 0; k < {K}; k++)
        memcpy(y + k * {HW} + {full}, TY{idx} + k * {NR}, {rest} * sizeof(float));
"""
    code += "}\n"
    return code


def emit_conv_depthwise(idx, node, graph, ws):
    """A Conv function for ONE depthwise layer: each channel has its own small filter.

    One vector of neighbouring output pixels is computed at once. The KH*KW
    weights of a channel are broadcast into registers once per channel.
    Stride 2: each padded input row is split into its even and odd columns first,
    so every load is a straight run of memory (no gathering every other value)."""
    C, H, W, K, OH, OW, w, b, KH, KW, pt, pl, pb, pr, S, group = conv_params(node, graph)
    if not (group == C == K):
        raise ValueError(f"grouped Conv with group={group} not supported (only depthwise)")
    if S not in (1, 2):
        raise ValueError("depthwise stride must be 1 or 2")
    HP, WP, pad_decl, pad_body = padded_source(idx, C, H, W, pt, pl, pb, pr)
    W_ptr, B_ptr = ws.add(w.reshape(C, KH * KW)), ws.add(b)
    vact, sact = vector_activation(node, "r"), activation(node, "s")

    # where tap (i, j) of output pixel `ow` reads from, as (row array, offset)
    if S == 1:
        def tap(i, j):
            return f"xc + (oh + {i}) * {WP} + ow + {j}"
        prep = ""
        decl = ""
    else:
        WE, WO = (WP + 1) // 2, WP // 2                       # even / odd column counts
        decl = f"static float EV{idx}[{HP * WE}], OD{idx}[{HP * max(WO, 1)}];   /* even / odd columns */\n"
        prep = f"""        for (int h = 0; h < {HP}; h++) {{                 /* split rows into even and odd columns */
            const float *row = xc + h * {WP};
            for (int q = 0; q < {WE}; q++) EV{idx}[h * {WE} + q] = row[2 * q];
            for (int q = 0; q < {WO}; q++) OD{idx}[h * {WO} + q] = row[2 * q + 1];
        }}
"""
        def tap(i, j):
            arr, width = (f"EV{idx}", WE) if j % 2 == 0 else (f"OD{idx}", WO)
            return f"{arr} + (2 * oh + {i}) * {width} + ow + {j // 2}"

    T = KH * KW
    wload = "\n".join(f"        const vf w{t} = SPLAT(wc[{t}]);" for t in range(T))
    vec_taps = "\n".join(f"                acc += LOADV({tap(i, j)}) * w{i * KW + j};"
                         for i in range(KH) for j in range(KW))
    sc_taps = "\n".join(f"                s += *({tap(i, j)}) * wc[{i * KW + j}];"
                        for i in range(KH) for j in range(KW))
    vec_end = OW - OW % VEC

    return f"""
/* {node.op_type} depthwise layer {idx}: {C}x{H}x{W} -> {K}x{OH}x{OW}, {KH}x{KW} filter, stride {S} */
{pad_decl}{decl}static void conv_layer{idx}(const float *x, float *y)
{{
{pad_body}
    const float *wall = {W_ptr}, *wb = {B_ptr};
    for (int c = 0; c < {C}; c++) {{
        const float *xc = src + c * {HP * WP};
        const float *wc = wall + c * {T};
        float *yc = y + c * {OH * OW};
        const vf bias = SPLAT(wb[c]);
{wload}
{prep}        for (int oh = 0; oh < {OH}; oh++) {{
            int ow = 0;
            for (; ow < {vec_end}; ow += {VEC}) {{           /* a vector of output pixels at once */
                vf acc = bias;
{vec_taps}
                vf r = acc; {vact}
                STOREV(yc + oh * {OW} + ow, r);
            }}
            for (; ow < {OW}; ow++) {{                         /* leftover pixels */
                float s = wb[c];
{sc_taps}
                {sact}
                yc[oh * {OW} + ow] = s;
            }}
        }}
    }}
}}
"""


def emit_gemm(idx, node, graph, ws):
    """A Gemm function for ONE layer. Weights are rearranged so that each input
    value updates a block of up to 64 outputs at once, using SIMD vectors in registers."""
    x = node.inputs[0]
    if graph.shapes[x][0] != 1 or node.attributes.get("transA", 0):
        raise ValueError("only Gemm with one input row is supported")
    K = graph.shapes[x][1]
    N = graph.shapes[node.outputs[0]][1]
    w = graph.weights[node.inputs[1]].astype(np.float32)
    if not node.attributes.get("transB", 0):
        w = w.T                                              # -> [N][K]
    w = w * node.attributes.get("alpha", 1.0)
    b = graph.weights[node.inputs[2]].astype(np.float32).ravel() * node.attributes.get("beta", 1.0)

    VB = min(8, (N + VEC - 1) // VEC)                       # vectors per block (up to 64 outputs)
    NBK = VB * VEC
    NB = (N + NBK - 1) // NBK
    NP = NB * NBK
    wpad = np.zeros((NP, K), np.float32)
    wpad[:N] = w
    packed = wpad.reshape(NB, NBK, K).transpose(0, 2, 1)    # -> [NB][K][NBK]
    bias = np.zeros(NP, np.float32)
    bias[:N] = b
    W_ptr, B_ptr = ws.add(packed), ws.add(bias)

    L = [f"""
/* Gemm layer {idx}: {K} inputs -> {N} outputs, in {NB} block(s) of {NBK} outputs */
static void gemm_layer{idx}(const float *x, float *y)
{{
    float tmp[{NBK}];
    for (int nb = 0; nb < {NB}; nb++) {{
        const float *wblk = {W_ptr} + nb * {K * NBK};
        const float *bias = {B_ptr} + nb * {NBK};"""]
    for v in range(VB):
        L.append(f"        vf a{v} = LOADV(bias + {v * VEC});")
    L.append(f"        for (int k = 0; k < {K}; k++) {{")
    L.append("            vf xk = SPLAT(x[k]);")
    for v in range(VB):
        L.append(f"            a{v} += xk * LOADV(wblk + k * {NBK} + {v * VEC});")
    L.append("        }")
    for v in range(VB):
        L.append(f"        STOREV(&tmp[{v * VEC}], a{v});")
    L.append(f"""        int n = {N} - nb * {NBK};
        memcpy(y + nb * {NBK}, tmp, (n < {NBK} ? n : {NBK}) * sizeof(float));
    }}
}}
""")
    return "\n".join(L)


def generate_c(graph, assignment, buffers, kernel, weights_file, ws):
    numel = lambda name: math.prod(graph.shapes[name])
    buf = lambda name: f"B{assignment[name]}"
    splat = "static inline vf SPLAT(float f) { vf v = {" + ", ".join(["f"] * VEC) + "}; return v; }"
    out = ["/* generated by codegenerator.py -- do not edit by hand */",
           f"#define VEC_WIDTH {VEC}", KERNELS.replace("SPLAT_DEFINITION", splat)]
    layer_code = []
    body = []

    in_name, out_name = graph.inputs[0], graph.outputs[0]
    body.append(f"    memcpy({buf(in_name)}, input, {numel(in_name)} * sizeof(float));")

    for idx, node in enumerate(graph.nodes):
        x, y = node.inputs[0], node.outputs[0]
        xs, ys = graph.shapes[x], graph.shapes[y]
        a = node.attributes
        body.append(f"\n    /* {node.op_type}: {x} {xs} -> {y} {ys} */")
        op = node.op_type

        if op in ("Conv", "ConvRelu", "ConvClip") and kernel == "special":
            w_shape = graph.shapes[node.inputs[1]]
            is_1x1 = (w_shape[2] == w_shape[3] == 1 and a.get("strides", [1, 1]) == [1, 1]
                      and not any(a.get("pads", [0, 0, 0, 0])))
            if a.get("group", 1) == 1 and is_1x1:
                layer_code.append(emit_conv_1x1(idx, node, graph, ws))
            elif a.get("group", 1) == 1:
                layer_code.append(emit_conv_dense(idx, node, graph, ws))
            else:
                layer_code.append(emit_conv_depthwise(idx, node, graph, ws))
            body.append(f"    conv_layer{idx}({buf(x)}, {buf(y)});")

        elif op in ("Conv", "ConvRelu"):                   # original kernels (naive / fast)
            if a.get("group", 1) != 1:
                raise ValueError("depthwise Conv needs kernel='special'")
            w_name, b_name = node.inputs[1], node.inputs[2]
            layer_code.append(f"#define {c_name(w_name)} {ws.add(graph.weights[w_name])}")
            layer_code.append(f"#define {c_name(b_name)} {ws.add(graph.weights[b_name])}")
            ws_shape = graph.shapes[w_name]
            pads = a.get("pads", [0, 0, 0, 0])
            stride = a.get("strides", [1, 1])[0]
            body.append(f"    CONV({buf(x)}, {xs[1]}, {xs[2]}, {xs[3]}, "
                        f"{c_name(w_name)}, {c_name(b_name)}, {ws_shape[0]}, {ws_shape[2]}, {ws_shape[3]}, "
                        f"{pads[0]}, {pads[1]}, {stride}, {1 if op == 'ConvRelu' else 0}, "
                        f"{buf(y)}, {ys[2]}, {ys[3]});")

        elif op == "Gemm" and kernel == "special":
            layer_code.append(emit_gemm(idx, node, graph, ws))
            body.append(f"    gemm_layer{idx}({buf(x)}, {buf(y)});")

        elif op == "Gemm":
            w_name, b_name = node.inputs[1], node.inputs[2]
            layer_code.append(f"#define {c_name(w_name)} {ws.add(graph.weights[w_name])}")
            layer_code.append(f"#define {c_name(b_name)} {ws.add(graph.weights[b_name])}")
            body.append(f"    gemm({buf(x)}, {xs[1]}, {c_name(w_name)}, {c_name(b_name)}, "
                        f"{ys[1]}, {a.get('transB', 0)}, {buf(y)});")

        elif op == "Relu":
            body.append(f"    for (int i = 0; i < {numel(x)}; i++) "
                        f"{{ float r = {buf(x)}[i]; if (r < 0.0f) r = 0.0f; {buf(y)}[i] = r; }}")

        elif op == "Clip":
            lo, hi = clip_bounds(node, graph)
            body.append(f"    for (int i = 0; i < {numel(x)}; i++) {{ float r = {buf(x)}[i]; "
                        f"if (r < {cfloat(lo)}) r = {cfloat(lo)}; if (r > {cfloat(hi)}) r = {cfloat(hi)}; "
                        f"{buf(y)}[i] = r; }}")

        elif op == "Add":
            x2 = node.inputs[1]
            if graph.shapes[x2] != xs:
                raise ValueError("Add with broadcasting not supported yet")
            body.append(f"    for (int i = 0; i < {numel(x)}; i++) "
                        f"{buf(y)}[i] = {buf(x)}[i] + {buf(x2)}[i];")

        elif op == "MaxPool":
            k = a["kernel_shape"][0]
            stride = a.get("strides", a["kernel_shape"])[0]
            body.append(f"    maxpool2d({buf(x)}, {xs[1]}, {xs[2]}, {xs[3]}, {k}, {stride}, "
                        f"{buf(y)}, {ys[2]}, {ys[3]});")

        elif op in ("ReduceMean", "GlobalAveragePool"):
            if op == "ReduceMean":
                from ir import reduce_axes
                if reduce_axes(node, graph, len(xs)) != [2, 3]:
                    raise ValueError("ReduceMean only supported over height and width")
            hw = xs[2] * xs[3]
            body.append(f"    for (int c = 0; c < {xs[1]}; c++) {{ float s = 0.0f; "
                        f"for (int i = 0; i < {hw}; i++) s += {buf(x)}[c * {hw} + i]; "
                        f"{buf(y)}[c] = s / {hw}.0f; }}")

        elif op in ("Reshape", "Flatten"):
            # same numbers, new layout: just copy if the planner gave it a different buffer
            if buf(x) != buf(y):
                body.append(f"    memcpy({buf(y)}, {buf(x)}, {numel(x)} * sizeof(float));")

        else:
            raise ValueError("No C code for op: " + op)

    body.append(f"\n    memcpy(output, {buf(out_name)}, {numel(out_name)} * sizeof(float));")

    # all weights live in one array, loaded from weights.bin the first time the model runs
    path = Path(weights_file).resolve().as_posix().replace('"', '\\"')
    out.append(f"""
/* ---------- weights: {ws.size:,} floats, loaded from a file ---------- */
#define WEIGHTS_FILE "{path}"
static float WTS[{max(ws.size, 1)}];

static void load_weights(void)
{{
    FILE *f = fopen(WEIGHTS_FILE, "rb");
    if (!f || fread(WTS, sizeof(float), {max(ws.size, 1)}, f) != {max(ws.size, 1)}) {{
        fprintf(stderr, "could not load weights from %s\\n", WEIGHTS_FILE);
        exit(1);
    }}
    fclose(f);
}}
""")
    out.append("/* ---------- one function per layer ---------- */")
    out.extend(layer_code)
    out.append("\n/* ---------- buffers from the memory planner ---------- */")
    for i, nbytes in enumerate(buffers):
        out.append(f"static float B{i}[{nbytes // 4}];   /* {nbytes:,} bytes */")

    out.append("\n/* ---------- the model, one step per node ---------- */")
    out.append("void model_run(const float *input, float *output)\n{")
    out.append("    static int loaded = 0;\n    if (!loaded) { load_weights(); loaded = 1; }\n")
    out.extend(body)
    out.append("}")
    return "\n".join(out) + "\n"


def build(model_path, build_dir, fast=True, exe_name="model.exe", kernel=None):
    # kernel: "naive" (original), "fast" (row-sweep), "special" (per-layer SIMD)
    if kernel is None:
        kernel = "special" if fast else "naive"
    global VEC
    VEC = detect_vector_width()
    graph = from_onnx(model_path)
    infer_shapes(graph)
    fuse_conv_activation(graph)
    assignment, buffers, lifetimes = plan_memory(graph)
    check_plan(assignment, lifetimes)

    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)
    in_size = math.prod(graph.shapes[graph.inputs[0]])
    out_size = math.prod(graph.shapes[graph.outputs[0]])

    # each program gets its own weights file (the kernels store weights in different layouts)
    weights_file = build_dir / (Path(exe_name).stem + "_weights.bin")
    ws = WeightStore()
    (build_dir / "model.c").write_text(generate_c(graph, assignment, buffers, kernel, weights_file, ws))
    ws.write(weights_file)

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