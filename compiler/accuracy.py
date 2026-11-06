import math
import subprocess
import numpy as np
from pathlib import Path
import onnxruntime as ort
from codegenerator import build

# A tiny C "main" that runs EVERY image in a file through model_run, in one go,
# and writes all the scores to another file. Much faster than starting the
# program 10,000 separate times.
BATCH_MAIN = r"""
#include <stdio.h>

void model_run(const float *input, float *output);

int main(int argc, char **argv)
{
    static float input[INPUT_SIZE], output[OUTPUT_SIZE];
    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (!in || !out) { printf("could not open files\n"); return 1; }

    int n = 0;
    while (fread(input, sizeof(float), INPUT_SIZE, in) == INPUT_SIZE) {
        model_run(input, output);
        fwrite(output, sizeof(float), OUTPUT_SIZE, out);
        n++;
    }
    fclose(in);
    fclose(out);
    printf("%d\n", n);
    return 0;
}
"""


def load_mnist_test(root):
    from torchvision import datasets
    test_set = datasets.MNIST(root=root / "data", train=False, download=True)
    images = test_set.data.numpy().astype(np.float32) / 255.0      # same as ToTensor()
    labels = test_set.targets.numpy()
    return images.reshape(-1, 1, 28, 28), labels


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    model_path = root / "models" / "mnist_cnn.onnx"
    build_dir = root / "build"

    # 1. generate model.c with the fast Conv, then compile it with the batch main
    graph, _ = build(model_path, build_dir, fast=True)
    in_size = math.prod(graph.shapes[graph.inputs[0]])
    out_size = math.prod(graph.shapes[graph.outputs[0]])
    main = BATCH_MAIN.replace("INPUT_SIZE", str(in_size)).replace("OUTPUT_SIZE", str(out_size))
    (build_dir / "main_batch.c").write_text(main)
    exe = build_dir / "model_batch.exe"
    subprocess.run(["gcc", "-O3", "-march=native", "-DFAST_CONV", "-o", str(exe),
                    str(build_dir / "model.c"), str(build_dir / "main_batch.c")], check=True)

    # 2. write all 10,000 test images into one file
    images, labels = load_mnist_test(root)
    images.astype(np.float32).tofile(build_dir / "test_images.bin")
    print(f"Running {len(labels):,} test images...")

    # 3. run them all through the C program
    subprocess.run([str(exe), str(build_dir / "test_images.bin"), str(build_dir / "test_scores.bin")],
                   capture_output=True, text=True, check=True)
    c_scores = np.fromfile(build_dir / "test_scores.bin", dtype=np.float32).reshape(-1, out_size)
    c_pred = c_scores.argmax(axis=1)

    # 4. run them all through ONNX Runtime
    session = ort.InferenceSession(str(model_path))
    name = graph.inputs[0]
    ort_scores = np.concatenate([session.run(None, {name: img[None]})[0] for img in images])
    ort_pred = ort_scores.argmax(axis=1)

    # 5. results
    c_acc = (c_pred == labels).mean() * 100
    ort_acc = (ort_pred == labels).mean() * 100
    agree = (c_pred == ort_pred).sum()
    wrong = np.where(c_pred != labels)[0]

    print()
    print(f"Our C compiler:  {c_acc:.2f}% correct   ({len(labels) - len(wrong):,} / {len(labels):,})")
    print(f"ONNX Runtime:    {ort_acc:.2f}% correct")
    print(f"Same prediction as ONNX Runtime on {agree:,} / {len(labels):,} images")
    print(f"Biggest score difference anywhere: {np.max(np.abs(c_scores - ort_scores)):.1e}")
    print()
    print("First 10 images the model gets wrong:")
    for i in wrong[:10]:
        print(f"  image {i:>5}: true digit {labels[i]}, model said {c_pred[i]}")