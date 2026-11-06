import torch
import torchvision
import onnx
from collections import Counter
from pathlib import Path

SUPPORTED = {"Conv", "Relu", "MaxPool", "Reshape", "Gemm"}

root = Path(__file__).parent.parent
path = root / "models" / "mobilenet_v2.onnx"

# 1. download pretrained MobileNetV2 and export it to ONNX
model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, str(path), input_names=["images"],
                  output_names=["logits"], opset_version=17)
print("Exported", path)

# 2. list every operation it uses, and whether our compiler supports it yet
m = onnx.load(str(path))
counts = Counter(n.op_type for n in m.graph.node)
print(f"\n{len(m.graph.node)} nodes, {len(counts)} different operations:\n")
for op, n in counts.most_common():
    status = "supported" if op in SUPPORTED else "NEW"
    print(f"  {op:<22}{n:>4}   {status}")

# 3. the different kinds of Conv it uses (our kernels only handle some of these)
kinds = Counter()
for n in m.graph.node:
    if n.op_type == "Conv":
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        kinds[(tuple(a.get("kernel_shape", [])), a.get("strides", [1, 1])[0], a.get("group", 1) > 1)] += 1
print("\nConv variants (kernel, stride, depthwise?):")
for (k, s, dw), n in kinds.most_common():
    print(f"  {k[0]}x{k[1]} kernel, stride {s}, {'depthwise' if dw else 'normal'}:  {n}")

weights = sum(onnx.numpy_helper.to_array(t).size for t in m.graph.initializer)
print(f"\nTotal weights: {weights:,}")