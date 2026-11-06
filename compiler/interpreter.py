import numpy as np
from pathlib import Path
from ops import relu, reshape, gemm, max_pool, conv
from ir import from_onnx, infer_shapes
from passes import fuse_conv_relu
from torchvision import datasets, transforms
import onnxruntime as ort


model_path = Path(__file__).parent.parent / "models" / "mnist_cnn.onnx"

## load the model into our IR, work out shapes, then optimize
graph = from_onnx(model_path)
infer_shapes(graph, (1, 1, 28, 28))
print("Nodes before fusion:", len(graph.nodes))

fuse_conv_relu(graph)
print("Nodes after fusion: ", len(graph.nodes))


## environment: start with a copy of the weights
tensor_dict = dict(graph.weights)


## load one test image from MNIST
data_path = Path(__file__).parent.parent / "data"
test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

image, label = test_set[0]
image = image.numpy().reshape(1, 1, 28, 28).astype(np.float32)

tensor_dict[graph.inputs[0]] = image
print("Loaded image, shape:", image.shape, "true digit is:", label)


## run the model
for node in graph.nodes:
    inputs = [tensor_dict[name] for name in node.inputs]
    attrs = node.attributes

    if node.op_type == "Conv":
        result = conv(inputs[0], inputs[1], inputs[2], attrs["pads"][0], attrs["strides"][0])
    elif node.op_type == "ConvRelu":
        result = relu(conv(inputs[0], inputs[1], inputs[2], attrs["pads"][0], attrs["strides"][0]))
    elif node.op_type == "Relu":
        result = relu(inputs[0])
    elif node.op_type == "MaxPool":
        result = max_pool(inputs[0], attrs["kernel_shape"][0], attrs["strides"][0])
    elif node.op_type == "Reshape":
        result = reshape(inputs[0], inputs[1])
    elif node.op_type == "Gemm":
        result = gemm(inputs[0], inputs[1], inputs[2])
    else:
        raise ValueError("Unsupported op: " + node.op_type)

    # check the actual result against shape inference
    assert result.shape == node.shape, f"{node.op_type}: got {result.shape}, expected {node.shape}"

    tensor_dict[node.outputs[0]] = result
    print(node.op_type, "->", node.outputs[0], result.shape)


## the final answer
logits = tensor_dict[graph.outputs[0]]
print("Predicted digit:", np.argmax(logits), "  True digit:", label)


## compare against ONNX Runtime
session = ort.InferenceSession(str(model_path))
ort_logits = session.run(None, {graph.inputs[0]: image})[0]

print("Biggest difference:", np.max(np.abs(logits - ort_logits)))
print("Match:", np.allclose(logits, ort_logits, atol=1e-5))