import onnx
import numpy as np
from onnx import numpy_helper
from pathlib import Path
from ops import relu, reshape, gemm, max_pool, conv
from torchvision import datasets, transforms
import onnxruntime as ort




model_path = Path(__file__).parent.parent / "models" / "mnist_cnn.onnx"



model_proto = onnx.load(model_path)

graph = model_proto.graph


##dictionary for mapping tensor names to numpy arrays
tensor_dict = {}


## inputs the input of the interpreters
for inp in graph.input:
    print("Input name:", inp.name)



## displays the outputs 
for out in graph.output:
    print("Output name: ", out.name)




## initializer to map tensors to numpy arrays, serves as the environment

for init in graph.initializer:
    print("Initializer name: ", init.name)
    tensor_dict[init.name] = numpy_helper.to_array(init)
    print(tensor_dict[init.name].shape)




## load one test image from MNIST
data_path = Path(__file__).parent.parent / "data"
test_set = datasets.MNIST(root=data_path, train=False, download=True, transform = transforms.ToTensor())

image, label = test_set[0]
image = image.numpy().reshape(1, 1, 28, 28).astype(np.float32)


tensor_dict["images"] = image
print("Loaded image, shape:", image.shape, "true digit is: ", label)

## loop over the nodes in the graph and display their information


for node in graph.node:
    print("Node name: ", node.name)
    print("Node op_type: ", node.op_type)
    print("Node inputs: ", node.input)
    print("Node outputs: ", node.output)
    print("Node attributes: ", node.attribute)


    for att in node.attribute:
        print("Attribute name: ", att.name)
        print("Attribute value: ", onnx.helper.get_attribute_value(att))




## run the model: go through each node in order and call the matching op
for node in graph.node:

    # 1. grab this node's inputs from the environment by name
    inputs = []
    for name in node.input:
        inputs.append(tensor_dict[name])

    # 2. collect this node's settings (pads, strides, ...) into a dictionary
    attrs = {}
    for att in node.attribute:
        attrs[att.name] = onnx.helper.get_attribute_value(att)

    # 3. call the op that matches this node's type
    if node.op_type == "Conv":
        result = conv(inputs[0], inputs[1], inputs[2], attrs["pads"][0], attrs["strides"][0])
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

    # 4. store the result under this node's output name, so the next node can find it
    tensor_dict[node.output[0]] = result
    print(node.op_type, "->", node.output[0], result.shape)


## the final answer: the 10 scores, and which digit scored highest
logits = tensor_dict["logits"]
print("Logits:", logits)
print("Predicted digit:", np.argmax(logits), "  True digit:", label)


## compare against ONNX Runtime (the official engine) to confirm the interpreter is correct
session = ort.InferenceSession(str(model_path))
ort_logits = session.run(None, {"images": image})[0]

print("ONNX Runtime logits:", ort_logits)
print("Biggest difference:", np.max(np.abs(logits - ort_logits)))
print("Match:", np.allclose(logits, ort_logits, atol=1e-5))