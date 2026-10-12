import numpy as np
import torch
import torchvision
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt 
from torchvision import datasets
from torchvision.transforms import v2

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)



batch_size = 64
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=true)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)



