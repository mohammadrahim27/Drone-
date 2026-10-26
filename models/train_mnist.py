import numpy as np
import torch
import onnxruntime as ort
from torchvision import datasets
from torchvision.transforms import v2
import torch.nn as nn
import onnx




#============ SECTION 2: Load the MNIST dataset for CV==========================

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
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)



#================ SECTION 3: Defining the model for CV==========================


#getting the device for training
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device: {device}")

#Define the model architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*7*7, 10)
        )



    def forward(self, x):
        logits = self.conv_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


X = torch.rand(1, 1, 28, 28, device=device)
logits = model(X)
pred_probabilities = nn.Softmax(dim=1)(logits)
y_pred = pred_probabilities.argmax(dim=1)
print(f"Predicted class: {y_pred}")

#================ SECTION 4: Training the model for CV==========================


#set the hyperparameters for training (batch_size is already set in Section 2)
learning_rate = 1e-3
epochs = 5

#initializing the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropogation for loss in gradient descent
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0


    #evaluate model using pytorch.nograd() to ensure that there are no gradients being calcualted during this test
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


#================ SECTION 5: Saving the model and test cases==========================

#save the trained weights
torch.save(model.state_dict(), "models/mnist_mlp.pth")
print("Saved model weights to models/mnist_mlp.pth")

#save one batch of test images and the model's outputs as the answer key for the interpreter
model.eval()
with torch.no_grad():
    test_images, test_labels = next(iter(test_dataloader))
    test_images = test_images.to(device)
    test_outputs = model(test_images)

np.save("models/test_images.npy", test_images.cpu().numpy())
np.save("models/test_labels.npy", test_labels.numpy())
np.save("models/test_outputs.npy", test_outputs.cpu().numpy())
print("Saved test images, labels, and outputs to models/")



#================ SECTION 6: Exporting the model to ONNX format==========================
model.eval()
dummy_input = torch.randn(1, 1, 28, 28, device=device)


torch.onnx.export(
    model, 
    (dummy_input,),
    "models/mnist_cnn.onnx",
    input_names=["images"],
    output_names=["logits"],
)

#checking to see if the file is valid on ONNX file
onnx_model = onnx.load("models/mnist_cnn.onnx")
onnx.checker.check_model(onnx_model)


#check to see if ONNX Runtime gives the same answer as the PyTorch model on one test image
session = ort.InferenceSession("models/mnist_cnn.onnx")
one_image = test_images[0:1].cpu().numpy()
onnx_output = session.run(None, {"image": one_image})[0]
torch_output = test_outputs[0:1].cpu().numpy()
np.testing.assert_allclose(onnx_output, torch_output, rtol=1e-4, atol=1e-5)
print("ONNX Runtime output matches PyTorch")
