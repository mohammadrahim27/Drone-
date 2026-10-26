## We create the operations for the interpreter. The operations are defined in the ops.py file, which contains the necessary functions and classes to perform various computations and manipulations on data structures. These operations are essential for the execution of programs within the interpreter, allowing for efficient processing and transformation of data.
## first 

import numpy as np


def relu(x):
   return np.maximum(0, x)



## function to perform reshape after MaxPool and ReLU

def reshape(x, shape):
    return np.reshape(x, shape)



## General Matrix Multiplication (GEMM)
def gemm(A, B, C): 
    return A @ B.T + C


## MaxPool function
def max_pool(x, kernel, stride):
    I, G, H, W = x.shape

    H_out = (H - kernel) // stride + 1
    W_out = (W - kernel) // stride + 1

    out = np.zeros((I, G, H_out, W_out))

    for img in range(I):
        for grid in range(G):
            for i in range(H_out):
                for j in range(W_out):
                    top = i * stride
                    left = j * stride
                    block = x[img, grid, top:top+kernel, left:left+kernel]
                    out[img, grid, i, j] = np.max(block)

    return out


## Conv function
def conv(x, weight, bias, pad, stride):
    I, G, H, W = x.shape
    F = weight.shape[0]
    K = weight.shape[2]

    x_padded = np.pad(x, ((0, 0), (0,0), (pad, pad), (pad, pad)))

    H_out = (H + 2 * pad - K) // stride + 1
    W_out = (W + 2 * pad - K) // stride + 1

    out = np.zeros((I, F, H_out, W_out))

    for img in range(I):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    top = i * stride
                    left = j * stride
                    block = x_padded[img, :, top:top+K, left:left+K]
                    out[img, f, i, j] = np.sum(block * weight[f]) + bias[f]

    return out















## test cases used for the functions defined above
if __name__ == "__main__":

    # relu test case
    test_input = np.array([-1, -2, 0, 1, 2])
    print("Input: ", test_input)
    relu_output = relu(test_input)
    print("ReLU Output: ", relu_output)

    # reshape test case
    test_grids = np.arange(784).reshape(1, 16, 7, 7)
    print("Before reshape ", test_grids.shape)
    reshape_output = reshape(test_grids, np.array([1, 784]))
    print("After reshape ", reshape_output.shape)
    print("First 10 values ", reshape_output[0, :10])

    # gemm test case
    A = np.array([1, 2, 3])
    B = np.array([[1, 2, 3,], 
                 [4, 5, 6,]])
    C = np.array([5, 6])
    print("Gemm result: ", gemm(A, B, C))

    # Max Pool test case 
    arr = np.array([[1, 4, 2, 0],
                    [3, 5, 6, 8],
                    [6, 3, 5, 6],
                    [0, 1, 3, 8]])
    arr = arr.reshape(1, 1, 4, 4)
    print("MaxPool result: ", max_pool(arr, 2, 2))

    # Conv Test Case 
     # Conv test case
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])
    x = x.reshape(1, 1, 3, 3)               

    weight = np.zeros((1, 1, 3, 3))         
    weight[0, 0, 1, 1] = 1.0                

    bias = np.array([0.5])                  

    print("Conv result: ", conv(x, weight, bias, 1, 1))




