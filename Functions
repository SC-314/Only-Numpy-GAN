import numpy as np
from PIL import Image
import time
from numba import njit
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
import random

@njit
def conv(X, W):
    OS = X.shape[-1] - W.shape[-1] + 1
    output = np.zeros((W.shape[0], OS, OS))
    for d in range(output.shape[0]):
        for r in range(output.shape[1]):
            for c in range(output.shape[2]):
                sum_value = 0.0
                for i in range(W.shape[1]):
                    for j in range(W.shape[2]):
                        for k in range(W.shape[3]):
                            sum_value += X[i, r+j, c+k] * W[d, i, j, k]
                output[d, r, c] = sum_value
    return output

@njit
def convT(X, W, stride):
    os = stride * (X.shape[-1] - 1) + W.shape[-1]
    output = np.zeros((W.shape[0], os, os))
    for fm in range(W.shape[0]):
        for r in range(X.shape[1]):
            for c in range(X.shape[2]):
                for i in range(W.shape[1]):
                    for j in range(W.shape[2]):
                        for k in range(W.shape[3]):
                            output[fm, stride*r+j, stride*c+k] += X[i, r, c] * W[fm, i, j, k]
    return output

@njit
def dLdW(X, dLdZ, W):
    output = np.zeros_like(W)
    for l in range(output.shape[0]):
        for d in range(output.shape[1]):
            for r in range(output.shape[2]):
                for c in range(output.shape[3]):
                    sum_value = 0.0
                    for j in range(dLdZ.shape[1]):
                        for k in range(dLdZ.shape[2]):
                            sum_value += X[d, r+j, c+k] * dLdZ[l, j, k]
                    output[l,d,r,c] = sum_value
    return output

@njit
def dLdX(dLdZ, W, X):
    output = np.zeros_like(X)
    for l in range(W.shape[0]):
        for d in range(W.shape[1]):
            for r in range(W.shape[2]):
                for c in range(W.shape[3]):
                    for j in range(dLdZ.shape[1]):
                        for k in range(dLdZ.shape[2]):
                            output[d,r+j,c+k] +=  dLdZ[l, j, k] * W[l, d, r, c]
    return output

@njit
def dLdWT(dLdZ, X1, W0, stride):
    output = np.zeros(W0.shape)
    for fm in range(output.shape[0]):
        for r in range(output.shape[2]):
            for c in range(output.shape[3]):
                for i in range(X1.shape[0]):
                    for j in range(X1.shape[1]):
                        for k in range(X1.shape[2]):
                            output[fm, i, r, c] += dLdZ[fm, r+stride*j, c+stride*k] * X1[i, j, k]
    return output


@njit
def dLdXT(dLdZ1, W1, X1, stride):
    output = np.zeros(X1.shape)
    for fm in range(W1.shape[0]):
        for r in range(output.shape[1]):
            for c in range(output.shape[2]):
                for i in range(W1.shape[1]):
                    for j in range(W1.shape[2]):
                        for k in range(W1.shape[3]):
                            output[i, r, c] += dLdZ1[fm, stride*r+j, stride*c+k] * W1[fm, i, j, k]
    return output

@njit
def maxpool(X, MPsize):
    OS = int(np.ceil(X.shape[-1]/MPsize))
    output = np.zeros((X.shape[0],OS, OS)) - np.inf
    mask = np.zeros((X.shape[0], OS, OS, 2))
    for d in range(output.shape[0]):
        counter = -1
        for r in range(output.shape[1]):
            for c in range(output.shape[2]):
                counter += 1
                for i in range(min(X.shape[-2], (r+1)*MPsize)-r*MPsize):
                    for j in range(min(X.shape[-1], (c+1)*MPsize)-c*MPsize):
                        if output[d, r, c] < X[d, r*MPsize+i, c*MPsize+j]:
                            output[d, r, c] = X[d, r*MPsize+i, c*MPsize+j]
                            mask[d,r,c] = (r*MPsize+i, c*MPsize+j)
    return output, mask

@njit
def maxpoolBP(X, dLdZ, Z_M):
    output = np.zeros_like(X)
    for d in range(dLdZ.shape[0]):
        for r in range(dLdZ.shape[1]):
            for c in range(dLdZ.shape[2]):
                coord = Z_M[d, r, c]
                output[d, int(coord[0]), int(coord[1])] = dLdZ[d, r, c]
    return output

def pad(X, pad_size):
    os = X.shape[-1] + 2*pad_size
    output = np.zeros((X.shape[0], os, os))
    for l in range(X.shape[0]):
        output[l] = np.pad(X[l], pad_size)
    return output

@njit
def pad_bp(dLdDX, pad_size):
    os = dLdDX.shape[-1] - 2*pad_size
    output = np.zeros((dLdDX.shape[0], os, os))
    for l in range(dLdDX.shape[0]):
        output[l] = dLdDX[l][pad_size:-pad_size, pad_size:-pad_size]
    return output

@njit
def sigmoid(X):
    return 1/(1+np.exp(-X))

@njit
def tanh(X):
    return np.tanh(X)

@njit
def cross_entropy(A, sols1hot):
    sum_value = 0.0
    for c in range(A.shape[-1]):
        sum_value -= np.log(A[0, c]+0.0000001) * sols1hot[0, c]
    return sum_value

def ReLU(X):
    return np.where(X >= 0, X, 0)

def ReLU_BP(X):
    return np.where(X >0, 1, 0)

def Glorot_init(input_size, output_size, filter_size=False):
    if filter_size == False:
        return np.random.randn(input_size, output_size) * np.sqrt(2/(input_size+output_size))
    else:
        return np.random.randn(input_size, output_size, filter_size, filter_size) * np.sqrt(2 / (filter_size**2 * (output_size + input_size)))
