import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ReLU(Z):
    return np.maximum(0, Z)

def SoftMax(Z): 
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True) 
    expZ = np.exp(Z_shifted)
    return expZ / np.sum(expZ, axis=0, keepdims=True)  

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def ReLU_deriv(Z):
    return Z > 0