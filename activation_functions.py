import numpy as np 

 # relu activation
def relu(x):
    # return np.maximum(0, x)
    return x * (x > 0)

# derivative of relu
def relup(x):
    return 1.0 * (x > 0) 

 # sigmoid activation
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# derivative of sigmoid
def sigmoidp(x):
    return sigmoid(x) * (1 - sigmoid(x)) 

# softmax activation
def softmax(x):
    g = np.exp(x - np.max(x, axis=0))
    return g / np.sum(g, axis=0) 

# derivative of softmax
def softmaxp(x, e):
    g = softmax(x)
    return g*e - (np.sum(g*e, axis=0) * g)
