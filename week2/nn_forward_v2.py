import numpy as np
from week1.dnn_utils import sigmoid, relu

def linear_forward(A, W, b, D):
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = [A, W, b, D]
    return Z, cache

def keep_prob_wrap(A, D, keep_prob):
    return A * D / keep_prob

def linear_activation_forward(A_prev, W, b, activation, keep_prob):
    D = (np.random.rand(W.shape[0], A_prev.shape[1]) < keep_prob).astype(int)
    Z, linear_cache = linear_forward(A_prev, W, b, D)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    A = keep_prob_wrap(A,  D, keep_prob)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward_reg(X, parameters, keep_prob):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    np.random.seed(1)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu", keep_prob)
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid", 1)
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches