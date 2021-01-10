import numpy as np

from week1.dnn_utils import compute_cost, relu_backward
from week1.nn_backward import linear_activation_backward
from week1.nn_forward import linear_activation_forward

# x1 * x1 * 10 == x2 -> y = 1, else y = 0
X = np.array([(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8), (0.1,0.4,0.6,1.6,0.33,0.1,4.9,6.4)])
Y = np.array([(1,1,0,1,0,0,1,1)])
print(X.shape)
print(Y.shape)

def initialize_parameters_deep(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 10 + 10
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

layers_dims = [2,2,2,2,1]
parameters = initialize_parameters_deep(layers_dims)
print(parameters)

def L_model_forward_debug(X, parameters, debug = False):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
        if debug and (l == 1 or l == L - 1):
            print("forward " + str(l) + ": " + str(A))

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    if debug :
        print("forward " + str(L) + ": " + str(AL))
    return AL, caches

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

# sigmoid only happened in last layer, so we pass dZ to dA to direct use it.
# use sigmoid_backward will cause precision issue
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = dA
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward_debug(AL, Y, caches, debug = False):
    grads = {}
    L = len(caches)  # the number of layers
    dZ = AL - Y
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dZ, current_cache, "sigmoid")

    if debug:
        print("backward " + str(L - 1) + ": " + str(grads["dW" + str(L)]))
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu")
        if debug and l == 0:
            print("backward " + str(l) + ": " + str(grads["dW" + str(l+1)]))
    return grads

def update_parameters_debug(parameters, grads, learning_rate, debug = False):
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        if (debug and (l == 0 or l == L - 1)):
            print("W" + str(l + 1) + ": " + str(parameters["W" + str(l + 1)]))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

for i in range(0, 10):
    debug = True
    if debug:
        print("iteration i:" + str(i))
    AL, caches = L_model_forward_debug(X, parameters, debug)
    cost = compute_cost(AL, Y)
    if debug:
        print("cost:" + str(cost))
        print()
    grads = L_model_backward_debug(AL, Y, caches, debug)
    parameters = update_parameters_debug(parameters, grads, 0.1, debug)