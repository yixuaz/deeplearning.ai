import numpy as np
from week2.gradientExplode.gradient_explode import update_parameters_debug
from week1.dnn_utils import compute_cost
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

# all sigmoid activation
def L_model_forward_debug(X, parameters, debug = False):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L+1):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "sigmoid")
        if debug and (l == 1 or l == L):
            print("forward " + str(l) + ": " + str(A))
        caches.append(cache)
    return A, caches

# all sigmoid activation
def L_model_backward_debug(AL, Y, caches, debug = False):
    grads = {}
    L = len(caches)  # the number of layers
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA" + str(L)] = dAL
    for l in reversed(range(L)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "sigmoid")
        if debug and (l == 0 or l == L-1):
            print("backward " + str(l) + ": " + str(grads["dW" + str(l+1)]))
    return grads

for i in range(0, 1000):
    debug = i == 0 or i == 999
    if debug:
        print("iteration i:" + str(i))
    AL, caches = L_model_forward_debug(X, parameters, debug)
    cost = compute_cost(AL, Y)
    if debug:
        print("cost:" + str(cost))
        print()
    grads = L_model_backward_debug(AL, Y, caches, debug)
    parameters = update_parameters_debug(parameters, grads, 0.1, debug)