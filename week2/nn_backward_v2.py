import numpy as np
from week1.dnn_utils import sigmoid_backward, relu_backward, relu_backward_dropout

def linear_backward_reg(dZ, cache, lambd, keep_prob):
    A_prev, W, b, current_D, prev_D = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m + lambd * W / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    if prev_D is not None:
        dA_prev = dA_prev * prev_D / keep_prob
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward_reg(dA, cache, activation, lambd, keep_prob, first_ele_is_dZ=False):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = dA if first_ele_is_dZ else sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward_reg(dZ, linear_cache, lambd, keep_prob)
    return dA_prev, dW, db


def L_model_backward_reg(AL, Y, caches, lambd, keep_prob):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    dZ = AL - Y
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    current_cache[0].append(caches[L - 2][0][3])
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] \
        = linear_activation_backward_reg(dZ, current_cache, "sigmoid", lambd, keep_prob, True)

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        prev_D = None if l == 0 else caches[l - 1][0][3]
        current_cache[0].append(prev_D)
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward_reg(grads["dA" + str(l + 1)], current_cache, "relu", lambd, keep_prob)

    return grads