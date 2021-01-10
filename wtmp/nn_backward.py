import numpy as np
from week1.dnn_utils import sigmoid_backward, relu_backward


def batchnorm_backward(dout, cache):
    D, N = dout.shape
    x_, gamma, x_minus_mean, var_plus_eps = cache

    # calculate gradients
    dgamma = np.sum(x_ * dout, axis=1, keepdims=True)
    dbeta = np.sum(dout, axis=1, keepdims=True)

    dx_ = np.matmul(np.ones((N, 1)), gamma.reshape((1, -1))).T * dout
    dx = N * dx_ - np.sum(dx_, axis=1, keepdims=True) - x_ * np.sum(dx_ * x_, axis=1, keepdims=True)
    dx *= (1.0 / N) / np.sqrt(var_plus_eps)

    return dx, dgamma, dbeta

def linear_backward(dZ, cache, is_bn = True):
    A_prev, W, bn_cache = cache
    dgamma = None
    dbeta = None
    dZ_n = dZ
    if is_bn:
        dZ_n, dgamma, dbeta = batchnorm_backward(dZ, bn_cache)
        # print("dZn:" + str(dZ_n[:, 0]))
    dW = np.dot(dZ_n, A_prev.T)
    dA_prev = np.dot(W.T, dZ_n)
    db = np.sum(dZ_n, axis=1, keepdims=True)
    # print("db:" + str(db))
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)

    return dA_prev, dW, db, dgamma, dbeta


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        # print("dZ:" + str(dZ[:,0]))
        dA_prev, dW, db, dgamma, dbeta = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = dA
        dA_prev, dW, db, dgamma, dbeta = linear_backward(dZ, linear_cache, False)
    else:
        raise ValueError('Invalid  activation mode "%s"' % activation)

    return dA_prev, dW, db, dgamma, dbeta


def L_model_backward(dAL, Y, caches):
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


    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)], grads["dgamma" + str(L)], grads["dbeta" + str(L)] \
        = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l+1)], grads["dgamma" + str(l + 1)], grads["dbeta" + str(l + 1)] \
            = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")

    return grads