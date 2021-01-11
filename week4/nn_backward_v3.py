import numpy as np
from week1.dnn_utils import sigmoid_backward, relu_backward, relu_backward_dropout

def batchnorm_backward(dZ, cache):
  Z, gamma, beta, Z_norm, mean, var, eps = cache
  N = Z.shape[1]

  dgamma = np.sum(dZ * Z_norm, axis = 1, keepdims=True)
  dbeta = np.sum(dZ, axis = 1, keepdims=True)
  dZ_norm = dZ * gamma

  dvar = -0.5 * np.sum(dZ_norm * (Z - mean), axis=1, keepdims=True) * np.power(var + eps, -1.5)

  dmean = -np.sum(dZ_norm , axis=1, keepdims=True) / np.sqrt(var + eps) - 2 * dvar * np.sum(Z - mean, axis=1, keepdims=True)/ N

  dx = dZ_norm /np.sqrt(var + eps) + 2.0 * dvar * (Z - mean) / N + dmean / N

  return dx, dgamma, dbeta

def linear_backward_reg(dZ, cache, lambd, keep_prob):
    A_prev, W, current_D, prev_D = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) + lambd * W / m

    dA_prev = np.dot(W.T, dZ)
    if prev_D is not None:
        dA_prev = dA_prev * prev_D / keep_prob
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    return dA_prev, dW


def linear_activation_backward_reg(dA, cache, activation, lambd, keep_prob, first_ele_is_dZ=False):
    linear_cache, bn_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dZ, dg, db = batchnorm_backward(dZ, bn_cache)
    elif activation == "sigmoid":
        dZ = dA if first_ele_is_dZ else sigmoid_backward(dA, activation_cache)
        dg, db = None, None

    dA_prev, dW = linear_backward_reg(dZ, linear_cache, lambd, keep_prob)
    return dA_prev, dW, dg, db


def L_model_backward_reg(dZ, caches, lambd, keep_prob):
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
    current_cache[0].append(caches[L - 2][0][2])
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["dg" + str(L)], grads["db" + str(L)] \
        = linear_activation_backward_reg(dZ, current_cache, "sigmoid", lambd, keep_prob, True)

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        prev_D = None if l == 0 else caches[l - 1][0][2]
        current_cache[0].append(prev_D)
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["dg" + str(l + 1)], grads["db" + str(l + 1)]\
            = linear_activation_backward_reg(grads["dA" + str(l + 1)], current_cache, "relu", lambd, keep_prob)

    return grads