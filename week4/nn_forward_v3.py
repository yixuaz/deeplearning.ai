import numpy as np

from week1.dnn_utils import sigmoid, relu, softmax

def batchnorm_forward(Z, gamma, beta, bn_param):
    D, N = Z.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros((D, 1), dtype=Z.dtype))
    running_var = bn_param.get('running_var', np.zeros((D, 1), dtype=Z.dtype))
    if mode == 'train':
        mean = np.mean(Z, axis=1, keepdims=True)
        var = np.var(Z, axis=1, keepdims=True)
        Z_norm = (Z - mean) / np.sqrt(var + eps)
        Zn = gamma * Z_norm + beta
        cache = [Z, gamma, beta, Z_norm, mean, var, eps]
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        # TEST：要用整个训练集的均值、方差
    elif mode == 'test':
        Z_norm = (Z - running_mean) / np.sqrt(running_var + eps)
        Zn = gamma * Z_norm + beta
        cache = None
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return Zn, cache

def linear_forward(A, W, D):
    Z = np.dot(W, A)
    assert (Z.shape == (W.shape[0], A.shape[1]))
    return Z, [A, W, D]


def keep_prob_wrap(A, D, keep_prob):
    return A * D / keep_prob


def linear_activation_forward(A_prev, W, beta, gamma, activation, keep_prob, bn_param):
    D = (np.random.rand(W.shape[0], A_prev.shape[1]) < keep_prob).astype(int)
    Z, linear_cache = linear_forward(A_prev, W, D)
    bn_cache = None
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        Z, bn_cache = batchnorm_forward(Z, beta, gamma, bn_param)
        A, activation_cache = relu(Z)
    A = keep_prob_wrap(A, D, keep_prob)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, bn_cache, activation_cache)
    return A, cache


def L_model_forward_reg(X, parameters, keep_prob, bn_params, softmax):
    caches = []
    A = X
    L = len(parameters) // 3  # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["g" + str(l)],
                                             parameters["b" + str(l)], "relu", keep_prob, bn_params[l-1])
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["g" + str(L)],
                                          parameters["b" + str(L)],
                                          "softmax" if softmax else "sigmoid",
                                          1, bn_params[L-1])
    caches.append(cache)
    assert (AL.shape == (2 if softmax else 1, X.shape[1]))
    return AL, caches
