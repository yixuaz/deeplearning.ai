import numpy as np
from week1.dnn_utils import sigmoid, relu


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
        cache = [Z_norm, gamma, Z - mean, var + eps]
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

def linear_forward(A, W, b, gamma, beta, bn_param):
    Z = np.dot(W, A) + b
    if bn_param != None:
        Zn, b_cache = batchnorm_forward(Z, gamma, beta, bn_param)
        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b_cache)
        return Zn, cache
    return Z, (A, W, b)

def linear_activation_forward(A_prev, W, b, activation, gamma, beta, bn_param):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b, gamma, beta, None)
        A, activation_cache = Z, Z
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b, gamma, beta, bn_param)
        A, activation_cache = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters, bn_params):
    caches = []
    A = X
    L = len(parameters) // 4  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        # print("layers %d" % (l))
        # print("W:" + str(parameters["W" + str(l)]))
        # print("gamma:" + str(parameters["gamma" + str(l)]))
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu",
        parameters["gamma" + str(l)], parameters["beta" + str(l)], bn_params[l-1])

        # print("out:" + str(A[:,0]))
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)],  parameters["b" + str(L)], "sigmoid",
                                          None, None, bn_params[L - 1])
    caches.append(cache)
    # print("layers %d" % (L))
    # print("out:" + str(AL[:,0]))
    assert (AL.shape == (2, X.shape[1]))
    return AL, caches