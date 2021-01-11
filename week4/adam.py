import numpy as np

def initialize_adam(parameters):
    L = len(parameters) // 3
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape) if l != L-1 else None
        v["dg" + str(l + 1)] = np.zeros(parameters["g" + str(l + 1)].shape) if l != L-1 else None
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape) if l != L-1 else None
        s["dg" + str(l + 1)] = np.zeros(parameters["g" + str(l + 1)].shape) if l != L-1 else None
    return v, s

def update_one_param(v_corrected, s_corrected, l, val, parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    v["d" + val + str(l + 1)] = beta1 * v["d" + val + str(l + 1)] + (1 - beta1) * grads["d" + val + str(l + 1)]
    v_corrected["d" + val + str(l + 1)] = v["d" + val + str(l + 1)] / (1 - np.power(beta1, t))
    s["d" + val + str(l + 1)] = beta2 * s["d" + val + str(l + 1)] + (1 - beta2) * np.power(grads["d" + val + str(l + 1)], 2)
    s_corrected["d" + val + str(l + 1)] = s["d" + val + str(l + 1)] / (1 - np.power(beta2, t))
    parameters[val + str(l + 1)] = parameters[val + str(l + 1)] - learning_rate * v_corrected["d" + val + str(l + 1)] / (
        np.sqrt(s_corrected["d" + val + str(l + 1)]) + epsilon)
    
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 3  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary
    for l in range(L-1):
        update_one_param(v_corrected, s_corrected, l, "W", parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        update_one_param(v_corrected, s_corrected, l, "b", parameters, grads, v, s, t, learning_rate, beta1, beta2,
                         epsilon)
        update_one_param(v_corrected, s_corrected, l, "g", parameters, grads, v, s, t, learning_rate, beta1, beta2,
                         epsilon)
    update_one_param(v_corrected, s_corrected, L - 1, "W", parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
    return parameters, v, s
