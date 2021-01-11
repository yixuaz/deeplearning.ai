import numpy as np
import math

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k: mini_batch_size * k + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k: mini_batch_size * k + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches: m]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_velocity(parameters):
    L = len(parameters) // 3
    v = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape) if l != L-1 else None
        v["dg" + str(l + 1)] = np.zeros(parameters["g" + str(l + 1)].shape) if l != L-1 else None
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 3
    for l in range(L-1):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        v["dg" + str(l + 1)] = beta * v["dg" + str(l + 1)] + (1 - beta) * grads["dg" + str(l + 1)]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
        parameters["g" + str(l + 1)] = parameters["g" + str(l + 1)] - learning_rate * v["dg" + str(l + 1)]
    v["dW" + str(L)] = beta * v["dW" + str(L)] + (1 - beta) * grads["dW" + str(L)]
    parameters["W" + str(L)] = parameters["W" + str(L)] - learning_rate * v["dW" + str(L)]
    return parameters, v