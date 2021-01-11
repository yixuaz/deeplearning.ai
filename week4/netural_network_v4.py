import matplotlib.pyplot as plt

import numpy as np

from week4.gradient_check import gradient_check_n
from week4.nn_backward_v3 import L_model_backward_reg
from week4.nn_forward_v3 import L_model_forward_reg
from week4.adam import initialize_adam, update_parameters_with_adam
from week4.mini_batch_gd import initialize_velocity, update_parameters_with_momentum, random_mini_batches
from week4.opt_utils import compute_cost_minibatch

def initialize_parameters_deep(layer_dims, op = 1):
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(op / layer_dims[l-1])
        parameters['g' + str(l)] = np.ones((layer_dims[l], 1)) if l != L - 1 else None
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) if l != L - 1 else None
    return parameters

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 3  # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        if l != L - 1:
            parameters["g" + str(l + 1)] = parameters["g" + str(l + 1)] - learning_rate * grads["dg" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layers_dims, optimizer, lambd=0, keep_prob=1, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True, init = 'HE', softmax = False):
    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]  # number of training examples

    # Initialize parameters
    op = 2 if init == 'HE' else 1
    parameters = initialize_parameters_deep(layers_dims, op = op)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    bn_params = [{'mode': 'train'} for i in range(len(layers_dims))]
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            aL, caches = L_model_forward_reg(minibatch_X, parameters, keep_prob, bn_params, softmax)

            # Compute cost and add to the cost total
            cost, dZ = compute_cost_minibatch(aL, minibatch_Y, softmax, parameters, lambd)
            cost_total += cost

            # Backward propagation
            grads = L_model_backward_reg(dZ, caches, lambd, keep_prob)

            if i < 3:
                gradient_check_n(parameters, grads, minibatch_X, minibatch_Y, softmax, lambd=lambd)
            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)
        cost_avg = cost_total / m

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    for bn_param in bn_params:
        bn_param['mode'] = 'test'
    return parameters, bn_params


