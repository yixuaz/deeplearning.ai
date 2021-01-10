import numpy as np
from week1.dnn_utils import compute_cost
import matplotlib.pyplot as plt

from wtmp.gradient_check import gradient_check_n
from wtmp.nn_backward import L_model_backward
from wtmp.nn_forward import L_model_forward
from wtmp.gc_utils import softmax_loss


def initialize_parameters_deep(layer_dims, seed=1, op = 1):
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.normal(0.0, 5e-2, (layer_dims[l], layer_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        # if l != L - 1:
        parameters['gamma' + str(l)] = np.ones((layer_dims[l], 1))
        parameters['beta' + str(l)] = np.zeros((layer_dims[l], 1))
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 4  # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        assert (parameters['W' + str(l+1)].shape == grads["dW" + str(l + 1)].shape)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        assert (parameters['b' + str(l+1)].shape == grads["db" + str(l + 1)].shape)
        if l != L - 1:
            parameters["gamma" + str(l + 1)] = parameters["gamma" + str(l + 1)] - learning_rate * grads["dgamma" + str(l + 1)]
            assert (parameters['gamma' + str(l+1)].shape == grads["dgamma" + str(l + 1)].shape)
            parameters["beta" + str(l + 1)] = parameters["beta" + str(l + 1)] - learning_rate * grads["dbeta" + str(l + 1)]
            assert (parameters['beta' + str(l+1)].shape == grads["dbeta" + str(l + 1)].shape)
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    costs = []  # keep track of cost
    oldX = np.copy(X)
    oldY = np.copy(Y)
    parameters = initialize_parameters_deep(layers_dims, op=2)
    bn_params = [{'mode': 'train'} for i in range(len(layers_dims))]
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters, bn_params)

        # Compute cost.
        cost, dAL = softmax_loss(AL, Y)

        # Backward propagation.
        grads = L_model_backward(dAL, Y, caches)
        # for i in range(3):
        #     print('db%d' % (i+1) + str(grads['db%d' % (i+1)]))
        #     print('dW%d' % (i + 1) + str(grads['dW%d' % (i + 1)]))
        #     print('dgamma%d' % (i + 1) + str(grads['dgamma%d' % (i + 1)]))
        #     print('dbeta%d' % (i + 1) + str(grads['dbeta%d' % (i + 1)]))
        if i < 3:
            gradient_check_n(parameters, grads, X, Y, oldX, oldY)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 1 == 0:
            costs.append(cost)
    if print_cost:
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    for bn_param in bn_params:
        bn_param['mode'] = 'test'
    return parameters, bn_params


def predict(X, y, parameters, bn_params):
    """
    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    x, caches = L_model_forward(X, parameters, bn_params)
    probs = np.exp(x - np.max(x, axis=0, keepdims=True))
    probs /= np.sum(probs, axis=0, keepdims=True)
    # convert probas to 0/1 predictions
    for i in range(0, probs.shape[1]):
        if probs[1, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    acc = np.sum((p == y) / m)
    # print("Accuracy: " + str(acc))
    return p, acc