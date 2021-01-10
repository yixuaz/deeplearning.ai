import numpy as np

from week2.gradientCheck.gradient_check import gradient_check_n
from week2.nn_forward_v2 import L_model_forward_reg
from week2.nn_backward_v2 import L_model_backward_reg
from week1.dnn_utils import compute_cost
from week1.netural_network import initialize_parameters_deep, update_parameters
import matplotlib.pyplot as plt

def L_layer_model(X, Y, layers_dims, lambd=0, keep_prob=1, learning_rate=0.0075, num_iterations=3000,
                  print_cost=0, seed = 3, enable_gc=False):  # lr was 0.009
    costs = []  # keep track of cost
    parameters = initialize_parameters_deep(layers_dims, seed)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward_reg(X, parameters, keep_prob)
        # AL, caches = forward_propagation_with_dropout(X, parameters, keep_prob)
        # Compute cost.
        cost = compute_cost(AL, Y, parameters, lambd)

        # Backward propagation.
        grads = L_model_backward_reg(AL, Y, caches, lambd, keep_prob)

        if enable_gc and i < 3:
            gradient_check_n(parameters, grads, X, Y, lambd = lambd)
        # grads = backward_propagation_with_dropout(AL, Y, caches, keep_prob)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % print_cost == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % (print_cost/10) == 0:
            costs.append(cost)
    if print_cost == 10000:
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters

