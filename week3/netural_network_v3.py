import matplotlib.pyplot as plt

from week1.netural_network import initialize_parameters_deep, update_parameters
from week2.nn_backward_v2 import L_model_backward_reg
from week2.nn_forward_v2 import L_model_forward_reg
from week3.adam import initialize_adam, update_parameters_with_adam
from week3.mini_batch_gd import initialize_velocity, update_parameters_with_momentum, random_mini_batches
from week3.opt_utils import compute_cost_minibatch


def L_layer_model(X, Y, layers_dims, optimizer, lambd=0, keep_prob=1, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True, init = 'HE'):
    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]  # number of training examples

    # Initialize parameters
    op = 2 if init == 'HE' else 1
    parameters = initialize_parameters_deep(layers_dims, 3, op = op)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            aL, caches = L_model_forward_reg(minibatch_X, parameters, keep_prob)

            # Compute cost and add to the cost total
            cost_total += compute_cost_minibatch(aL, minibatch_Y)

            # Backward propagation
            grads = L_model_backward_reg(aL, minibatch_Y, caches, lambd, keep_prob)

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

    return parameters