import numpy as np

from week2.gradientCheck.gc_utils import gradients_to_vector, dictionary_to_vector, vector_to_dictionary, \
    gradient_check_n_test_case
from week4.nn_backward_v3 import L_model_backward_reg
from week4.nn_forward_v3 import L_model_forward_reg
from week4.opt_utils import compute_cost_minibatch

def forward_propagation_n(X, Y, parameters, softmax, lambd = 0):
    bn_params = [{'mode': 'train'} for i in range(len(parameters) // 3)]
    AL, caches = L_model_forward_reg(X, parameters, 1, bn_params, softmax)
    cost, _ = compute_cost_minibatch(AL, Y, softmax, parameters, lambd)
    return cost / X.shape[1], caches, AL


def gradient_check_n(parameters, gradients, X, Y, softmax, epsilon=1e-7, lambd = 0):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    # Set-up variables
    parameters_values, _, shapes = dictionary_to_vector(parameters)

    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):

        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
        J_plus[i], _, _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus, parameters.keys(),shapes), softmax, lambd)  # Step 3

        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
        J_minus[i], _, _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus, parameters.keys(),shapes), softmax, lambd)  # Step 3
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / 2 / epsilon

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference

if __name__ == "__main__":
    X, Y, parameters = gradient_check_n_test_case()

    cost, cache, AL = forward_propagation_n(X, Y, parameters)
    gradients = L_model_backward_reg(AL, Y, cache, 0, 1)
    difference = gradient_check_n(parameters, gradients, X, Y)
