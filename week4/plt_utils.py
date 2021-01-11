import matplotlib.pyplot as plt
import numpy as np
from week4.nn_forward_v3 import L_model_forward_reg


def predict_dec(parameters, X, bn_params, softmax):
    # Predict using forward propagation and a classification threshold of 0.5
    probas, caches = L_model_forward_reg(X, parameters, 1, bn_params, softmax)
    if softmax:
        probas = probas[1, :]
    predictions = (probas > 0.5)
    return predictions

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()

def predict(X, y, parameters, bn_params, softmax):
    """
    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward_reg(X, parameters, 1, bn_params, softmax)
    if softmax:
        probas = probas[1,:].reshape((1, X.shape[1]))
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    acc = np.sum((p == y) / m)
    # print("Accuracy: " + str(acc))
    return p, acc