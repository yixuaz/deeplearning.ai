import os
import sklearn.datasets
import scipy.io
import matplotlib.pyplot as plt
from wtmp.nn_forward import L_model_forward
import numpy as np

def load_2D_dataset():

    data = scipy.io.loadmat(os.path.join(os.getcwd(), '../week2/datasets/data.mat'))
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    return train_X, train_Y, test_X, test_Y


def predict_dec(parameters, bn_params, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    x, cache = L_model_forward(X, parameters, bn_params)
    probs = np.exp(x - np.max(x, axis=0, keepdims=True))
    probs /= np.sum(probs, axis=0, keepdims=True)
    predictions = (probs > 0.5)
    return predictions[1,:].reshape((1, predictions.shape[1]))

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