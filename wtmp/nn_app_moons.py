import matplotlib.pyplot as plt
import numpy as np
from week3.opt_utils import load_dataset
from wtmp.netural_network import L_layer_model, predict
from wtmp.reg_utils import plot_decision_boundary, predict_dec

if __name__ == "__main__":
    np.random.seed(1)
    train_X, train_Y = load_dataset()
    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 2]

    parameters, bn_params = L_layer_model(train_X, train_Y, layers_dims, 0.005, print_cost=True)

    # Predict
    predictions, acc = predict(train_X, train_Y, parameters, bn_params)
    print("Accuracy: " + str(acc))
    # Plot decision boundary
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-2, 3])
    axes.set_ylim([-1.5, 2])
    plot_decision_boundary(lambda x: predict_dec(parameters, bn_params, x.T), train_X, train_Y)

