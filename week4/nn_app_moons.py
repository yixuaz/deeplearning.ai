from week4.plt_utils import plot_decision_boundary, predict_dec, predict
from week4.netural_network_v4 import L_layer_model
from week4.opt_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(12)
    train_X, train_Y = load_dataset()
    # train 3-layer model

    for optimizer in ["gd", "momentum", "adam"]:
        use_softmax = True
        layers_dims = [train_X.shape[0], 5, 2, 2 if use_softmax else 1]
        parameters, bn_params = L_layer_model(train_X, train_Y, layers_dims, optimizer, lambd=0, num_epochs=3000, mini_batch_size=64, learning_rate=0.007, init = 'X', softmax = use_softmax)

        # Predict
        predictions, acc = predict(train_X, train_Y, parameters, bn_params, use_softmax)
        print("Accuracy: " + str(acc))
        # Plot decision boundary
        plt.title("Model with Gradient Descent optimization")
        axes = plt.gca()
        axes.set_xlim([-1.5, 2.5])
        axes.set_ylim([-1, 1.5])
        plot_decision_boundary(lambda x: predict_dec(parameters, x.T, bn_params, use_softmax), train_X, train_Y)