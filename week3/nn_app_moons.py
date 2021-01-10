import matplotlib.pyplot as plt

from week1.netural_network import predict
from week2.reg_utils import predict_dec, plot_decision_boundary
from week3.netural_network_v3 import L_layer_model
from week3.opt_utils import load_dataset

if __name__ == "__main__":
    train_X, train_Y = load_dataset()
    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    for optimizer in ["gd", "momentum", "adam"]:
        parameters = L_layer_model(train_X, train_Y, layers_dims, optimizer=optimizer)

        # Predict
        predictions, acc = predict(train_X, train_Y, parameters)
        print("Accuracy: " + str(acc))
        # Plot decision boundary
        plt.title("Model with Gradient Descent optimization")
        axes = plt.gca()
        axes.set_xlim([-1.5, 2.5])
        axes.set_ylim([-1, 1.5])
        plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)