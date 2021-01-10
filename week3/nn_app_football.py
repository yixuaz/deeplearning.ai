from week1.netural_network import predict
from week2.reg_utils import *
from week3.netural_network_v3 import L_layer_model

train_X, train_Y, test_X, test_Y = load_2D_dataset()

layers_dims = [train_X.shape[0], 20, 3, 1]

def draw(title):
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


parameters = L_layer_model(train_X, train_Y, layers_dims, "momentum", mini_batch_size=512, lambd=0.3, keep_prob = 0.95, learning_rate = 0.3, num_epochs=1000)
# parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)
predictions_train, acc_train = predict(train_X, train_Y, parameters)
print ("On the train set:" + str(acc_train))
predictions_test, acc_test = predict(test_X, test_Y, parameters)
print ("On the test set:"+ str(acc_test))
draw("Model with both-regularization")