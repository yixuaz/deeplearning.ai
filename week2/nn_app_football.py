from week1.netural_network import predict
from week2.netural_network_v2 import L_layer_model
from week2.reg_utils import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()

layers_dims = [train_X.shape[0], 20, 3, 1]

def draw(title):
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


# test without regularization
# parameters = L_layer_model(train_X, train_Y, layers_dims, learning_rate=0.3, num_iterations=30000,
#                            print_cost=10000, enable_gc=True)
# predictions_train, acc_train = predict(train_X, train_Y, parameters)
# print ("On the train set:" + str(acc_train))
# predictions_test, acc_test = predict(test_X, test_Y, parameters)
# print ("On the test set:"+ str(acc_test))
# draw("Model with non-regularization")
#
#
# # test L2 regularization
# parameters = L_layer_model(train_X, train_Y, layers_dims, lambd=0.7, learning_rate=0.3, num_iterations=30000,
#                            print_cost=10000, enable_gc=True)
# predictions_train, acc_train = predict(train_X, train_Y, parameters)
# print ("On the train set:" + str(acc_train))
# predictions_test, acc_test = predict(test_X, test_Y, parameters)
# print ("On the test set:"+ str(acc_test))
# draw("Model with L2-regularization")
#
# # test dropout regularization
# parameters = L_layer_model(train_X, train_Y, layers_dims, keep_prob = 0.86, learning_rate = 0.3, num_iterations=30000, print_cost=10000)
# # parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)
# predictions_train, acc_train = predict(train_X, train_Y, parameters)
# print ("On the train set:" + str(acc_train))
# predictions_test, acc_test = predict(test_X, test_Y, parameters)
# print ("On the test set:"+ str(acc_test))
# draw("Model with dropout-regularization")

# test both l2 and dropout regularization
parameters = L_layer_model(train_X, train_Y, layers_dims, lambd=0.3, keep_prob = 0.95, learning_rate = 0.3, num_iterations=30000, print_cost=10000)
predictions_train, acc_train = predict(train_X, train_Y, parameters)
print ("On the train set:" + str(acc_train))
predictions_test, acc_test = predict(test_X, test_Y, parameters)
print ("On the test set:"+ str(acc_test))
draw("Model with both-regularization")