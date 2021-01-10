import os.path as path
import h5py
import matplotlib.pyplot as plt
from week1.dnn_utils import *
from week1.netural_network import predict
from week3.netural_network_v3 import L_layer_model

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def load_data():
    my_path = path.abspath(path.dirname(__file__))
    train_dataset = h5py.File(path.join(my_path, '../week1/datasets/train_catvnoncat.h5'), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(path.join(my_path,'../week1/datasets/test_catvnoncat.h5'), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# The "-1" makes reshape flatten the remaining dimensions
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
final_layers_dims = []
max_accuracy = 0


parameters =  L_layer_model(train_x, train_y, [12288, 20, 7, 5, 1], 'momentum', lambd = 0.35, mini_batch_size=64, learning_rate=0.0075, num_epochs = 400)
pred_train, acc_train = predict(train_x, train_y, parameters)
pred_test, acc_test = predict(test_x, test_y, parameters)
print(" accuracy " + str(acc_train) + "," + str(acc_test))