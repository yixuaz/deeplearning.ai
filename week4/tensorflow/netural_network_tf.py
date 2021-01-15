import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
from week4.mini_batch_gd import random_mini_batches


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    return X, Y


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        wl, bl = 'W' + str(l), 'b' + str(l)
        parameters[wl] = tf.get_variable(wl, [layer_dims[l], layer_dims[l-1]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters[bl] = tf.get_variable(bl, [layer_dims[l], 1], initializer=tf.zeros_initializer())
    return parameters


def forward_propagation(X, parameters, layer_dims):
    L = len(layer_dims)
    ZL, AL = None, X
    for l in range(1, L-1):
        wl, bl = 'W' + str(l), 'b' + str(l)
        ZL = tf.matmul(parameters[wl], AL) + parameters[bl]
        AL = tf.nn.relu(ZL)
    return tf.matmul(parameters['W' + str(L-1)], AL) + parameters['b' + str(L-1)]

def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def model(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(layer_dims)
    Z = forward_propagation(X, parameters, layer_dims)
    cost = compute_cost(Z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters