import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets

def load_dataset():
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
    # Visualize the data
    # plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y


def compute_cost_minibatch(aL, Y, softmax, parameters, lambd):
    N = aL.shape[1]
    cost_total, sum, L = 0, 0, len(parameters) // 3
    if lambd > 0:
        for l in range(L):
            w = parameters['W' + str(l + 1)]
            val = 0 if w is None else w
            sum += np.sum(np.square(val))
        cost_total = lambd / 2 * sum
    if softmax:
        cost_total += -np.sum(np.log(aL[Y, np.arange(N)]))
        dAL = aL.copy()
        dAL[Y, np.arange(N)] -= 1
        return cost_total, dAL / N
    logprobs = np.multiply(-np.log(aL), Y) + np.multiply(-np.log(1 - aL), 1 - Y)
    cost_total += np.sum(logprobs)
    return cost_total, (aL - Y) / N
