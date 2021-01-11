import numpy as np

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    shapes = {}
    for key in sorted(parameters):
        if parameters[key] is None:
            continue
        # flatten parameter
        shapes[key] = parameters[key].shape
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys, shapes


def vector_to_dictionary(theta, keys, shapes):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    idx = 0
    for key in sorted(keys):
        if not(key in shapes):
            parameters[key] = None
            continue
        res = 1 if (len(shapes[key]) == 1) else shapes[key][1]
        next = idx + shapes[key][0] * res
        parameters[key] = theta[idx:next].reshape(shapes[key])
        idx = next
    return parameters


def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    count = 0
    for key in sorted(gradients.keys()):
        if not (key.startswith('dW') or key.startswith('db') or key.startswith('dg'))\
                or gradients[key] is None:
            continue
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

def gradients_to_vector2(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    count = 0
    for key in sorted(gradients.keys()):
        if not (key.startswith('W') or key.startswith('b') or key.startswith('gamma') or key.startswith('beta')):
            continue
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

def gradient_check_n_test_case():
    np.random.seed(1)
    x = np.random.randn(4, 3)
    y = np.array([[1, 1, 0]])
    W1 = np.random.randn(5, 4)
    b1 = np.random.randn(5, 1)
    W2 = np.random.randn(3, 5)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return x, y, parameters