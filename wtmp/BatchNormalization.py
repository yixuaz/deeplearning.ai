import numpy as np

from wtmp.nn_backward import batchnorm_backward
from wtmp.nn_forward import batchnorm_forward
from wtmp2.gradient_check import eval_numerical_gradient_array


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

N, D1, D2, D3 = 200, 50, 60, 3
X = np.random.randn(D1, N)
W1 = np.random.randn(D2, D1)
W2 = np.random.randn(D3, D2)
a = W2.dot(np.maximum(0, W1.dot(X)))

print('Before batch normalization:')
print('  means: ', a.mean(axis=1))
print('  stds: ', a.std(axis=1))

# Means should be close to zero and stds close to one
print('After batch normalization (gamma=1, beta=0)')
a_norm, _ = batchnorm_forward(a,  np.zeros((D3,1)), np.ones((D3,1)), {'mode': 'train'})
print('  mean: ', a_norm.mean(axis=1))
print('  std: ', a_norm.std(axis=1))

# Now means should be close to beta and stds close to gamma
gamma = np.asarray([[1.0, 2.0, 3.0]]).T
beta = np.asarray([[11.0, 12.0, 13.0]]).T
a_norm, _ = batchnorm_forward(a,  beta, gamma,{'mode': 'train'})
print('After batch normalization (nontrivial gamma, beta)')
print('  means: ', a_norm.mean(axis=1))
print('  stds: ', a_norm.std(axis=1))


bn_param = {'mode': 'train'}
gamma = np.ones((D3,1))
beta = np.zeros((D3,1))
bn_param["momentum"] = 0.99
for t in range(500):
  X = np.random.randn(D1, N)
  a = W2.dot(np.maximum(0, W1.dot(X)))
  batchnorm_forward(a,  beta,gamma, bn_param)
bn_param['mode'] = 'test'
X = np.random.randn(D1, N)
a = W2.dot(np.maximum(0, W1.dot(X)))
a_norm, _ = batchnorm_forward(a, beta, gamma, bn_param)

# Means should be close to zero and stds close to one, but will be
# noisier than training-time forward passes.
print('After batch normalization (test-time):')
print('  means: ', a_norm.mean(axis=1))
print('  stds: ', a_norm.std(axis=1))

np.random.seed(1)
N, D = 4, 5
x = 5 * np.random.randn(D, N) + 12
gamma = np.random.randn(D, 1)
beta = np.random.randn(D, 1)
dout = np.random.randn(D, N)

bn_param = {'mode': 'train'}
fx = lambda x: batchnorm_forward(x, beta, gamma, bn_param)[0]
fg = lambda a: batchnorm_forward(x, beta, gamma, bn_param)[0]
fb = lambda b: batchnorm_forward(x, beta, gamma, bn_param)[0]

dx_num = eval_numerical_gradient_array(fx, x, dout)
da_num = eval_numerical_gradient_array(fg, gamma, dout)
db_num = eval_numerical_gradient_array(fb, beta, dout)

_, cache = batchnorm_forward(x, beta, gamma, bn_param)
dx, dgamma, dbeta = batchnorm_backward(dout, cache)
print('dx error: ', rel_error(dx_num, dx))
print('dgamma error: ', rel_error(da_num, dgamma))
print('dbeta error: ', rel_error(db_num, dbeta))

