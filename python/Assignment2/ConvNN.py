"""
Created on：21 December 2016

Author: PFSW 
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from CS231n.classifiers.cnn import *
from CS231n.data_utils import get_CIFAR10_data
from CS231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from CS231n.layers import *
from CS231n.fast_layers import *
from CS231n.solver import Solver

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


#############################################################################
#                             Sanity check1
#############################################################################

model = FirstConvNet(use_batchnorm=True)

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print('Initial loss (no regularization):', loss)

model.reg = 1.0
loss, grads = model.loss(X, y)
print('Initial loss (with regularization): ', loss)

#############################################################################
#                       Sanity check2:  Gradietn check 
#############################################################################
num_inputs = 2
input_dim = (3, 12, 12)
reg = 0.0
num_classes = 10
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = FirstConvNet(input_dim=input_dim,
                     dtype=np.float64,
                     num_filters = [3],
                     hidden_dims = [3,3],
                     use_batchnorm = True)

loss, grads = model.loss(X, y)

for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print("%s max relative error: %e" % (param_name, rel_error(param_grad_num, grads[param_name])))


#############################################################################
#                             Sanity check3
#############################################################################

# Load the CIFAR10 data.
DIR_data = 'D:/Python Machine learning/'
# the mean image has been sub-tracted
data = get_CIFAR10_data(DIR_data, num_training=28000, num_validation=2000)
for k, v in data.items():
  print("The shape of %s is " %k,v.shape)

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = FirstConvNet(weight_scale=5e-2,use_batchnorm = False)

solver = Solver(model, small_data,
                num_epochs=20, batch_size=50,
                update_rule='Adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
solver.train()
'''
plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
'''

#############################################################################
#                               Test on the model 
#############################################################################

model_4 = FirstConvNet(weight_scale=5e-2, reg=0.01,
                       filter_size = 3,
                       use_batchnorm=True,
                       num_filters=[16, 32, 64],
                       hidden_dims=[500, 500])

solver_4 = Solver(model_4, data,
                num_epochs=5, batch_size=50,
                lr_decay = 0.95,
                update_rule='Adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver_4.train()

solver = solver_4
plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()