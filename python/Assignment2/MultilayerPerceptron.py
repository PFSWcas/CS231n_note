"""
Created on：18 December 2016

Author: PFSW 

Python 3.6.0 :: Anaconda 4.3.0 (64-bit)
win10 64-bit

"""
"""
In previous exercise, we implement a forward and a backward function each layer.
The forward function will receive inputs, weights, and other parameters 
and will return both an output and a cache object storing data needed for the backward pass.

After implementing a bunch of layers in this way, we will be able to easily 
combine them to build classifiers with different architectures.
"""

import time
import numpy as np
from CS231n.classifiers.fc_net import *
from CS231n.data_utils import get_CIFAR10_data
from CS231n.solver import Solver
from CS231n.gradient_check import eval_numerical_gradient
import matplotlib.pyplot as plt

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def run_model(weight_scale, learning_rate):
    model = FullyConnectedNet([100, 100, 100, 100], weight_scale=weight_scale, dtype=np.float64)
    solver = Solver(model, small_data, 
                    print_every=10, num_epochs=20, batch_size=25,
                    update_rule='SGD',
                    optim_config={
                      'learning_rate': learning_rate
                      }
                    )
    solver.train()
    return solver.train_acc_history 

if __name__ == '__main__':

  # Load the CIFAR10 data.
  DIR_data = 'D:/CS231N_CNN/DataSet/cifar-10-batches-py/'
  # the mean image has been sub-tracted
  data = get_CIFAR10_data(DIR_data, num_training=49000, num_validation=1000)
  for k, v in data.items():
    print("The shape of %s is " %k,v.shape)

  #############################################################################
  #                                    Solver       
  #############################################################################
  # We split the logic for training models into a separate class
  model = TwoLayerNet(hidden_dim=200, reg = 0.5)
  solver = Solver(model, data,
                  update_rule = 'SGD',
                  optim_config = {
                    'learning_rate': 1e-3
                  },
                  lr_decay = 0.95,
                  num_epochs = 5,
                  batch_size = 300,
                  print_every = 100)
  solver.train()


  plt.subplot(2, 1, 1)
  plt.title('Training loss')
  plt.plot(solver.loss_history, 'o')
  plt.xlabel('Iteration')

  plt.subplot(2, 1, 2)
  plt.title('Accuracy')
  plt.plot(solver.train_acc_history, '-o', label='train')
  plt.plot(solver.val_acc_history, '-o', label='val')
  plt.plot([0.5] * len(solver.val_acc_history), 'k--')

  plt.annotate('Accuracy=0.5', xy=(1, 0.5), xytext=(0, 0.4),
            arrowprops=dict(facecolor='red'),
            )

  plt.xlabel('Epoch')
  plt.legend(loc='lower right')
  plt.gcf().set_size_inches(15, 12)
  plt.show()


  
  #############################################################################
  #                               Multilayer network      
  #############################################################################
  # Initial loss and gradient check
  # For gradient checking, you should expect to see errors around 1e-6 or less.

  N, D, H1,H2, H3, C = 2, 15, 20, 30, 40, 10
  X = np.random.randn(N, D)
  y = np.random.randint(C, size = (N,))

  for reg in [0, 3.14]:
    print('Running check with reg = ' + str(reg))
    model = FullyConnectedNet([H1, H2, H3], input_dim=D, num_classes=C, 
                              reg = reg, weight_scale=5e-2, dtype = np.float64)
    loss, grads = model.loss(X, y)
    print('Initial loss: ', loss)

    for name in sorted(grads):
      f = lambda _: model.loss(X, y)[0]
      grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
      print("%s relative error: %.2e" % (name, rel_error(grad_num, grads[name])))

  # As another sanity check, make sure you can overfit a small dataset of 50 images. 
  # First we will try a three-layer network with 100 units in each hidden layer. 
  # You will need to tweak the learning rate and initialization scale, 
  # but you should be able to overfit and achieve 100% training accuracy within 20 epochs.
  # TODO: Use a three-layer Net to overfit 50 training examples.

  num_train = 50
  small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
  }

  weight_scale = 1e-2
  learning_rate = 1e-2
  model = FullyConnectedNet([100, 100], weight_scale=weight_scale, dtype=np.float64)
  solver = Solver(model, small_data, print_every=10, num_epochs=20, batch_size = 25,
                  update_rule='SGD', optim_config={'learning_rate': learning_rate})
  solver.train()
  plt.plot(solver.loss_history, 'o')
  plt.title('Training loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Training loss')
  plt.show()

  # Now try to use a five-layer network with 100 units on each layer to overfit 50 training examples. 
  # Again you will have to adjust the learning rate and weight initialization, 
  # but you should be able to achieve 100% training accuracy within 20 epochs.

  # TODO: Use a five-layer Net to overfit 50 training examples.
  num_train = 50
  small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
  }

  not_reach = True
  while not_reach:
    weight_scale = 10**(np.random.uniform(-2,-1))
    learning_rate = 10**(np.random.uniform(-2,-1))
    train_acc_hist = run_model(weight_scale,learning_rate)
    if max(train_acc_hist) == 1.0:
      not_reach = False
      lr = learning_rate
      ws = weight_scale

  print("Has worked with %f and %f" % (lr,ws))
  plt.plot(solver.loss_history, 'o')
  plt.title('Training loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Training loss')
  plt.show()  


  #############################################################################
  #                               Update tules 
  #                                SGD+Momentum
  #                             RMSProp and Adam   
  #############################################################################
  # 
  # train a six-layer network with both SGD and SGD+momentum. 
  # We could see the SGD+momentum update rule converge faster.
  num_train = 8000
  small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
  }

  solvers = {}

  for update_rule in ['SGD', 'SGD_momentum']:
    print('running with ', update_rule)
    model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

    solver = Solver(model, small_data,
                    num_epochs=5, batch_size=100,
                    update_rule=update_rule,
                    optim_config={
                      'learning_rate': 1e-2,
                    },
                    verbose=True)
    solvers[update_rule] = solver
    solver.train()

  plt.subplot(3, 1, 1)
  plt.title('Training loss')
  plt.xlabel('Iteration')

  plt.subplot(3, 1, 2)
  plt.title('Training accuracy')
  plt.xlabel('Epoch')

  plt.subplot(3, 1, 3)
  plt.title('Validation accuracy')
  plt.xlabel('Epoch')

  for update_rule, solver in solvers.items():
    plt.subplot(3, 1, 1)
    plt.plot(solver.loss_history, 'o', label=update_rule)
    
    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label=update_rule)

    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o', label=update_rule)
    
  for i in [1, 2, 3]:
    plt.subplot(3, 1, i)
    plt.legend(loc='upper center', ncol=4)
  plt.gcf().set_size_inches(15, 15)
  plt.show()

  # RMSProp and Adam
  learning_rates = {'rmsprop': 1e-4, 'Adam': 1e-3}
  for update_rule in ['Adam', 'rmsprop']:
    print('running with ', update_rule)
    model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

    solver = Solver(model, small_data,
                    num_epochs=10, batch_size=100,
                    update_rule=update_rule,
                    optim_config={
                      'learning_rate': learning_rates[update_rule]
                    },
                    verbose=True)
    solvers[update_rule] = solver
    solver.train()
    print

  plt.subplot(3, 1, 1)
  plt.title('Training loss')
  plt.xlabel('Iteration')

  plt.subplot(3, 1, 2)
  plt.title('Training accuracy')
  plt.xlabel('Epoch')

  plt.subplot(3, 1, 3)
  plt.title('Validation accuracy')
  plt.xlabel('Epoch')

  for update_rule, solver in solvers.items():
    plt.subplot(3, 1, 1)
    plt.plot(solver.loss_history, 'o', label=update_rule)
    
    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label=update_rule)

    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o', label=update_rule)
    
  for i in [1, 2, 3]:
    plt.subplot(3, 1, i)
    plt.legend(loc='upper center', ncol=4)
  plt.gcf().set_size_inches(15, 15)
  plt.show()