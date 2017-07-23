
import time
import numpy as np
from CS231n.classifiers.fc_net import *
from CS231n.data_utils import get_CIFAR10_data
from CS231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
import matplotlib.pyplot as plt

# Load the CIFAR10 data.
DIR_data = 'D:/CS231N_CNN/DataSet/cifar-10-batches-py/'

# the mean image has been sub-tracted
data = get_CIFAR10_data(DIR_data, num_training=49000, num_validation=1000)
# data is a dictionary:
# data = {
#        'X_train': X_train, 'y_train': y_train,
#        'X_val': X_val, 'y_val': y_val,
#        'X_test': X_test, 'y_test': y_test,
#    }

for k, v in data.items():
    print("The shape of %s is " %k,v.shape)

from CS231n.solver import SolverCheckpoints

num_train = 20000
batch_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}
from CS231n.classifiers.cnn import FirstConvNet

model_0 = FirstConvNet(weight_scale=5e-2, 
                       reg=0.01,
                       filter_size = 3,
                       use_batchnorm=True,
                       num_filters=[16, 32, 64, 128], 
                       hidden_dims=[256, 256])

solver_0 = SolverCheckpoints(model_0, batch_data,
                             path='D:\\CS231N_CNN\\notebook\Assignment2\\experiments\\',
                             num_epochs=50, batch_size=50,
                             lr_decay = 0.95,
                             update_rule='Adam',
                             optim_config={
                                'learning_rate': 1e-3,
                             },
                             verbose=True, print_every=40)
solver_0.train()