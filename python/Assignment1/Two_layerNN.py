# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 19:23:29 2016
@author: ZhangHeng
"""

import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.data_utils import load_CIFAR10
from cs231n.vis_utils import visualize_grid

if __name__ == '__main__':
    # Load up the CIFAR-10 data
    cifar10_dir =  '../../DataSet/cifar-10-batches-py/'
    X_tr, y_tr, X_te, y_te = load_CIFAR10(cifar10_dir)
    num_training = 9000
    num_validation = 1000
    num_test = 1000

    # Subsample the data
    mask = range(num_training, num_training+num_validation)
    X_val = X_tr[mask]
    y_val = y_tr[mask]
    mask = range(num_training)
    X_train = X_tr[mask]
    y_train = y_tr[mask]
    mask = range(num_test)
    X_test = X_te[mask]
    y_test = y_te[mask]

    # Normalise the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    print('Train data shape:', X_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Validation data shape:', X_val.shape)
    print('Validation labels shape:', y_val.shape)
    print('Test data shape:', X_test.shape)
    print('Test labels shape:', y_test.shape)

    #%%
    # train a nerwork
    input_size = 32*32*3
    hidden_size = 81
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
                num_iters=1500, batch_size=200,
                learning_rate=7.5e-4, learning_rate_decay=0.95,
                reg=1.0, verbose=True)

    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()
    print('Validation accuracy: ', val_acc)

    #%%
    # Debug the training
    # plot the loss function and train/validation accuracies

    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='validation')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.show()

    #%%
    # Visualize the weights of the network
    def show_network_weights(network):
        W1 = network.params['W1']
        W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
        plt.imshow(visualize_grid(W1, padding=1).astype('uint8'))
        plt.gca().axis('off')
        plt.show()

    show_network_weights(net)

    #%%
    # Tune the hyperparameters using the validation set. Store your best trained
    # model in best_net

    hidden_size = 81
    best_val = -1
    best_Net = None

    learning_rate = np.array([2.5, 5, 7.5, 10, 15, 20]) * 1e-4
    regularization_strengths = [0.25, 0.5, 0.75, 1, 1.25]

    for lr in learning_rate:
        for reg in regularization_strengths:
            net = TwoLayerNet(input_size, hidden_size, num_classes)
            # train the network
            stats = net.train(X_train, y_train, X_val, y_val,
                            num_iters=1500, batch_size=200,
                            learning_rate=lr, learning_rate_decay = 0.95,
                            reg = reg, verbose=False)
            val_acc = (net.predict(X_val) == y_val).mean()
            if val_acc > best_val:
                best_val = val_acc
                best_Net = net
            print("Learing rate %e, regularization %e, validation accuracy : %f" % (lr, reg, val_acc))
    print("best validation accuracy achieved during cross-validation: %f" % best_val)