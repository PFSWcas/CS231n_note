# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:52:00 2016

@author: ZhangHeng
"""
"""
Softmax Classifier
    - implement a fully-vectorized loss function for the Softmax classifier
    - implement the fully-vectorized expression for its analytic gradient
    - check your implementation with numerical gradient
    - use a validation set to tune the learning rate and regularization strength
    - optimize the loss function with SGD
    - visualize the final learned weights 
"""
import numpy as np
import matplotlib.pyplot as plt

from cs231n.data_utils import load_CIFAR10
from cs231n.gradient_check import grad_check_sparse

from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.classifiers.linear_classifier import Softmax, LinearClassifier

import time
import math

if __name__ == '__main__':

    #######################################################################################
    #                      CIFAR-10 Data Loading and Preprocessing                        #
    #######################################################################################
    # Load the raw CIFAR-10 data.
    X_tr, y_tr, X_te, y_te = load_CIFAR10('D:/CS231N_CNN/DataSet/cifar-10-batches-py/')
    # As a sanity check, we print out the size of the traning ans test data
    print('Training data shape:   ', X_tr.shape)
    print('Training labels shape: ', y_tr.shape)
    print('Test data shape:       ', X_te.shape)
    print('Test labels shape:     ', y_te.shape) 

    # visualize some examples from the dataset
    # We show a few examples of trainging images from each class
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)

    samples_per_class = 7
    plt.figure()
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_tr == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_tr[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

    # Subsample the data for more efficient code execution in this exercise.
    num_training = 29000
    num_validation = 1000
    num_test = 1000

    # Validation set will be num_validation points from the original training set.
    mask = range(num_training, num_training + num_validation)
    X_val = X_tr[mask]
    y_val = y_tr[mask]

    # Training set will be the first num_train points from the original training set.
    mask = range(num_training)
    X_train = X_tr[mask]
    y_train = y_tr[mask]

    # Use the first num_test points of the original test set as the test set.
    mask = range(num_test)
    X_test = X_te[mask]
    y_test = y_te[mask]

    print('Train data shape:        ', X_train.shape) 
    print('Train labels shape:      ', y_train.shape) 
    print('Validation data shape:   ', X_val.shape) 
    print('Validation labels shape: ', y_val.shape) 
    print('Test data shape:         ', X_test.shape) 
    print('Test labels shape:       ', y_test.shape) 

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val   = np.reshape(X_val, (X_val.shape[0], -1))
    X_test  = np.reshape(X_test, (X_test.shape[0], -1))

    # As a sanity check, print out the shapes of the data
    print('Training data shape:   ', X_train.shape) 
    print('Validation data shape: ', X_val.shape) 
    print('Test data shape:       ', X_test.shape) 

    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis = 0)
    plt.figure(figsize=(4,4))
    plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
    plt.title('Mean Image')
    plt.show()
    # second: substract the mean image from train and test data
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
    # only has to worry about optimizing a single weight matrix W.
    # Also, lets transform both data matrices so that each image is a column
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T

    print(X_train.shape, X_val.shape, X_test.shape)

    #######################################################################################
    #                                   Softmax Classifier                                #
    #######################################################################################
    # Evaluate the naive implementation of the loss:
    # generate a random weight matrix of small numbers
    W = np.random.randn(num_classes, X_train.shape[0]) * 0.0001
    loss, grad = softmax_loss_naive(W, X_train, y_train, 0.0)
    print('softmax loss vaive is %f' % loss)
    print('sanity check: %f' % (-np.log(0.1)))

    # Complete the implementation of softmax_loss_naive and implement a (naive)
    # version of the gradient that uses nested loops.
    loss, grad = softmax_loss_naive(W, X_train, y_train, 0.0)

    # As we did for the SVM, use numeric gradient checking as a debugging tool.
    # The numeric gradient should be close to the analytic gradient.

    f = lambda w: softmax_loss_naive(w, X_train, y_train, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)
    
    # Now that we have a naive implementation of the softmax loss function and its gradient,
    # implement a vectorized version in softmax_loss_vectorized.
    # The two versions should compute the same results, but the vectorized version should be
    # much faster.
    tic = time.time()
    loss_naive, grad_naive = softmax_loss_naive(W, X_train, y_train, 0.00001)
    toc = time.time()
    print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_train, y_train, 0.00001)
    toc = time.time()
    print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    # Use the Frobenius norm to compare the two versions of the gradient.
    grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
    print('Gradient difference: %f' % grad_difference)

    #######################################################################################
    #                           Stochastic Gradient Descent                               #
    #######################################################################################

    # Now implement SGD in Softmax.train() function and run it with the code below
    softmax = Softmax()
    tic = time.time()
    loss_hist = softmax.train(X_train, y_train, learning_rate = 1e-7, reg = 5e4, num_iters=1500, verbose=True)
    toc = time.time()

    print('That took %fs' % (toc - tic))

    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    # Write the Softmax.predict function and evaluate the performance on both
    # training and validation set
    y_train_pred = softmax.predict(X_train)
    print('training accuracy:%f' % (np.mean(y_train == y_train_pred)))
    y_val_pred = softmax.predict(X_val)
    print('validation accuracy:%f' % (np.mean(y_val == y_val_pred)))

    # use the validation set to tune hyperparameter (regularization strength and
    # learning rate). You should experiment with different ranges for the learning 
    # rates and regularization strengths

    learning_rates = [1e-7, 2e-7, 5e-7, 1e-6]
    regularization_strengths = [1e4, 2e4, 5e4, 1e5, 5e5, 1e6]

    # results are dictionary mapping tuples of the form 
    # (learning_rate, regularization_strength) to tuples of the form 
    # (training_accuracy, validation_accuracy). The accuracy is simply the fration 
    # of data points that are correctly classified.

    results = {}
    best_val = -1    # The highest validation accuracy that we have seen so far.
    best_softmax = None  # The LinearSVM object that chieved the highest validation rate.

    for learning in learning_rates:
        for regularition in regularization_strengths:

            softmax = Softmax()
            softmax.train(X_train, y_train, learning_rate=learning, reg = regularition, num_iters=2000)

            y_train_pred = softmax.predict(X_train)
            training_accuracy = np.mean(y_train == y_train_pred)
            
            y_val_pred = softmax.predict(X_val)
            validation_accuracy = np.mean(y_val == y_val_pred)

            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_softmax = softmax
                print('best val is %f' % best_val)

            results[(learning, regularition)] = (training_accuracy, validation_accuracy)
            print('learning rate %e,regularization %e. train accuracy:%f val accuracy: %f' % (learning, regularition, training_accuracy, validation_accuracy))

    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    # Visualize the cross-validation results
    
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    sz = [results[x][0]*1500 for x in results]
    plt.subplot(1,2,1)
    plt.scatter(x_scatter, y_scatter, sz)
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    # plot validation accuracy
    sz = [results[x][1]*1500 for x in results]
    plt.subplot(1,2,2)
    plt.scatter(x_scatter, y_scatter, sz)
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show()

    # Evaluate the best softmax on test set
    y_test_pred = best_softmax.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('Softmax on raw pixels final test set accuracy: %f' % test_accuracy)

    # Visulize the learned weights for each class.
    # Depending on your choice of learning rate and regularition strength, these may 
    # or may not be nice to look at.

    w = best_softmax.W[:,:-1]     # strip out the bias
    w = w.reshape(10,32,32,3)
    w_min, w_max = np.min(w), np.max(w)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i+1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])

    plt.show()