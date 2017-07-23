# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 11:43:00 2016

@author: ZhangHeng
"""
"""
The kNN classifier consists of two stages:
  1) During training, the classifier takes the training data and simply remembers it
  2) During testing, kNN classifies every test image by comparing to all training 
     images and transfering the labels of the k most similar training examples
The value of k is cross-validated
"""

import numpy as np
import random
import sys
import matplotlib.pyplot as plt

from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.K_NearestNeighbor import KNearestNeighbor

def time_cost(f, *args):
    """
    Call a function f with args ans return the time (in seconds) that it took
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

if __name__ == '__main__':

    # load the CiFAR-10 data
    X_tr, y_tr, X_te, y_te = load_CIFAR10('../../DataSet/cifar-10-batches-py/')

    # As a sanity check, we print out the size of the training and test data
    print('Training data shape:   ', X_tr.shape)
    print('Training labels shape: ', y_tr.shape)
    print('Test data shape:       ', X_te.shape)
    print('Test labels shape:     ', y_te.shape)

    # Visuallize  some examples from the dataset 
    classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    num_class = len(classes)
    samples_per_class = 7
    plt.figure()
    for y, cls in enumerate(classes): # eg:y=0, cls='plane'; y=1, cls='car'
        idxs = np.flatnonzero(y_tr == y)  # find the indicies of y_tr==y
        idxs = np.random.choice(idxs, samples_per_class,replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_class + y + 1
            plt.subplot(samples_per_class, num_class, plt_idx)
            plt.imshow(X_tr[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)

    plt.show()

    # Subsample the data for more efficient code excution in this exercise
    num_training = 5000
    IndSel = range(num_training)
    X_train = X_tr[IndSel]
    y_train = y_tr[IndSel]

    num_test = 2000
    IndSel = range(num_test)
    X_test = X_te[IndSel]
    y_test = y_te[IndSel]
    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test  = np.reshape(X_test, (X_test.shape[0], -1))
    print(X_train.shape, X_test.shape)

    #% KNN classifier creation
    # Creat a KNN classifier instance
    # Trainning a KNN classifier is a loop
    # the classifier simply remembers the data  and does no further processing

    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    #% compute_distances_two_loops
    dists = classifier.compute_dis_two_loops(X_test)
    print(dists.shape)

    # We can visualize the distance matrix: each row is a single test 
    # example and its distances to training examples
    plt.figure()
    plt.imshow(dists, interpolation='none')
    plt.show()

    # Now implement the function predict_labels and run the code below:
    # We use k = 1 (which is Nearest Neighbor).
    y_test_pred = classifier.predict_labels(dists, k=1)

    # Compute and print the fraction of correctly predicted examples
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

    y_test_pred = classifier.predict_labels(dists, k=5)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

    # Now lets speed up distance matrix computation by using partial vectorization
    # with one loop
    dists_one = classifier.compute_dis_one_loops(X_test)
    difference = np.linalg.norm(dists - dists_one, ord='fro')
    print('Difference was: %f' % (difference))

    if difference < 0.001:
        print('Good! The distance matrices are the same')
    else:
        print('Uh-oh! The distance matrices are different')

    # Now implement the fully vectorized version inside compute_distances_no_loops
    dists_noloop = classifier.compute_dis_no_loops(X_test)
    difference = np.linalg.norm(dists - dists_noloop, ord='fro')
    print('Difference was: %f' % (difference))

    if difference < 0.001:
        print('Good! The distance matrices are the same')
    else:
        print('Uh-oh! The distance matrices are different')

    # Let's compare how fast the implementations are


    two_loop_time = time_cost(classifier.compute_dis_two_loops,X_test)
    print('Two loop version took %f seconds' % two_loop_time)
    one_loop_time = time_cost(classifier.compute_dis_one_loops, X_test)
    print('One loop version took %f seconds' % one_loop_time)
    no_loop_time = time_cost(classifier.compute_dis_no_loops, X_test)
    print('No loop version took %f seconds' % no_loop_time)
    
    #% cross validation
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 20, 50, 100]
    X_train_folds = []
    y_train_folds = []

    # Split up the training data into folds. After splitting, X_train_folds and y_train_folds
    # should each be lists of length num_fods, where y_train_folds[i] is the label
    # vector for the points in X_train_folds[i]

    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    # A dictionary holding the accuracies for the different values of k that 
    # we find when running cross-validation. After running cross-validation, 
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using the value of k
    k_to_accuracies = {}

    # Perform k-fold cross validation to find the best value of k. For each possible
    # value of k, run the k-nearest-neighbor algorithm num_folds times, where in each
    # case you use all but one of the folds as training data and all vaues of k in 
    # the k_to_accuracies dictionary

    for k in k_choices:
        k_to_accuracies[k] = []

    for k in k_choices:
        print( 'evaluating k=%d' % k)
        for j in range(num_folds):
            X_train_slides = np.vstack(X_train_folds[0:j] + X_train_folds[j+1:])
            X_test_slides  = X_train_folds[j]

            y_train_slides = np.hstack(y_train_folds[0:j] + y_train_folds[j+1:])
            y_test_slides  = y_train_folds[j]

            classifier.train(X_train_slides, y_train_slides)
            dists_slides = classifier.compute_dis_no_loops(X_test_slides)
            y_test_pred  = classifier.predict_labels(dists_slides, k)
            num_correct = np.sum(y_test_pred == y_test_slides)
            accuracy = float(num_correct) / y_test_slides.shape[0]

            k_to_accuracies[k].append(accuracy)

    # Pirnt  out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print( 'k = %d, accuracy = %f' % (k,accuracy))

    # plot the raw observations
    plt.figure()
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std  = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()