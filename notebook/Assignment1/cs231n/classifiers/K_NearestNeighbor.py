# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 21:26:29 2016

@author: ZhangHeng
"""

import numpy as np

class KNearestNeighbor(object):
    """ a KNN classifier with L2 distance"""
    def __init__(self):
        pass

    def train(self,X,y):
        """
        Train the classifier. For KNN, this is just memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data consisting of 
             num_train samples each of dimension D.
        - y: A numpy array of shape(N,) containing the training labels, where y[i] is the label for x[i].
        """
        self.X_tr = X
        self.y_tr = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting of 
             num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implenmentation to use to compute distances 
                     between training points and testing points.

        Returns:
        - y: A numpy array array of shape (num_test,) containging predicted labels for 
             the test data, where y[i] is the predicted label for the test point X[i]
        """
        if num_loops == 0:
            dists = self.compute_dis_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_dis_one_loops(X)
        elif num_loops == 2:
            dists = self.compute_dis_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_dis_two_loops(self,X):
        """
        Compute the distances between each test point in X and each training point 
        in self.X_tr using a nested loop over both the training data the test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting 
             of num_test samples each of dimension D.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i,j] 
                 is the Euclidean distance between the i-th 
                 test point and the j-th training point.
        """
        num_test = X.shape[0]
        num_train = self.X_tr.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j] = np.sqrt(np.sum(np.square(X[i, :] - self.X_tr[j,:])))

        return dists

    def compute_dis_one_loops(self,X):
        """
        Compute the distances between each test point in X and each training 
        point in self.X_tr using a single loop the test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting 
             of num_test samples each of dimension D.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i,j] 
                 is the Euclidean distance between the i-th test point and the j-th training point.
        """
        num_test = X.shape[0]
        num_train = self.X_tr.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:] = np.sqrt(np.sum(np.square(self.X_tr - X[i,:]), axis=1)) # broadcasting
        
        return dists

    def compute_dis_no_loops(self,X):
        """
        Compute the distances between each test point in X and each training point in self.X_tr 
        using no explicit loops.
        
        Hint: Try to formulate the L2 distance using matrix multiplication and two broadcast sums

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting of num_test 
             samples each of dimension D.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i,j] is the Euclidean 
                 distance between the i-th test point and the j-th training point.
        """
        num_test = X.shape[0]
        num_train = self.X_tr.shape[0]
        dists = np.zeros((num_test, num_train))
        
        test_sum = np.sum(np.square(X), axis=1)          # num_test x 1
        train_sum = np.sum(np.square(self.X_tr), axis=1) # num_train x 1
        inner_ptoduct = np.dot(X, self.X_tr.T)           # num_test X num_train
        dists = np.sqrt(-2 * inner_ptoduct + (test_sum.reshape(-1,1) + train_sum)) # broadcast

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distance between test points and training points,
        predict a label for each test point.

        Input:
        dists: A num_test x num_train arrray where dists[i,j] gives the distance between
               the i-th test point and the j-th training point.
        Output:
        y: A vector of length num_test where y[i] is the predicted label for the i-th test point
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # creat a list of length k storing the labels of the k nearst neighbors to the i-th
            # test point,
            closest_y = []
            y_ind = np.argsort(dists[i,:], axis=0)
            closest_y = self.y_tr[y_ind[0:k]]
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.
            y_pred[i] = np.argmax(np.bincount(closest_y)) 

        return y_pred
