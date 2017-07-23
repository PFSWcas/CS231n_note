# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 20:03:40 2016

@author: ZhangHeng
"""

import os
import numpy as np
# import cPickle as pickle # python2.7
import pickle # python 3

def load_CIFAR_batch(filename):
    """
    cifad data is stored in 5 natches, in this funxtion, we load one batch
    - parameters: filename
    - outputs: X,Y: data and corresponding labels
    """
    with open(filename,'rb') as f:
        datadict = pickle.load(f, encoding='latin1') # python3
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float") # [N, height, width, channels]
        Y = np.array(Y)
        return X,Y
        

def load_CIFAR10(RootDir):
    """
    load the entire cifar-10 data sets
    - parameters: root dir
    - outputs：X_tr, Y_tr: training set data and labels
               X_te, Y_te: test set data and labels
    """
    xs = []
    ys = []
    
    for b in range(1,6):
        f = os.path.join(RootDir,"data_batch_%d" % (b,) )
        print(f)
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)

    X_tr = np.concatenate(xs)
    Y_tr = np.concatenate(ys)   # [[...],[...]] to array

    del X, Y

    X_te, Y_te = load_CIFAR_batch(os.path.join(RootDir,"test_batch"))
    
    return X_tr, Y_tr, X_te, Y_te