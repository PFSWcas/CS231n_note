import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - W: K x D array of weights
    - X: D x N array of training data. Data are D-dimensional colunmn vector.
         the number of training data is N.
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros(W.shape)

    #############################################################################
    # TODO                                                                      #
    # Compute the softmax loss and its gradient using explicit loops.           #
    # Store the loss in loss and the gradient in dW. If you are not             #
    # here, it is easy to run into numeric instability. Don't forget            #
    # the regularization                                                        #
    #############################################################################

    (K, D) = W.shape
    num_train = X.shape[1]

    for i in range(num_train):
        scores = W.dot(X[:, i])
        scores -= np.max(scores) # correct for numerical stability

        # Loss = -f_yi + log(sum(e^(f_j)))
        loss -= scores[y[i]]

        sum_exp = 0.0
        for s in scores:
            sum_exp += np.exp(s)

        for j in range(0,K):
            dW[j, :] += 1.0 / sum_exp * np.exp(scores[j]) * X[:,i].T
            if j == y[i]:
                dW[j, :] -= X[:,i].T
                
            
        loss += np.log(sum_exp)

    # Right now the loss is a sum over all training example, but we wat it to 
    # be an average instead so we divide by num_train.

    loss /= num_train

    # Average gradients as well
    dW /= num_train

    # Add regularization to the loss
    loss += 0.5 * reg * np.sum(W * W)

    # Add regularization to the gradient
    dW += reg * W

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as the  softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero 
    loss = 0.0
    dW = np.zeros(W.shape)

    #############################################################################
    # TODO                                                                      #
    # Compute the softmax loss and its gradient using no explicit loops.        #
    # Store the loss in loss and gredient in dW. If you are not careful here,   #
    # it is easy to run into numeric instability. Do not forget the             #
    # regularization                                                            #
    #############################################################################

    (K, D) = W.shape
    N = X.shape[1]

    scores = np.dot(W, X)
    scores -= np.max(scores)        # K x N

    y_mat = np.zeros(shape = (K, N))
    y_mat[y, range(N)] = 1

    # matrix of all zeros except for a single wx + log C value in each column
    # that corresponds to the quantity we need to substruct from each row of scores

    # correct_wx = np.multiply(y_mat, scores)  # extract f_yi

    # creat a single row of the correct wx_y + log C values fro each data point
    # sums_wy = np.sum(correct_wx, axis=0)  # sum over each colun

    exp_scores = np.exp(scores)
    sums_exp = np.sum(exp_scores, axis = 0)
    result = np.log(sums_exp) - scores[y, np.arange(0, scores.shape[1])]

    # result -= sums_wy
    loss = np.sum(result)

    # Right now the loss is a sum over all training example, but we wat it to 
    # be an average instead so we divide by num_train.
    loss /= float(N)

    # Add regularization to the loss
    loss += 0.5 * reg * np.sum(W * W)

    sums_exp_scores = np.sum(exp_scores, axis=0)
    sums_exp_scores = 1.0 / (sums_exp_scores + 1e-8)  # 1 x N

    dW = exp_scores * sums_exp_scores                 # K x N
    dW = np.dot(dW, X.T)
    dW -= np.dot(y_mat, X.T)

    dW /= float(N)

    dW += reg * W

    return loss, dW 





