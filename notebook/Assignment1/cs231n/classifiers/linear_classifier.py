
"""
This a class definition file that define:
LinearClassifier
LinearSVM
Softmax
"""
import numpy as np
class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
               batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: D x N array of training data. Each training point is a D-dimensional column vector.
        - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step
        - verbose: (boolean) If true, print progree during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration. 
        """
        dim, num_train = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            # initialize W
            self.W = np.random.randn(num_classes, dim) * 0.001

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            #######################################################################
            # TODO:
            # Sample batch_size elements from the training data and their         #
            # corresponding labels to use in this round of gradient descent.      #
            # Store the data in X_batch and their corresponding labels in y_batch #
            # after sampling X_batch should have shape (batch_size, dim), y_batch #
            # should have shape (batch_size,)                                     #
            #######################################################################
            rand_idx = np.random.choice(num_train, batch_size)
            X_batch = X[:, rand_idx]
            y_batch = y[rand_idx]

            # evaluta loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #######################################################################
            # TODO                                                                #
            # Update the weights using the gradient and the learning rate.        #
            #######################################################################
            self.W += -1 * learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classfier to predict labels for data points.
        
        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional vector.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional array of length N, 
                  and each element is an integer giving the predicted class.
        """
        y_pred = np.zeros(X.shape[1])
        scores = np.dot(self.W, X)
        y_pred = scores.argmax(axis=0)
        
        return y_pred
        
    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivatives.
        Subclasses will override this.

        Inputs:
        - X_batch: D x N dimensional array of data; each column is a D-dimensional vector;
        - y_batch: 1-dimensional array of length N with 0...K-1 for K classes.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a signle float
        - gradient with respect to self.W; an rray of the same shape as W
        """

        pass

from cs231n.classifiers.linear_svm import svm_loss_vectorized        
#from linear_svm import svm_loss_vectorized
class LinearSVM(LinearClassifier):
    """ A subclass that uses the multicalss SVM loss function """
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


from cs231n.classifiers.softmax import softmax_loss_vectorized        
#from softmax import softmax_loss_vectorized
class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax and Cross-entropy loss function """
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

        

    
