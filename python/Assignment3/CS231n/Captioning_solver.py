import numpy as np 
from CS231n import optim 
from CS231n.coco_utils import sample_coco_minibatch

class CaptioningSolver(object):
    """
    A CaptioningSolver encapsulates all the logic necessary for training image
    captioning models. The CaptioningSolver performs stochastic gradient descent
    using different update rules defined in optim.py 

    The solver accepts both training and validation data and labels so it can 
    periodically check classification accuracy on both training and validation data
    to watch out for overfitting. 

    To train a model, you will first construct a CaptioningSolver instance, passing the model,
    dataset, and various options (learning_rate, batch size, etc) to the constructor. You will 
    then call the train() method to run the optimization procedure and train the model. 

    Afer the train() method returns, model.params will contain the parameters that performed 
    best on the validation set over the course of training. In addition, the instance variable 
    solver.loss_history will contain a list of all losses encountered during training and 
    the instance variables solver.train_acc_history and solver.val_acc_history will be lists containing
    the accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = load_coco_data()
    model = mymodel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                              update_rule='sgd',
                              optim_config={
                                  'learning_rate': 1e-3
                              },
                              lr_decay = 0.95,
                              num_epoches = 5, batch_size = 100,
                              print_every = 100)
    solver.train()

    A CaptioningSolver works on a model object that must confirm to the following API:
    - model.params must be a dictionary mapping string parameter names to numpy arrays containing
      parameter values. 
    - model.loss(features, captions) must be a function that computes trainging-timte loss and gradients, with the
      following inpts and outputs:
      Inputs:
        - features: array giving a minibatch of features of images, of shape (N, D)
        - captions: array of captions for those images, of shape (N, T) where each element is in range (0, V)
      Returns:
        - loss； scalar giving the loss 
        - grads: Dictionary with the same keysas self.params mapping parameters names to gradients of the loss
                W.R.T. those parameters. 
    """
    def __init__(self, model, data, **kwargs):
        """
        Construct a new CaptioningSoler instance. 

        Required arguments:
          - model: A model object conforming to the API described above. 
          - data: A dictionary of training and validation from load_coco_data. 
        Optional arguments:
          - update_rules: A string giving the name of an update rule in optim.py 
          - optim_config: A dictionary containing hyperparameters that will be
                          passed to the chosen update rule. Each update rule requires different
                          hyperparameters (see optim.py) but all update rules require a
                          'learning_rate' parameter so that should always be present.
          - lr_decay: A scalar for learning rate decay; after each epoch the learning
                      rate is multiplied by this value. 
          - batch_size: size of minibatches used to compute loss and gradient during training. 
          - print_every: Integer; training losses will be printed every print_every iterations.
          - verbose: Boolean; if set to false then no output will be printed during training.
        """
        self.model = model 
        self.data= data 

        # unpack keyword arguments 
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay',1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epoches = kwargs.pop('num_epoches', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True) 

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)
        
        # make sure the update rule exists, then replace the string 
        # name with the actual function.
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. 
        """
        # set up some variables for book-keeping 
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter 
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k,v in self.optim_config.items()}
            self.optim_configs[p] = d 
    
    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manully. 
        """
        # make a minibatch of training data 
        minibath = sample_coco_minibatch(self.data, 
                                        batch_size=self.batch_size,
                                        split = 'train')
        captions, features, urls = minibatch
        # Compute loss and gradient
        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config
    
    def check_accuracy(self, x, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data. 

        Inputs:
          - x: Array of data, of shape (N, d_1, ..., d_k)
          - y: array of labels, of shape (N, )
          - num_samples: If not None, subsample the data and only test the model
                         on num_samples datapoints.
          - batch_size: Split X and y into batches of this size to avoid using too much memory.
        Returns:
          - acc: scalar giving the fraction of instances that were correctly classified by the model. 
        """
        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = np.int(N / batch_size)
        if N % batch_size != 0:
            num_batches += 1
        
        y_pred = []
        for i in xrange(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))

        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc
    def train(self):
        """
        Run optimization to train the model. 
        """
        num_train = self.data['train_captions'].shape[0]
        iterations_per_epoch = max(np.int(num_train/self.batch_size), 1)
        num_iterations = self.num_epoches * iterations_per_epoch

        for t in range(num_iterations):
            self._step()
        
        # may print the training loss
        if self.verbose and t % self.print_every == 0:
            print('(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1]))
        
        # At the end of every epoch, increment the epoch counter and decay the learning rate.
        epoch_end = (t+1) % iterations_per_epoch == 0
        if epoch_end:
            self.epoch += 1
            for p in self.optim_configs:
                self.optim_configs[p]['learning_rate'] *= self.lr_decay
        
        # Check train and val accuracy on the first iteration, the last
        # iteration, and at the end of each epoch.