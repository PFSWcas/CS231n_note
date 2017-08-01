"""
Platform: win10 64-bits
python: 3.6
"""
import numpy as np
"""
This file implements various first-order update rules that are commonly used for 
training neural networks. 
Each update rule accepts current weights and the gradient of the loss with 
respect to those weights and produces the next set of weights. 
Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
 - w: a numpy array giving current weights. 
 - dw: a numpy array of the same shape as w giving the gradient of the loss
       with respect to w. 
- config: A dictionary containing hyperparameter values such as learning rate,
  momentum, etc. If the update rule requires caching values over many iterations,
  then config will also hold these cached values.

Returns:
- next_w: The next point after the update.
- config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and setting
next_w equal to w.
"""

def SGD(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent. 

    config format:
    - learning_rate: Scalar learning rate. 
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)  # set the default value of learning rate.

    w -= config['learning_rate'] * dw 
    return w, config

def SGD_momentum(w, dw, config=None):
    """
    Performs stochastic gradeint descent with moment. 

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + next_v

    config['velocity'] = next_v

    return next_w, config

def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of the squared 
    gradient values to set adaptive per-parameter learning rates. 

    config format:
    - learning_rate: Scalar learning rate. 
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache. 
    - epsilon: Small scalar used for smoothing to avoid dividing by zero. 
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config={}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    config['cache'] = config['cache'] * config['decay_rate'] + (1-config['decay_rate']) * dx**2
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']+config['epsilon']))

    return next_x, config

def Adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporate moving averages of both the 
    gradient and its square and a bias correction term. 

    config format: 
    - learning_rate: Scalar learning rate. 
    - beta1: decay rate for moving average of first moment of gradient. 
    - beta2: decay rate for moving avarage of second moment of gradient. 
    - epsilon: Small scalar used for smoothing to avoid divding by zero. 
    - m: moving average of gradient. 
    - v: moving average of squared gradient. 
    - t: Iteration number. 
    """
    if config is None:
        config={}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']

    # Value after the update
    config['t'] += 1
    config['m'] = beta1 * config['m'] + (1- beta1) * dx
    config['v'] = beta2 * config['v'] + (1- beta2) * dx**2
    mt_hat = config['m'] / (1 - (beta1)**config['t'])
    vt_hat = config['v'] / (1 - (beta2)**config['t'])
    next_x = x - learning_rate * mt_hat / (np.sqrt(vt_hat + epsilon))

    return next_x, config

