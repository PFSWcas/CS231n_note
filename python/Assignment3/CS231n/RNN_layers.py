import numpy as np 

"""
This file defined layer types that are commonly used for recurrent neural nerworks. 
"""

def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function. 

  The input data has dimension D, the hidden state has dimension H, and we use a minibatch
  size of N. 

  Inputs:
    - x: Input data for this timestep, of shape (N, D). 
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H, )
  Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass. 
  """

  next_h, cache = None, None
  ##############################################################################
  # TODO: 
  # Implement a single forward step for the vanilla RNN. Store the next        #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  affine_output = prev_h.dot(Wh) + x.dot(Wx) + b
  next_h = np.tanh(affine_output)
  cache = (x, prev_h, Wx, Wh, affine_output, next_h)

  return next_h, cache 

def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN. 

  Inputs;
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass. 
  Return a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO:                                                                      #
  # Implement the backward pass for a single step of a vanilla RNN.                                                                           #
  # f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  # df(x) / d(x) = 1 - f^2(x)
  ##############################################################################    
  (x, prev_h, Wx, Wh, affine_output, next_h) = cache 

  daffine_output = dnext_h * (1 - next_h * next_h)

  dx = daffine_output.dot(Wx.T)
  dWx = x.T.dot(daffine_output)

  dprev_h = daffine_output.dot(Wh.T)   # pay attenion to this transpose
  dWh = prev_h.T.dot(daffine_output)

  db = np.sum(daffine_output, axis = 0)

  return dx, dprev_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden size of 
  H, and we work over a minibatch containing N sequences. After runing the RNN forward, we 
  return the hidden states for all timesteps. 

  Inputs:
   - x: Input data for the entire timeseries, of shape (N, T, D). 
   - h0: Initial hidden state, of shape (N, H)
   - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
   - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)  

  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  N, T, D = x.shape 
  _, H = h0.shape 
  h = np.zeros(N, T, H)
  cache = {}
  for t in range(T):
    if t==0:
      h[:, t, :], cache[t] = rnn_step_forward(x[:, t, :], h0, Wx, Wh, b)
    else:
      h[:, t, :], cache[t] = rnn_step_forward(x[:, t, :], h[:, t-1, :], Wx, Wh, b)
  
  return h, cache 

def rnn_backward(dh, back):
  """
  Compute the backward pass for a vanilla RNN over entire sequence of data. 

  Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: gradient of input-to-hidden weights, of shape (D, H)
    - dWh: gradient of hidden-to-hidden weights, of shape (N, H)
    - db: gradient of biases, of shap (H, )
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None 
  ##############################################################################
  # TODO: 
  # Implement the backward pass for a vanilla RNN running an entire            #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  N, T, H = dh.shape 
  x_slice, prev_h, Wx, Wh, affine_output, next_h = cache[T-1]
  N, D = x_slice.shape

  dx = np.zeros((N,T,D))
  dWx = np.zeros(Wx.shape)
  dWh = np.zeros(Wh.shape)
  db = np.zeros((H))
  dprev = np.zeros(prev_h.shape)

  for t in range(T-1, -1, -1):
    dx[:, t, :], dprev, dWx_local, dWh_local, db_local = rnn_step_backward(dh[:,t,:] + dprev, cache[t])
    dWx += dWx_local
    dWh += dWh_local
    db += db_local 
  
  dh0 = dprev 

  return dx, dh0, dWx, dWh
