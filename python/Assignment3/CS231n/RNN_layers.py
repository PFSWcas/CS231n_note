import numpy as np 

"""
This file defined layer types that are commonly used for recurrent neural nerworks. 
"""

def RNN_step_forward(x, prev_h, Wx, Wh, b):
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

def RNN_step_backward(dnext_h, cache):
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

def RNN_forward(x, h0, Wx, Wh, b):
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
      h[:, t, :], cache[t] = RNN_step_forward(x[:, t, :], h0, Wx, Wh, b)
    else:
      h[:, t, :], cache[t] = RNN_step_forward(x[:, t, :], h[:, t-1, :], Wx, Wh, b)
  
  return h, cache 

def RNN_backward(dh, back):
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
    dx[:, t, :], dprev, dWx_local, dWh_local, db_local = RNN_step_backward(dh[:,t,:] + dprev, cache[t])
    dWx += dWx_local
    dWh += dWh_local
    db += db_local 
  
  dh0 = dprev 

  return dx, dh0, dWx, dWh

def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. 
  We assume a vocabulary of V words, assigning each to a vector of dimension D. 

  Inputs;
    - x: integer array of shape (N, T) giving indices of words. 
         Each element idx of x must be in the range 0 <= idx < V. 
    - W:  Weight matrix of shape (V, D) giving word vectors for all words. 
  
  Returns a tuple of:
    - out: array of shape (N, T, D) giving word vectors for all input words. 
    - cache: values needed for the backward pass
  """
  out, cache = None, None 

  N, T = x.shape 
  V, D = W.shape 

  out = np.zeros((N, T, D))
  for n in range(N):
    for t in range(T):
      out[n, t, :] = W[x[n,t],:]
  
  cache = (x, v, D)

  return out, cache

def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.

  HINT: Look up the function np.add.at
  
  Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
  
  Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None 
  x, V, D = cache 
  dW = np.zeros((V, D))
  np.add.at(dW, x, dout)

  return dW

def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.

  sigmoid(x) = 
                  1/(1+e^(-x)), when x >= 0
                  e^x/(1+e^x), when x < 0

  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)

  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])

  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]

  return top / (1 + z)

def LSTM_step_forward(x, prev_h, prev_c, Wh, b):
  """
  Forward pass for a single timestep of an LSTM. 

  The input data has dimension D, the hidden state has dimension H, and we use a 
  minibatch size of N. 

  Inputs:
    - x: input data, of shape (N, D)
    - prev_h: previous hidden state, of shape (N, H)
    - prev_c: previouos cell state, of shape (N, H)
    - Wx: input-to-hidden weights, of shape (D, 4*H)
    - Wh: hidden-to-hidden weigths, of shape (H, 4*H)
    - b: Biases, of shape (4H, )

  Returns a tuple of:
    - next_h: next hidden state, of shape (N, H)
    - next_c: next cell state, of shape (N, H)
    - cache: tuple of values needed for backward pass.   
  """
  next_h, next_c, cache = None, None, None
  H = prev_h.shape[1]

  a = x.dot(Wx) + prev_h.dot(Wh) + b # of shape (N, 4H)
  i = sigmoid(a[:, : H])
  f = sigmoid(a[:, H:2*H])
  o = sigmoid(a[:, 2*H:3*H])
  g = np.tanh(a[:, 3*H:])

  next_c = f*prev_c + i*g 
  next_h = o*np.tanh(next_c)
  cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_h, next_c)

  return next_h, next_c, cache 

def LSTM_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of LSTM. 

  Inputs:
    - dnext_h: gradients of next hidden state, of shape (N, H)
    - dnext_c: gradients of next cell state, of shape (N, H)
    - cache: values from forward pass. 
  
  Returns a tuple of:
    - dx: gradient of input data, of shape (N, D)
    - dprev_h: gradient of previous hiden state, of shape (N, H)
    - dprev_c: gradient of previous cell state, of shape (N, H)
    - dWx: gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
  """
  dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
  (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_h, next_c) = cache 

  dnext_c = dnext_c + o*(1-np.tanh(next_c)**2)*dnext_h
  di = dnext_c * g
  df = dnext_c*prev_c 
  do = dnext_h*np.tanh(next_c)
  dg = dnext_c*i 
  dprev_c = f * dnext_c
  da = np.hstack((i*(1-i)*di, f*(1-f)*df, o*(1-o)*do, (1-g**2)*dg))

  dx = da.dot(Wx.T)
  dprev_h = da.dot(Wh.T)
  dWx = x.T.dot(da)
  dWh = prev_h.T.dot(da)
  db = np.sum(da, axis = 0)

  return dx, dprev_h, dprev_c, dWx, dWh, db

def LSTM_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input sequence composed
  of T vectors, each of dimension D. The LSTM uses a hidden size of H, and we work over a minibatch
  containing N sequences. After running the LSTM forward, we return the hidden states for all timesteps. 

  Note that the initial cell stete is passed as input, but the inital cell state is set to zero. 
  Also note that the cell state is not returned; it is an internal variable to the LSTM and 
  is not accessed from outside. 

  Inputs:
    - x； input data of shape (N, T, D)
    - h0: initial hidden state of shape (N, H)
    - Wx: weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: biases of shape (4H, )
  
  Returns a tuple of:
    - h: hidden states for all timesteps of al sequences, of shape (N, T, H)
    - cache: values needed for the backward pass. 
  """
  h, cache = None, None
  
  N, T, D = x.shape 
  _, H = h0.shape 
  
  h = np.zeros((N, T, H))
  c = np.zeros((N, T, H))
  c0 = np.zeros((N, H)) # Cell state is initialized to zeros
  cache = {}
  for t in range(T):
    if t == 0:
      h[:, t, :], c[:, t, :], cache[t] = LSTM_step_forward(x[:,t,:], h0, c0, Wx, Wh, b)
    else:
      h[:, t, :], c[:, t, :], cache[t] = LSTM_step_forward(x[:,t,:], h[:, t-1, :], c[:, t-1, :], Wx, Wh, b)
  
  return h, cache 

def LSTM_backeard(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data. 

  Inputs:
    - dh: upstream gradients of hidden states, of shape (N, T, H)
    - cache: values from the forward pass. 
  Returns a tuple of:
    - dx: gradient of input data, of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None 
  N, T, H = dh.shape
  x_slice, prev_h, prev_c, Wx, Wh, i, f, o, g, next_h, next_c = cache[T-1]

  N, D = x_slice.shape
  dx = np.zeros((N, T, D))
  dWx = np.zeros(Wx.shape)
  dWh = np.zeros(Wh.shape)
  db = np.zeros((4*H))

  dprev_h = np.zeros(prev_h.shape)
  dprev_c = np.zeros(prev_c.shape) 

  for t in range(T-1, -1, -1):
    dx[:, t, :], dprev_h, dprev_c, dWx_local, dWh_local, db_local = LSTM_step_backward(dh[:,t,:]+dprev_h, dprev_c, cache[t])
    
    dWx+=dWx_local
    dWh+=dWh_local
    db +=db_local

  dh0 = dprev 

  return dx, dh0, dWx, dWh, db

def temporal_affine_forward(x, W, b):
  """
  Forward pass for a temporal affine layer. 
  The input is a set of D-dimensional vectors arranged into a minibatch of N timeseries, each of 
  length T. We use an affine function to tranform each of those vectors into a new vector of dimension
  M.

  Inputs:
    - x: input data of shape (N, T, D)
    - w: weights of shape (D, M)
    - b: biases of shape (M, )

  Returns a tuple of:
    - out: output data of shape (N, T, M)
    - cache: values needed for the backward pass.  
  """
  N, T, D = x.shape
  M = b.shape 
  out = x.reshape(N*T, D).dot(W).reshape(N, T, M) + b 
  cache = x, W, b, out 
  return out, cache

def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

  Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
  """
  x, W, b, out = cache 
  N, T, D = x.shape 
  M = b.shape 

  dx = dout.reshape(N*T, M).dot(W.T).reshape(N, T, D)
  dW = dout.reshape(N*T, M).T.dot(x.reshape(N*T, D)).T 
  db = dout.sum(axis=(0, 1))

  return dx, dw, db

def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNS. 
  We assume that we are making predictions over a vocabulary of size V for each timestep of a 
  timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary 
  elements at all timesteps, and y gives the indices of the ground-truth element at each timestep. 
  We use a cross-entropy loss at each timestep, summing the loss over all the timesteps and averaging
  across the minibatch. 

  As an additioanl complication, we may want to ignore the model output at some timesteps, since sequences
  of different length may have been combined into a minibatch and padded with NULL tokens. The optional mask
  argument tells us which elemnts should contribute to the loss. 

  Inputs:
    - x: input scores, of shape (N, T, V) 
    - y: Ground-truth indices, of shape (N, T) where each element is in the range 0 <= y[i,t] < V 
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not the scores at x[i,t] 
            should contribute to the final loss. 

  Returns a tuple of:
    - loss: scalar giving loss 
    - dx: gradient of loss with respect to scores x.  
  """
  N, T, V = x.shape 

  x_flat = x.reshape(N*T, V)
  y_flat = y.reshape(N*T)
  mask_flat = mask.reshape(N*T)

  probs = np.exp(x_flat - np.max(x_flat, axis = 1, keepdims=True))  # of shape (N*T, V)
  probs /= np.sum(probs, axis=1, keepdims=True)                     # of shape (N*T, V)
  
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N*T), y_flat])) / N

  dx_flat = probs.copy 
  dx_flat[np.range(N*T), y_flat] -= 1
  dx_flat /= N 
  dx_flat *= mask[:,None]

  if verbose: print('dx_flat shape: ', dx_flat.shape)

  dx = dx_flat.reshape(N, T, V)

  return loss, dx