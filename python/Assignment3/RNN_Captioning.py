"""
Created on 28 July, 2017
Reference: CS231n
Platform: win10 64-bits
python: 3.6
Object: 
 (1) train a vanilla recurrent neural networks
 (2) use them it to train a model that can generate novel captions for images
"""
#%% Basic settings
from CS231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from CS231n.RNN_layers import *
from CS231n.classifiers.rnn import CaptioningRNN
from CS231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from CS231n.image_utils import image_from_url

import time, os, json
import numpy as np
import matplotlib.pyplot as plt

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def check_loss(N, T, V, p):
  x = 0.001 * np.random.randn(N, T, V)
  y = np.random.randint(V, size=(N, T))
  mask = np.random.rand(N, T) <= p
  print(temporal_softmax_loss(x, y, mask)[0])

if __name__ == '__main__':
  #%% Microsoft COCO
  # Load COCO data from disk; this returns a dictionary
  # We'll work with dimensionality-reduced features for this notebook, but feel
  # free to experiment with the original features by changing the flag below.
  data = load_coco_data(pca_features=True)

  # Print out all the keys and values from the data dictionary
  for k, v in data.items():
    if type(v) == np.ndarray:
      print(k, type(v), v.shape, v.dtype)
    else:
      print(k, type(v), len(v))
  """    
  #%% Look at the data
  # Sample a minibatch and show the images and captions
  batch_size = 3

  captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
  for i, (caption, url) in enumerate(zip(captions, urls)):
    plt.imshow(image_from_url(url))
    plt.axis('off')
    caption_str = decode_captions(caption, data['idx_to_word'])
    plt.title(caption_str)
    plt.show()
  """
  #%% Recurrent Neural Networks

  #  Vanilla RNN: step forward
  N, D, H = 3, 10, 4

  x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
  prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
  Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
  Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
  b = np.linspace(-0.2, 0.4, num=H)

  next_h, _ = RNN_step_forward(x, prev_h, Wx, Wh, b)
  expected_next_h = np.asarray([
    [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
    [ 0.66854692,  0.79562378,  0.87755553,  0.92795967],
    [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])

  print('next_h error: ', rel_error(expected_next_h, next_h))

  #  Vanilla RNN: step backward
  N, D, H = 4, 5, 6
  x = np.random.randn(N, D)
  h = np.random.randn(N, H)
  Wx = np.random.randn(D, H)
  Wh = np.random.randn(H, H)
  b = np.random.randn(H)

  out, cache = RNN_step_forward(x, h, Wx, Wh, b)

  dnext_h = np.random.randn(*out.shape)

  fx = lambda x: RNN_step_forward(x, h, Wx, Wh, b)[0]
  fh = lambda prev_h: RNN_step_forward(x, h, Wx, Wh, b)[0]
  fWx = lambda Wx: RNN_step_forward(x, h, Wx, Wh, b)[0]
  fWh = lambda Wh: RNN_step_forward(x, h, Wx, Wh, b)[0]
  fb = lambda b: RNN_step_forward(x, h, Wx, Wh, b)[0]

  dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
  dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
  dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
  dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
  db_num = eval_numerical_gradient_array(fb, b, dnext_h)

  dx, dprev_h, dWx, dWh, db = RNN_step_backward(dnext_h, cache)

  print( 'dx error: ', rel_error(dx_num, dx))
  print( 'dprev_h error: ', rel_error(dprev_h_num, dprev_h))
  print( 'dWx error: ', rel_error(dWx_num, dWx))
  print( 'dWh error: ', rel_error(dWh_num, dWh))
  print( 'db error: ', rel_error(db_num, db))

  # Vanilla RNN: forward
  N, T, D, H = 2, 3, 4, 5

  x = np.linspace(-0.1, 0.3, num=N*T*D).reshape(N, T, D)
  h0 = np.linspace(-0.3, 0.1, num=N*H).reshape(N, H)
  Wx = np.linspace(-0.2, 0.4, num=D*H).reshape(D, H)
  Wh = np.linspace(-0.4, 0.1, num=H*H).reshape(H, H)
  b = np.linspace(-0.7, 0.1, num=H)

  h, _ = RNN_forward(x, h0, Wx, Wh, b)
  expected_h = np.asarray([
    [
      [-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],
      [-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],
      [-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],
    ],
    [
      [-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],
      [-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],
      [-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]])
  print('h error: ', rel_error(expected_h, h))

  # Vanilla RNN: backward
  N, D, T, H = 2, 3, 10, 5

  x = np.random.randn(N, T, D)
  h0 = np.random.randn(N, H)
  Wx = np.random.randn(D, H)
  Wh = np.random.randn(H, H)
  b = np.random.randn(H)

  out, cache = RNN_forward(x, h0, Wx, Wh, b)

  dout = np.random.randn(*out.shape)

  dx, dh0, dWx, dWh, db = RNN_backward(dout, cache)

  fx = lambda x: RNN_forward(x, h0, Wx, Wh, b)[0]
  fh0 = lambda h0: RNN_forward(x, h0, Wx, Wh, b)[0]
  fWx = lambda Wx: RNN_forward(x, h0, Wx, Wh, b)[0]
  fWh = lambda Wh: RNN_forward(x, h0, Wx, Wh, b)[0]
  fb = lambda b: RNN_forward(x, h0, Wx, Wh, b)[0]

  dx_num = eval_numerical_gradient_array(fx, x, dout)
  dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
  dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
  dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
  db_num = eval_numerical_gradient_array(fb, b, dout)

  print('dx error: ', rel_error(dx_num, dx))
  print('dh0 error: ', rel_error(dh0_num, dh0))
  print('dWx error: ', rel_error(dWx_num, dWx))
  print('dWh error: ', rel_error(dWh_num, dWh))
  print('db error: ', rel_error(db_num, db))

  # Word embedding: forward
  # In deep learning systems, we commonly represent words using vectors. 
  # Each word of the vocabulary will be associated with a vector, 
  # and these vectors will be learned jointly with the rest of the system.
  N, T, V, D = 2, 4, 5, 3

  x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
  W = np.linspace(0, 1, num=V*D).reshape(V, D)

  out, _ = word_embedding_forward(x, W)
  expected_out = np.asarray([
  [[ 0.,          0.07142857,  0.14285714],
    [ 0.64285714,  0.71428571,  0.78571429],
    [ 0.21428571,  0.28571429,  0.35714286],
    [ 0.42857143,  0.5,         0.57142857]],
  [[ 0.42857143,  0.5,         0.57142857],
    [ 0.21428571,  0.28571429,  0.35714286],
    [ 0.,          0.07142857,  0.14285714],
    [ 0.64285714,  0.71428571,  0.78571429]]])

  print('out error: ', rel_error(expected_out, out))

  # Word embedding: backward
  N, T, V, D = 50, 3, 5, 6

  x = np.random.randint(V, size=(N, T))
  W = np.random.randn(V, D)

  out, cache = word_embedding_forward(x, W)
  dout = np.random.randn(*out.shape)
  dW = word_embedding_backward(dout, cache)

  f = lambda W: word_embedding_forward(x, W)[0]
  dW_num = eval_numerical_gradient_array(f, W, dout)

  print('dW error: ', rel_error(dW, dW_num))

  # Temporal Affine layer
  # At every timestep we use an affine function to transform the RNN hidden vector 
  # at that timestep into scores for each word in the vocabulary.
  # Gradient check for temporal affine layer
  N, T, D, M = 2, 3, 4, 5

  x = np.random.randn(N, T, D)
  w = np.random.randn(D, M)
  b = np.random.randn(M)

  out, cache = temporal_affine_forward(x, w, b)

  dout = np.random.randn(*out.shape)

  fx = lambda x: temporal_affine_forward(x, w, b)[0]
  fw = lambda w: temporal_affine_forward(x, w, b)[0]
  fb = lambda b: temporal_affine_forward(x, w, b)[0]

  dx_num = eval_numerical_gradient_array(fx, x, dout)
  dw_num = eval_numerical_gradient_array(fw, w, dout)
  db_num = eval_numerical_gradient_array(fb, b, dout)

  dx, dw, db = temporal_affine_backward(dout, cache)

  print('dx error: ', rel_error(dx_num, dx))
  print('dw error: ', rel_error(dw_num, dw))
  print('db error: ', rel_error(db_num, db))
  
  # Temporal Softmax loss
  # In an RNN language model, at every timestep we produce a score for each word 
  # in the vocabulary. We know the ground-truth word at each timestep, so we use 
  # a softmax loss function to compute loss and gradient at each timestep. 
  # We sum the losses over time and average them over the minibatch.
  # Sanity check for temporal softmax loss

  N, T, V = 100, 1, 10
    
  check_loss(100, 1, 10, 1.0)   # Should be about 2.3
  check_loss(100, 10, 10, 1.0)  # Should be about 23
  check_loss(5000, 10, 10, 0.1) # Should be about 2.3

  # Gradient check for temporal softmax loss
  N, T, V = 7, 8, 9

  x = np.random.randn(N, T, V)
  y = np.random.randint(V, size=(N, T))
  mask = (np.random.rand(N, T) > 0.5)

  loss, dx = temporal_softmax_loss(x, y, mask, verbose=False)

  dx_num = eval_numerical_gradient(lambda x: temporal_softmax_loss(x, y, mask)[0], x, verbose=False)

  print('dx error: ', rel_error(dx, dx_num))

  # RNN for image captioning
  N, D, W, H = 10, 20, 30, 40
  word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
  V = len(word_to_idx)
  T = 13

  model = CaptioningRNN(word_to_idx, 
                        input_dim=D,    # Dimension D of input image feature vectors
                        wordvec_dim=W,  #  Dimension W of word vectors.
                        hidden_dim=H,   # Dimension H for the hidden state of the RNN
                        cell_type='RNN',
                        dtype=np.float64)
  # set all model parameters to fixed values 
  for k, v in model.params.items():
    model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

  features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)
  captions = (np.arange(N * T) % V).reshape(N, T)
  loss, grads = model.loss(features, captions)
  expected_loss = 9.83235591003

  print('loss: ', loss)
  print('expected loss: ', expected_loss)
  print('difference: ', abs(loss - expected_loss))