"""
Created on 28 July, 2017
Reference: CS231n
Platform: win10 64-bits
python: 3.6
Object: 
 (1) train a vanilla recurrent neural networks
 (2) use them it to train a model that can generate novel captions for images
"""
from CS231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from CS231n.RNN_layers import *
from CS231n.classifiers.rnn import CaptioningRNN
from CS231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from CS231n.image_utils import image_from_url

import time, os, json
import numpy as np
import matplotlib.pyplot as plt

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