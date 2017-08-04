"""
Created on 3 Aug., 2017
Reference: CS231n
Platform: win10 64-bits
python: 3.6
Object: 
 (1) train a vanilla recurrent neural networks
 (2) use them it to train a model that can generate novel captions for images
"""
import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from CS231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from CS231n.RNN_layers import *
from CS231n.Captioning_solver import CaptioningSolver
from CS231n.classifiers.rnn import CaptioningRNN
from CS231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from CS231n.image_utils import image_from_url


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

small_data = load_coco_data(max_train=50)

small_lstm_model = CaptioningRNN(
          cell_type='LSTM',
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          hidden_dim=512,
          wordvec_dim=512,
          dtype=np.float32,
        )

small_lstm_solver = CaptioningSolver(small_lstm_model, small_data,
           update_rule='Adam',
           num_epochs=50,
           batch_size=25,
           optim_config={
             'learning_rate': 5e-3,
           },
           lr_decay=0.995,
           verbose=True, print_every=10,
         )

small_lstm_solver.train()

# Plot the training losses
plt.plot(small_lstm_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()

for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = small_lstm_model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()

## To relieve overfitting, did:
## (1) incrase max_train from 50 to 500
## (2) decrease num_epoches from 50 to 25

small_data2 = load_coco_data(max_train=1000)

good_model = CaptioningRNN(
      cell_type='LSTM',
      word_to_idx=data['word_to_idx'],
      input_dim=data['train_features'].shape[1],
      hidden_dim=512,
      wordvec_dim=512,
      dtype=np.float32,
    )

good_solver = CaptioningSolver(good_model, small_data2,
       update_rule='Adam',
       num_epochs=25,
       batch_size=25,
       optim_config={
         'learning_rate': 5e-3,
       },
       lr_decay=0.995,
       verbose=True, print_every=50,
     )

good_solver.train()

# Plot the training losses
plt.plot(good_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()

for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(small_data2, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = good_model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()