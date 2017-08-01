import numpy as np 

from CS231n.layers import *
from CS231n.RNN_layers import *

class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent neural network. 

    The RNN receives input vectors of size D, has a vocab size of V, works on sequences of length 
    T, has an RNN hidden dimension of H, uses word vectors of dimension W, and operates on minibatch
    of size N. 

    Note that we donot use any REGULARIZATION for the CaptionningRNN. 
    """
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,hidden_dim=128,cell_type='RNN',dtype=np.float32):
        """
        Construct a new CaptioningRNN instance. 

        Inputs: 
          - word_to_idx：A dictionary giving the vocabulary. It cantains V entries, and maps 
                         each string to a unique integer in the range [0,V-1]
          - input_dim: Dimension D of input image feature vectors. 
          - wordvec_dim: Dimension W of word vectors. 
          - hidden_dim: Dimension H for the hidden state of the RNN. 
          - cell_type: What type of RNN to use; either 'RNN' or 'LSTM'.
          - dtype: numpy datatype to use; use float32 for training and float64 
                   for numeric gradient checking.
        """
        if cell_type not in {'RNN', 'LSTM'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)
        
        self.cell_type = cell_type
        self.dtype = dtype 
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # initialize word vectros 
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100 

        # initialzie CNN -> hidden state projection parameters 
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # initialzie parameters for the RNN 
        dim_mul = {'LSTM': 4, 'RNN':1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul*hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # initialze output to vocab weights 
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # cast parameters to correct dtype 
        for k, v in self.params.items():
            self.params[k] = v.astype(self.stype)
    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and ground-truth captions for those images, 
        and use an RNN (or LSTM) to compute loss and gradients on all parameters. 
        
        Inputs:
          - features: input image features, of shape (N, D)
          - captions: ground-truth captions; an array of shape (N, T) where each element 
                      is in the range 0 <= y[i, t] < V.
        
        Returns a tuple of:
          - loss: scalar loss 
          - grads: dictionary of gradients parallel to self.params 
        """
        # NOTE:
        # cut captions into two pieces: captions_in has everything but the last word 
        # and will be input to the RNN; captions_out has everythong but the first word
        # and this is what we will expect the RNN to generate. These are offset by one 
        # relative to each other because the RNN should produce word(t+1) after 
        # receiving word t. The first element of captions_in will be the START token, 
        # and the first element of captions_out will be the fitst word. 
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # mask
        mask = (captions_out != self._null)

        # weight and bias for the affine transform from image features to initial hidden state. 
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN 
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation. 
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        #############################################################################
        # TODO: 
        # Implement the forward and backward passes for the CaptioningRNN.          #
        # In the forward pass you will need to do the following:                    #
        # 
        # (1) Use an affine transformation to compute the initial hidden state      #
        #     from the image features. This should produce an array of shape (N, H) #
        #                                                                           #
        # (2) Use a word embedding layer to transform the words in captions_in      #
        #     from indices to vectors, giving an array of shape (N, T, W).          #
        # 
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to     #
        #     process the sequence of input word vectors and produce hidden state   #
        #     vectors for all timesteps, producing an array of shape (N, T, H).     #
        # 
        # (4) Use a (temporal) affine transformation to compute scores over the     #
        #     vocabulary at every timestep using the hidden states, giving an       #
        #     array of shape (N, T, V).                                             #
        # 
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring   #
        #     the points where the output word is <NULL> using the mask above.      #
        #                                                                           #
        # In the backward pass you will need to compute the gradient of the loss    #
        # with respect to all model parameters. Use the loss and grads variables    #
        # defined above to store loss and gradients: 
        # grads[k] should give the gradients for self.params[k].                    #
        #############################################################################