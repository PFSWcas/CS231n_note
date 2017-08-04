"""
Platform: win10 64-bits
python: 3.6
"""
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

        self._null = word_to_idx['<NULL>']             # index for '<NULL>'
        self._start = word_to_idx.get('<START>', None) # index for '<START>'
        self._end = word_to_idx.get('<END>', None)     # index for '<END>'

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
            self.params[k] = v.astype(self.dtype)
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
        captions_out = captions[:, 1:]   # expected output

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
        #     from indices to vectors, giving an array of shape (N, T, D).          #
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
        # step 1: get initial hidden state
        initial_h = features.dot(W_proj) + b_proj  # of shape [N, H]

        # step 2: transform the words from indices to vectors
        embed_word, embed_word_cache = word_embedding_forward(captions_in, W_embed)  # of shape (N, T, D)

        # step 3: RNN or LSTM
        if self.cell_type == 'RNN':
            h, h_cache = RNN_forward(embed_word, initial_h, Wx, Wh, b)
        elif self.cell_type == 'LSTM':
            h, h_cache = LSTM_forward(embed_word, initial_h, Wx, Wh, b)
        
        # step 4: compute scores using a temporal affine transformation
        affine_forward_out, affine_forward_cache = temporal_affine_forward(h, W_vocab, b_vocab)

        # step 5: compute loss 
        loss, dscore = temporal_softmax_loss(affine_forward_out, captions_out, mask, verbose=False)

        # backprop 
        daffine_out, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dscore, affine_forward_cache)

        if self.cell_type == 'RNN':
            dword_vector, dh0, grads['Wx'], grads['Wh'], grads['b'] = RNN_backward(daffine_out, h_cache)
        elif self.cell_type == 'LSTM':
            dword_vector, dh0, grads['Wx'], grads['Wh'], grads['b'] = LSTM_backward(daffine_out, h_cache)
        
        grads['W_embed'] = word_embedding_backward(dword_vector, embed_word_cache)
        grads['W_proj'] = features.T.dot(dh0)
        grads['b_proj'] = np.sum(dh0, axis=0)

        return loss, grads
    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input features vectors. 

        At each timestep, we embed the current word, pass it and the previous hidden state to the RNN 
        to get the next hiiden state, use the hidden state to get scores for all vocab words, and 
        choose the word with highest score as the next word. The initial hidden state is computed 
        by applying an affine transform to the input iamges features, and the initial word is the 
        <START> token. 

        For LSTM you will also have to keep track of the cell statel. in that case, the inital cell
        state should be zero. 

        Inputs: 
          - features: array of input imagegs features of shape (N, D)
          - max_length: maximum length T of generated captions 
        - Returns:
          - captions: array of shape (N, max_length) giving sampled captions, where each element is
                      an integer in the range (0,V), the first element of captions should be the sampled 
                      word, not the <START> token.   
        """
        N = features.shape[0]
        captions = self._null * np.ones((N,max_length), dtype=np.int32)

        # Unpack parameters 
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # 
        # (1) Embed the previous word using the learned word embeddings           #
        # 
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # 
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # 
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: 
        # You will not be able to use the rnn_forward or lstm_forward             #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        ###########################################################################
        (N, D) = features.shape
        prev_h = features.dot(W_proj) + b_proj
        prev_c = np.zeros(prev_h.shape)

        # self._start is the index of the word '<START>'
        current_word_index = [self._start]*N 

        for i in range(max_length):
            
            x = W_embed[current_word_index]   # get word_vector from word_index 
            
            if self.cell_type == 'RNN':
                next_h, _ = RNN_step_forward(x, prev_h, Wx, Wh, b)
            elif self.cell_type == 'LSTM':
                next_h, next_c, _ = LSTM_step_forward(x, prev_h, prev_c, Wx, Wx, b)
                prev_c = next_c   # update cell state, of shape (N, H)
            prev_h = next_h       # update the hidden state, of shape (N, H)

            next_h = np.expand_dims(next_h, axis=1) # of shape (N, 1, H)
            score, _ = temporal_affine_forward(next_h, W_vocab, b_vocab)
            captions[:,i] = list(np.argmax(score, axis=2))
            current_word_index = captions[:,i]
        
        return captions 
