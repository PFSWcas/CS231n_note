from CS231n.layers import *
from CS231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU 

    Inputs:
     - x: Input to the affine layer
     - w, b: weights for the affine layer

     Returns a tuple of:
     - out: output from the ReLU 
     - cache: Object to give to the backward pass
    """

    a, fc_cache = affine_forward(x, w, b)   # affine_forward is form layers.py
    out, relu_cache = ReLU_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = ReLU_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db 

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a BN and ReLU 

    Inputs:
     - x: Inputs to the affine layer
     - w, b: Weights for the affine layer
     - gamma, beta : Weight for the batch norm regularization
     - bn_params : Contain variable use to BN, running_mean and var
    Returns a tuple of:
    - Out: Output from the ReLU 
    - cache: Object to give to the backward pass
    """
    h, h_cache = affine_forward(x, w, b)
    hnorm, hnorm_cache = batchnorm_forward(h, gamma, beta, bn_param)
    hnorm_relu, relu_cache = ReLU_forward(hnorm)

    cache = (h_cache, hnorm_cache, relu_cache)

    return hnorm_relu, cache

def affine_norm_relu_backward(dout, cache):
    """
    Backward pass for the affine-BN-relu convenience layer
    """
    h_cache, hnorm_cache, relu_cache = cache

    dhnorm_relu = ReLU_backward(dout, relu_cache)
    dhnorm, dgamma, dbeta = batchnorm_backward_alt(dhnorm_relu, hnorm_cache)
    dx, dw, db = affine_backward(dhnorm, h_cache)

    return dx, dw, db, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU. 

    Inputs:
     - x: Input to the convolutional layer
     - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
     -out: output from the ReLU 
     - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = ReLU_forward(a)

    cache = (conv_cache, relu_cache)

    return out, cache

def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu layer
    """
    conv_cache, relu_cache = cache
    da = ReLU_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)

    return dx, dw, db

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convience layer that performs a convolution, a ReLU, and a maxing-pool 

    Inputs:
     - x: Input to the convolutional layer
     - w, b, conv_param: Weights and parameters for the convolutional layer
     - pool_param: Parameters for the pooling layer
    
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = ReLU_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)

    cache = (conv_cache, relu_cache, pool_cache)

    return out, cache

def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool layer
    """
    conv_cache, relu_cache, pool_cache = cache

    ds = max_pool_backward_fast(dout, pool_cache)
    da = ReLU_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def conv_norm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
    """
    Convenience layer that performs a convolution, spatial BN, a ReLU, and
    a max pooling. 

    Inputs:
     - x: Input to the convolutional layer
     - w, b, conv_param: Weights and parameters for the convolutional layer
     - pool_param: Parameters for the pooling layer

    Returns a tuple of:
     - out: Output from the pooling layer
     - cache: Object to give to the backward pass
    """
    conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
    #norm, norm_param = batchnorm_forward(conv, gamma, beta, bn_param)
    # test spatial_batchnorm_forward
    norm, norm_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
    out, relu_cache = ReLU_forward(norm)

    cache = (conv_cache, norm_cache, relu_cache)
    return out, cache

def conv_norm_relu_backward(dout, cache):
    """
    Backward pass for the conv-BN-relu layer
    """
    conv_cache, norm_cache, relu_cache = cache

    drelu = ReLU_backward(dout, relu_cache)
    #dnorm, dgamma, dbeta = batchnorm_backward_alt(drelu, norm_cache)
    # test spatial_batchnorm_bacward
    dnorm, dgamma, dbeta = spatial_batchnorm_backward(drelu, norm_cache)
    dx, dw, db = conv_backward_fast(dnorm, conv_cache)

    return dx, dw, db, dgamma, dbeta

def conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
    """
    Convenience layer that performs a convolution, spatial BN, a ReLU, and a max pooling. 

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
    norm, norm_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
    relu, relu_cache = ReLU_forward(norm)
    out, pool_cache = max_pool_forward_fast(relu, pool_param)

    cache = (conv_cache, norm_cache, relu_cache, pool_cache)

    return out, cache


def conv_norm_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, norm_cache, relu_cache, pool_cache = cache

    dpool = max_pool_backward_fast(dout, pool_cache)
    drelu = ReLU_backward(dpool, relu_cache)
    dnorm, dgamma, dbeta = spatial_batchnorm_backward(drelu, norm_cache)
    dx, dw, db = conv_backward_fast(dnorm, conv_cache)

    return dx, dw, db, dgamma, dbeta
    