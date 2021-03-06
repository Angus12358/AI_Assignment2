from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def affine_bn_relu_forward(x , w , b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta =  batchnorm_backward_alt(dbn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_norm_relu_dropout_forward(x, w, b, normalization=None,
                                     gamma=None, beta=None,
                                     bn_params=None,
                                     dropout=None,
                                     dropout_param=None):
    """
    affine -- [bn/layer norm] -- _relu -- [dropout] forward pass.
    Input:
        x: shape of (N, D)
        w: affine's weight parameter, shape of (D, hidden_dim)
        b: affine's bias parameter, shape of (hidden_dim,)
        normalization: What type of normalization the network should use.
        Valid values are "batchnorm", "layernorm", or None for no normalization (the default).
        gamma: batch/layer norm's scale parameters
        beta: batch/layer norm's shift parameters
        bn_params:
            mode: train, or test
        dropout: True or False, whether use dropout or not
        dropout_params:
            mode: 'train', or 'test'
            p: the probability of keeping the neuron
            seed:
    Return:
         out: shape of (N, hidden_dim)
         cache: tuple of (affine_cache, norm_cache, relu_cache, dropout_cache),
           to be used in the backward pass
    """

    out, cache = None, None
    affine_out, affine_cache = affine_forward(x, w, b)


    if normalization == 'batchnorm':
        norm_out, norm_cache = batchnorm_forward(affine_out, gamma, beta, bn_params)
    elif normalization == 'layernorm':
        norm_out, norm_cache = layernorm_forward(affine_out, gamma, beta, bn_params)
    else:
        norm_out, norm_cache = affine_out, affine_cache

    relu_out, relu_cache = relu_forward(norm_out)

    if dropout is True:
        dropout_out, dropout_cache = dropout_forward(relu_out, dropout_param)
    else:
        dropout_out, dropout_cache = relu_out, relu_cache

    out = dropout_out
    cache = (affine_cache, norm_cache, relu_cache, dropout_cache,
             normalization, dropout)

    return out, cache


def affine_norm_relu_dropout_backward(dout, cache):
    """
    affine -- [bn/layer norm] -- _relu -- [dropout] backward pass.
    Input:
        dout: shape of (N, D)
        cache: tuple of (affine_cache, norm_cache, relu_cache, dropout_cache,
             normalization, dropout)
    Return:
         dx, dw, db, dgamma, dbeta
    """

    dx, dw, db, dgamma, dbeta = None, None, None, None, None
    affine_cache, norm_cache, \
    relu_cache, dropout_cache, \
    normalization, dropout = cache

    if dropout is True:
        ddropout = dropout_backward(dout, dropout_cache)
    else:
        ddropout = dout

    drelu_out = relu_backward(ddropout, relu_cache)

    if normalization == 'batchnorm':
        dnorm_out, dgamma, dbeta = batchnorm_backward(drelu_out, norm_cache)
    elif normalization == 'layernorm':
        dnorm_out, dgamma, dbeta = layernorm_backward(drelu_out, norm_cache)
    else:
        dnorm_out, dgamma, dbeta = drelu_out, 0, 0

    dx, dw, db = affine_backward(dnorm_out, affine_cache)

    return dx, dw, db, dgamma, dbeta
