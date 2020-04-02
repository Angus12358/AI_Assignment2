from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros((hidden_dim,))
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros((num_classes,))

        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        # print(X.shape, W1.shape, b1.shape)
        # print(W2.shape, b2.shape)
        
        affine1_out, affine1_cache = affine_relu_forward(X, W1, b1)
        # print('len(affine1_cache):', len(affine1_cache))
        affine2_out, affine2_cache = affine_forward(affine1_out, W2, b2)
        scores = affine2_out

        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        loss_no_reg, dx_softmax = softmax_loss(scores, y)
        # print('loss_no_reg: ', loss_no_reg)

        # 因为除以batch_size会出错，所以这里的正则项不除以batch_size
        loss = loss_no_reg + self.reg * 0.5 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))  # / X.shape[0]
        # print('loss_reg: ', loss)

        dx_affine2, grads['W2'], grads['b2'] = affine_backward(dx_softmax, affine2_cache)
        grads['W2'] += self.reg * W2  # / X.shape[0]

        _, grads['W1'], grads['b1'] = affine_relu_backward(dx_affine2, affine1_cache)
        grads['W1'] += self.reg * W1  # / X.shape[0]

        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #shape1 = input_dim
        #for i, shape2 in enumerate(hidden_dims):
        #    self.params['W'+str(i+1)] = weight_scale * np.random.randn(shape1, shape2)
        #    self.params['b'+str(i+1)] = np.zeros(shape2)
        #    shape1 = shape2
        #    if self.normalization:
        #        self.params['gamma'+str(i+1)] = np.ones(shape2)
        #        self.params['beta'+str(i+1)] = np.zeros(shape2)
        #self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(shape1, num_classes)
        #self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        
        
        for i in range(self.num_layers):
            if i == 0:
                self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale,
                                                     size=(input_dim, hidden_dims[0]))
                self.params['b1'] = np.zeros((hidden_dims[0],))
                if self.normalization is not None:
                    self.params['gamma1'] = np.ones((hidden_dims[0],))
                    self.params['beta1'] = np.zeros((hidden_dims[0],))
                else:
                    self.params['gamma1'] = np.empty(1)
                    self.params['beta1'] = np.empty(1)
            elif i < self.num_layers - 1:
                self.params['W' + str(i + 1)] \
                    = np.random.normal(loc=0.0, scale=weight_scale,
                                       size=(hidden_dims[i - 1], hidden_dims[i]))
                self.params['b' + str(i + 1)] = np.zeros((hidden_dims[i],))
                if self.normalization is not None:
                    self.params['gamma' + str(i + 1)] = np.ones((hidden_dims[i],))
                    self.params['beta' + str(i + 1)] = np.zeros((hidden_dims[i],))
                else:
                    self.params['gamma' + str(i + 1)] = np.empty(1)
                    self.params['beta' + str(i + 1)] = np.empty(1)
            else:
                self.params['W' + str(self.num_layers)] \
                    = np.random.normal(loc=0.0, scale=weight_scale,
                                       size=(hidden_dims[-1], num_classes))
                self.params['b' + str(self.num_layers)] = np.zeros((num_classes,))

        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #ar_cache = {}
        #dp_cache = {}
        #layer_input = X
        #for i in range(1, self.num_layers):
        #    if self.normalization:
        #        layer_input, ar_cache[i-1] = affine_bn_relu_forward(
        #            layer_input, self.params['W'+str(i)], self.params['b'+str(i)],
        #            self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1])
        #    else:
        #        layer_input, ar_cache[i-1] = affine_relu_forward(
        #            layer_input, self.params['W'+str(i)], self.params['b'+str(i)])
        #    if self.use_dropout:
        #        layer_input, dp_cache[i-1] = dropout_forward(layer_input, self.dropout_param)
 
        #layer_out, ar_cache[self.num_layers] = affine_forward(
        #    layer_input,self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])
        #scores = layer_out
        
        
        hrs = {}
        # print('len(self.bn_params): ', len(self.bn_params))
        for i in range(self.num_layers - 1):
            if self.normalization is None:
                # print(i+1, "self.params['W' + str(i+1)], self.params['b' + str(i+1)]: ",
                #       self.params['W' + str(i+1)].shape, self.params['b' + str(i+1)].shape)
                hrs['out_rep_' + str(i+1)], hrs['cache_rep_' + str(i+1)] \
                    = affine_norm_relu_dropout_forward(X, self.params['W' + str(i+1)],
                                                       self.params['b' + str(i + 1)],
                                                       normalization=self.normalization,
                                                       gamma=self.params['gamma' + str(i+1)],
                                                       beta=self.params['beta' + str(i+1)],
                                                       bn_params=None,
                                                       dropout=self.use_dropout,
                                                       dropout_param=self.dropout_param)
            else:
                hrs['out_rep_' + str(i + 1)], hrs['cache_rep_' + str(i + 1)] \
                    = affine_norm_relu_dropout_forward(X, self.params['W' + str(i + 1)],
                                                       self.params['b' + str(i + 1)],
                                                       normalization=self.normalization,
                                                       gamma=self.params['gamma' + str(i + 1)],
                                                       beta=self.params['beta' + str(i + 1)],
                                                       bn_params=self.bn_params[i],
                                                       dropout=self.use_dropout,
                                                       dropout_param=self.dropout_param)

            X = hrs['out_rep_' + str(i+1)]

        affine_out, affine_cache = affine_forward(hrs['out_rep_' + str(self.num_layers - 1)],
                                                  self.params['W' + str(self.num_layers)],
                                                  self.params['b' + str(self.num_layers)])

        scores = affine_out

        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #loss, dscores = softmax_loss(scores, y)
        #loss += 0.5 * self.reg * np.sum(self.params['W' + str(self.num_layers)] * self.params['W' + str(self.num_layers)])
        #dout, dw, db = affine_backward(dscores, ar_cache[self.num_layers])
        #grads['W'+str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        #grads['b'+str(self.num_layers)] = db
        #for i in range(self.num_layers-1):
        #    layer = self.num_layers - i - 1
        #    loss += 0.5 * self.reg * np.sum(self.params['W' + str(layer)] * self.params['W' + str(layer)])
        #    if self.use_dropout:
        #        dout = dropout_backward(dout, dp_cache[layer-1])
        #    if self.normalization:
        #        dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, ar_cache[layer-1])
        #        grads['gamma' + str(layer)] = dgamma
        #        grads['beta' + str(layer)] = dbeta
        #    else:
        #        dout, dw, db = affine_relu_backward(dout, ar_cache[layer-1])
        #    grads['W' + str(layer)] = dw + self.reg * self.params['W' + str(layer)]
        #    grads['b' + str(layer)] = db
        
        
        loss, dx = softmax_loss(scores, y)

        for i in range(self.num_layers):
            loss += 0.5 * self.reg \
                    * np.sum(self.params['W' + str(i + 1)] ** 2)  # * 1 / X.shape[0]

        daffine_out, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] \
            = affine_backward(dx, affine_cache)
        grads['W' + str(self.num_layers)] \
            += self.reg * self.params['W' + str(self.num_layers)]   # * 1 / X.shape[0]

        rep_backward_in = daffine_out
        for i in range(self.num_layers - 1)[::-1]:
            # print(i + 1)
            drep_out, grads['W' + str(i + 1)], grads['b' + str(i + 1)], \
            grads['gamma' + str(i + 1)], grads['beta' + str(i + 1)] \
                = affine_norm_relu_dropout_backward(rep_backward_in, hrs['cache_rep_' + str(i+1)])
            grads['W' + str(i + 1)] \
                += self.reg * self.params['W' + str(i + 1)]  # * 1 / X.shape[0]

            rep_backward_in = drep_out

        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

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