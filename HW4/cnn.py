import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
               hidden_dim=100, num_classes=10, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters_1: Number of filters to use in the first convolutional layer
        - num_filters_2: Number of filters to use in the second convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        (self.C, self.H, self.W) = input_dim
        self.filter_size = filter_size
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Your initializations should work for any valid input dims,      #
        # number of filters, hidden dims, and num_classes. Assume that we use      #
        # max pooling with pool height and width 2 with stride 2.                  #
        #                                                                          #
        # For Linear layers, weights and biases should be initialized from a       #
        # uniform distribution from -sqrt(k) to sqrt(k),                           #
        # where k = 1 / (#input features)                                          #
        # For Conv. layers, weights should be initialized from a uniform           #
        # distribution from -sqrt(k) to sqrt(k),                                   #
        # where k = 1 / ((#input channels) * filter_size^2)                        #
        # Note: we use the same initialization as pytorch.                         #
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html           #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html           #
        #                                                                          #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights for the convolutional layer using the keys 'W1' and 'W2'   #
        # (here we do not consider the bias term in the convolutional layer);      #
        # use keys 'W3' and 'b3' for the weights and biases of the                 #
        # hidden fully-connected layer, and keys 'W4' and 'b4' for the weights     #
        # and biases of the output affine layer.                                   #
        #                                                                          #
        # Make sure you have initialized W1, W2, W3, W4, b3, and b4 in the         #
        # params dicitionary.                                                      #
        #                                                                          #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3. Calculate the size of W3 dynamically           #
        ############################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        scale_W1 = np.sqrt(2. / (self.C * self.filter_size * self.filter_size))
        self.params['W1'] = np.random.randn(self.num_filters_1, self.C, self.filter_size, self.filter_size) * scale_W1

        H1 = 1 + (self.H - self.filter_size)
        W1 = 1 + (self.W - self.filter_size)
        H1 = (H1 - 2) // 2 + 1 
        W1 = (W1 - 2) // 2 + 1

        scale_W2 = np.sqrt(2. / (self.num_filters_1 * self.filter_size * self.filter_size))
        self.params['W2'] = np.random.randn(self.num_filters_2, self.num_filters_1, self.filter_size, filter_size) * scale_W2
        
        H2 = 1 + (H1 - self.filter_size)
        W2 = 1 + (W1 - self.filter_size)
        H2 = (H2 - 2) // 2 + 1
        W2 = (W2 - 2) // 2 + 1

        # Fully Connected Layer 1 - Initialize weights and biases
        F1_input_dim = self.num_filters_2 * H2 * W2
        scale_W3 = np.sqrt(2. / F1_input_dim)
        self.params['W3'] = np.random.randn(F1_input_dim, self.hidden_dim) * scale_W3
        self.params['b3'] = np.zeros(self.hidden_dim)
        
        # Fully Connected Layer 2 (Output Layer) - Initialize weights and biases
        scale_W4 = np.sqrt(2. / self.hidden_dim)
        self.params['W4'] = np.random.randn(self.hidden_dim, self.num_classes) * scale_W4
        self.params['b4'] = np.zeros(self.num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3.                                                #
        ############################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        # out_1, cache_1 = conv_forward(X, W1)
        # out_2, cache_2 = relu_forward(out_1)
        # out_3, cache_3 = conv_forward(out_2, W2)
        # out_4, cache_4 = relu_forward(out_3)
        # out_5, cache_5 = max_pool_forward(out_4, pool_param)
        # out_5_flattened = out_5.reshape((out_5.shape[0], -1))  # Flatten after max pooling for the next FC layer
        # out_6, cache_6 = fc_forward(out_5_flattened, W3, b3)
        # out_7, cache_7 = relu_forward(out_6)
        # scores, cache_8 = fc_forward(out_7, W4, b4)


        out_1, cache_1 = conv_forward(X, W1)
        out_2, cache_2 = relu_forward(out_1)
        out_3, cache_3 = max_pool_forward(out_2, pool_param)  # First max pooling
        out_4, cache_4 = conv_forward(out_3, W2)
        out_5, cache_5 = relu_forward(out_4)
        out_6, cache_6 = max_pool_forward(out_5, pool_param)  # Second max pooling
        out_6_flattened = out_6.reshape((out_6.shape[0], -1))  # Flatten for FC layer
        out_7, cache_7 = fc_forward(out_6_flattened, W3, b3)
        out_8, cache_8 = relu_forward(out_7)
        scores, cache_9 = fc_forward(out_8, W4, b4)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k].                                                      #
        # Hint: The backwards from W3 needs to be un-flattened before it can be    #
        # passed into the max pool backwards                                       #
        ############################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")

        loss, dout = softmax_loss(scores, y)

        # dx_8, grads['W4'], grads['b4'] = fc_backward(dout, cache_8)
        # dx_7 = relu_backward(dx_8, cache_7)
        # dx_6, grads['W3'], grads['b3'] = fc_backward(dx_7, cache_6)
        # dx_6_reshaped = dx_6.reshape(out_5.shape)
        # dx_5 = max_pool_backward(dx_6_reshaped, cache_5)
        # dx_4 = relu_backward(dx_5, cache_4)
        # dx_3, grads['W2'] = conv_backward(dx_4, cache_3)
        # dx_2 = relu_backward(dx_3, cache_2)
        # dx_1, grads['W1'] = conv_backward(dx_2, cache_1)

        dx_9, grads['W4'], grads['b4'] = fc_backward(dout, cache_9)
        dx_8 = relu_backward(dx_9, cache_8)
        dx_7, grads['W3'], grads['b3'] = fc_backward(dx_8, cache_7)
        dx_7_reshaped = dx_7.reshape(out_6.shape)  # Reshape for max pooling
        dx_6 = max_pool_backward(dx_7_reshaped, cache_6)  # Second max pooling backward
        dx_5 = relu_backward(dx_6, cache_5)
        dx_4, grads['W2'] = conv_backward(dx_5, cache_4)
        dx_3 = max_pool_backward(dx_4, cache_3)  # First max pooling backward
        dx_2 = relu_backward(dx_3, cache_2)
        dx_1, grads['W1'] = conv_backward(dx_2, cache_1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
