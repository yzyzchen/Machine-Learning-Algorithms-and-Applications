from builtins import range
import numpy as np
import math


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn_layers.py!")


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: (x, w)

    """
    out = None

    # Extract shapes and constants
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1

    # Construct output
    out = np.zeros((N, F, H_prime, W_prime))

    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: loop through H_prime and W_prime, but vectorized the rest         #
    # for j in range(0, H_prime):                                             #
    #     for i in range(0, W_prime):                                         #
    #         ...                                                             #
    #         out[:,:,j,i] = ...                                              #
    # You may optionally want to implement a naive loop version for sanity    #
    # checking and then implement a vectorized version later.                 #
    # Hint: You can also check the max_pool_forward implementation for        #
    # implementation and vectorization tricks.                                #
    # Iterating through image height and width is not efficient for           #
    # large images, you can try to further remove the nested for loops for    #
    # H_prime and W_prime. However, for this problem the dataset is small     #
    # enough so it is not necessary to do so.                                 #
    ###########################################################################
    # ~~START DELETE~~
    # Naive Loops
    #for n in range(N):
    #    for f in range(F):
    #        for j in range(0, H_prime):
    #            for i in range(0, W_prime):
    #                out[n, f, j, i] = (x[n, :, j:j + HH, i:i + WW] * w[f, :, :, :]).sum()
    #for f in range(F):
    #    for j in range(0, H_prime):
    #        for i in range(0, W_prime):
    #            tmp_w = w[f, :, :, :]
    #            tmp_w = tmp_w[np.newaxis,:]
    #            tmp_w = np.repeat(tmp_w, N, axis=0)
    #            out[:, f, j, i] = np.sum(np.sum(np.sum(x[:, :, j:j + HH, i:i + WW] * tmp_w, axis=-3), axis=-2), axis=-1)
    #for j in range(0, H_prime):
    #    for i in range(0, W_prime):
    #        tmp_w = w
    #        tmp_w = tmp_w[np.newaxis,:]
    #        tmp_w = np.repeat(tmp_w, N, axis=0)
    #        tmp_x = x[:, :, j:j + HH, i:i + WW]
    #        tmp_x = tmp_x[:,np.newaxis]
    #        tmp_x = np.repeat(tmp_x, F, axis=1)
    #        out[:,:,j,i] = np.sum(np.sum(np.sum(tmp_x*tmp_w, axis=-1), axis=-1), axis=-1)

    # better solution with broadcasting
    for j in range(0, H_prime):
        for i in range(0, W_prime):
            # w : F x C x HH x WW
            # tmp_x : N x C x HH x WW
            tmp_x = x[:, :, j:j + HH, i:i + WW]
            # tmp_x : N x 1 x C x HH x WW
            tmp_x = tmp_x[:,np.newaxis]
            # tmp_x * w = N x F x C x HH x WW
            # out = N x F x H' x W'
            out[:,:,j,i] = (tmp_x * w).sum((2,3,4))

    # A nested for loop through image height and width can be super inefficient, espeically for large image datasets.
    # Therefore, you can further optimize the for loops via vectorization.
    # The CIFAR dataset is small enough, therefore you don't need to worry about it for now. 
    # We are just providing it for reference. 
    # indices_h = [range(j, j+HH) for j in range(H_prime)]
    # indices_w = [range(i, i+WW) for i in range(W_prime)]
    # tmp_xs = x[:,:,indices_h][..., indices_w]
    # tmp_xs = tmp_xs.transpose(2,4,0,1,3,5)
    # tmp_xs = np.concatenate(tmp_xs, axis=0)[:,:,np.newaxis]
    # j_s = [j//W_prime for j in range(H_prime*W_prime)]
    # i_s = [i for i in range(W_prime)] * H_prime
    # out[:,:,j_s, i_s]=(tmp_xs * w).sum((3,4,5)).transpose(1,2,0)
    # ~~END DELETE~~
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - dout: Upstream derivatives of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: A tuple of (x, w) as in conv_forward
      where x: Input data of shape (N, C, H, W)
      where w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    x, w = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1
    # Construct output
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    # w_pad : F x C x (2H - HH) x (2W - WW)
    pad_len = ((H_prime-1), (H_prime-1))
    w_pad = np.pad(w, ((0, 0), (0, 0), (pad_len[0], pad_len[0]), (pad_len[1], pad_len[1])), 'constant', constant_values=0)
    assert w_pad.shape[-2:] == (2*H - HH, 2*W - WW)

    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    # Hint: For vectorization, you can loop through HH and WW for dL/dw.      #
    # for j in range(0, HH):                                                  #
    #     for i in range(0, WW):                                              #
    #         ...                                                             #
    # Hint: For vectorization, you can loop through H and W for dL/dx.        #
    # for j in range(0, H):                                                   #
    #     for i in range(0, W):                                               #
    #         ...                                                             #
    # You may optionally want to implement a naive loop version for sanity    #
    # checking and then implement a vectorized version later.                 #
    #                                                                         #
    # Hint: You can also check the max_pool_backward implementation for       #
    # implementation and vectorization tricks.                                #
    # Hint: The full convolution is equal to padding the 'image' then         #
    # performing a valid convolution. The padded w is provided to you.        #
    # Hint: You can flip a numpy array by slicing.                            #
    # E.g. x[:,:,::-1,::-1] will flip the H and W dimensions of x.            #
    ###########################################################################
    # ~~START DELETE~~
    # Naive Loops
    #for n in range(N):
    #    for f in range(F):
    #        for j in range(0, H_prime):
    #            for i in range(0, W_prime):
    #                dw[f] += x_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * dout[n, f, j, i]
    #                dx_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] += w[f] * dout[n, f, j, i]
    # Extract dx from dx_pad
    #dx = dx_pad[:, :, pad:pad + H, pad:pad + W]

    #for f in range(F):
    #    for j in range(0, H_prime):
    #        for i in range(0, W_prime):
    #            tmp_dout = dout[:, f, j, i]
    #            tmp_dout = tmp_dout[:,np.newaxis]
    #            tmp_dout = np.repeat(tmp_dout, C, axis=1)
    #            tmp_dout = tmp_dout[:,:,np.newaxis]
    #            tmp_dout = np.repeat(tmp_dout, HH, axis=2)
    #            tmp_dout = tmp_dout[:,:, :, np.newaxis]
    #            tmp_dout = np.repeat(tmp_dout, WW, axis=3)
    #            dw[f] += np.sum(x_pad[:, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * tmp_dout, axis=0)
    #            dx_pad[:,:,j * stride:j * stride + HH, i * stride:i * stride + WW] += w[f]*tmp_dout

    # vectorized direct full/filt conv solution given in the problem documentation
    for i in range(0, HH):
        for j in range(0, WW):
            # tmp_x : N x 1 x C x H' x W'
            tmp_x = x[:, np.newaxis, :, i:i + H_prime, j:j + W_prime]
            # tmp_dout : N x F x 1 x H' x W'
            tmp_dout = dout[:,:,np.newaxis]
            # tmp_x * tmp_dout : N x F x C x H' x W'
            dw[:,:,i,j] = (tmp_x * tmp_dout).sum(axis=(0,3,4))
    for i in range(0, H):
        for j in range(0, W):
            # tmp_dout : N x F x 1 x H' x W'
            tmp_dout = dout[:,:,np.newaxis,::-1,::-1]
            # tmp_w : F x C x H' x W'
            tmp_w = w_pad[:, :, i:i + H_prime, j:j + W_prime]
            # tmp_w * tmp_dout : N x F x C x H' x W'
            dx[:,:,i,j] = (tmp_w * tmp_dout).sum((1,3,4))

    # direct full/filt conv solution by looping through H_prime and W_prime
    # dout : N x F x H' x W'
    # w : F x C x HH x WW
    #for j in range(0, H_prime):
    #    for i in range(0, W_prime):
    #        # tmp_dout : N x F
    #        tmp_dout = dout[:, :, j, i]
    #        # tmp_dout : N x F x 1 x 1 x 1
    #        tmp_dout = tmp_dout[:,:,np.newaxis,np.newaxis,np.newaxis]
    #        # tmp_x : N x C x HH x WW
    #        tmp_x = x[:, :, j:j + HH, i:i + WW]
    #        # tmp_x : N x 1 x C x HH x WW
    #        tmp_x = tmp_x[:,np.newaxis]
    #        # tmp_x * tmp_dout : N x F x C x HH x WW
    #        dw += np.sum(tmp_x * tmp_dout, axis=0)

    #        # tmp_dout_2 : N x F # (flip dout for valid conv)
    #        tmp_dout_2 = dout[:, :, H_prime - j - 1, W_prime - i - 1]
    #        # tmp_dout : N x F x 1 x 1 x 1
    #        tmp_dout_2 = tmp_dout_2[:,:,np.newaxis,np.newaxis,np.newaxis]
    #        # w_pad : F x C x .. x ..
    #        tmp_w = w_pad[:, :, j:j + H, i:i + W]
    #        # tmp_w : N x C x H x W
    #        # tmp_w * tmp_dout : N x F x C x H x W
    #        dx += np.sum(tmp_w*tmp_dout_2, axis=1)

    ## most optimized solution - trick to remove padding
    ## dout : N x F x H' x W'
    ## w : F x C x HH x WW
    #for j in range(0, H_prime):
    #    for i in range(0, W_prime):
    #        # tmp_dout : N x F
    #        tmp_dout = dout[:, :, j, i]
    #        # tmp_dout : N x F x 1 x 1 x 1
    #        tmp_dout = tmp_dout[:,:,np.newaxis,np.newaxis,np.newaxis]
    #        # tmp_x : N x C x HH x WW
    #        tmp_x = x[:, :, j:j + HH, i:i + WW]
    #        # tmp_x : N x 1 x C x HH x WW
    #        tmp_x = tmp_x[:,np.newaxis]
    #        # tmp_x * tmp_dout : N x F x C x HH x WW
    #        dw += np.sum(tmp_x * tmp_dout, axis=0)
    #        # w * tmp_dout : N x F x C x HH x WW
    #        dx[:,:,j:j + HH, i:i + WW] += np.sum(w*tmp_dout, axis=1)

    # ~~END DELETE~~
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    (N, C, H, W) = x.shape

    p_H = pool_param.get('pool_height', 3)
    p_W = pool_param.get('pool_width', 3)
    stride = pool_param.get('stride', 1)
    H_out = 1 + (H - p_H) / stride
    W_out = 1 + (W - p_W) / stride

    S = (N, C, math.floor(H_out), math.floor(W_out))
    out = np.zeros(S)
    #for n in range(N):
    #    for c in range(C):
    #        for x1 in range(math.floor(H_out)):
    #            for y in range(math.floor(W_out)):
    #                out[n, c, x1, y] = np.amax(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W])

    for x1 in range(math.floor(H_out)):
        for y in range(math.floor(W_out)):
            out[:, :, x1, y] = np.amax(np.amax(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W], axis=-1), axis=-1)
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    (x, pool_param) = cache
    (N, C, H, W) = x.shape
    p_H = pool_param.get('pool_height', 3)
    p_W = pool_param.get('pool_width', 3)
    stride = pool_param.get('stride', 1)
    H_out = 1 + (H - p_H) / stride
    W_out = 1 + (W - p_W) / stride

    dx = np.zeros(x.shape)
    #for n in range(N):
    #    for c in range(C):
    #        for x1 in range(math.floor(H_out)):
    #            for y in range(math.floor(W_out)):
    #                max_element = np.amax(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W])
    #                temp = np.zeros(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W].shape)
    #                temp = (x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] == max_element)
    #                dx[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] += dout[n, c, x1, y] * temp
    for x1 in range(math.floor(H_out)):
        for y in range(math.floor(W_out)):
            max_element = np.amax(np.amax(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W], axis=-1), axis=-1)
            max_element = max_element[:,:,np.newaxis]
            max_element = np.repeat(max_element, p_H, axis=2)
            max_element = max_element[:,:,:,np.newaxis]
            max_element = np.repeat(max_element, p_W, axis=3)
            temp = np.zeros(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W].shape)
            temp = (x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] == max_element)
            tmp_dout = dout[:,:,x1,y]
            tmp_dout = tmp_dout[:,:,np.newaxis]
            tmp_dout = np.repeat(tmp_dout, p_H, axis=2)
            tmp_dout = tmp_dout[:,:,:,np.newaxis]
            tmp_dout = np.repeat(tmp_dout, p_W, axis=3)
            dx[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] += tmp_dout * temp
    return dx
