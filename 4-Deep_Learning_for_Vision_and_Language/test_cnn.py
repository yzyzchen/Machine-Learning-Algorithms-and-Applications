import random
import pytest
try:
    import pytest_timeout  # !pip install pytest-timeout
except ImportError:
    pass
import unittest
from unittest.mock import MagicMock, call, patch
import random
import math
import numpy as np
import scipy.signal# create hooks to disallow calls to convolve2d and related
import cnn
from layers import fc_backward, fc_forward, relu_backward, relu_forward, softmax_loss
from test_utils import TensorTestCase, TestProgramWithCode
from pdb import set_trace as st
class DUMMY_MODULE:
    def __getattr__(self, name):
        raise NotImplementedError

try:
    import cnn_layers as cnn_layers_student
    import cnn as cnn_student
except ModuleNotFoundError:
    cnn_layers_student = DUMMY_MODULE()
    cnn_student = DUMMY_MODULE()

def conv_forward(x, w):
    out = None
    # Extract shapes and constants
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1
    # Construct output
    out = np.zeros((N, F, H_prime, W_prime))

    for j in range(0, H_prime):
        for i in range(0, W_prime):
           tmp_w = w
           tmp_w = tmp_w[np.newaxis,:]
           tmp_w = np.repeat(tmp_w, N, axis=0)
           tmp_x = x[:, :, j:j + HH, i:i + WW] 
           tmp_x = tmp_x[:,np.newaxis]
           tmp_x = np.repeat(tmp_x, F, axis=1)
           out[:,:,j,i] = np.sum(np.sum(np.sum(tmp_x*tmp_w, axis=-1), axis=-1), axis=-1) 

    cache = (x, w)
    return out, cache

def conv_backward(dout, cache):
    dx, dw = None, None
    x, w = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1
    # Construct output
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    for j in range(0, H_prime):
        for i in range(0, W_prime):
            # tmp_dout : N x F
            tmp_dout = dout[:, :, j, i]
            # tmp_dout : N x F x 1 x 1 x 1
            tmp_dout = tmp_dout[:,:,np.newaxis,np.newaxis,np.newaxis]
            # tmp_x : N x C x HH x WW
            tmp_x = x[:, :, j:j + HH, i:i + WW]
            # tmp_x : N x 1 x C x HH x WW
            tmp_x = tmp_x[:,np.newaxis]
            # tmp_x * tmp_dout : N x F x C x HH x WW
            dw += np.sum(tmp_x * tmp_dout, axis=0)
            # w * tmp_dout : N x F x C x HH x WW
            dx[:,:,j:j + HH, i:i + WW] += np.sum(w*tmp_dout, axis=1)
    return dx, dw

def max_pool_forward(x, pool_param):
    out = None
    (N, C, H, W) = x.shape

    p_H = pool_param.get('pool_height', 3)
    p_W = pool_param.get('pool_width', 3)
    stride = pool_param.get('stride', 1)
    H_out = 1 + (H - p_H) / stride
    W_out = 1 + (W - p_W) / stride

    S = (N, C, math.floor(H_out), math.floor(W_out))
    out = np.zeros(S)
    
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

def fail_with_message(*args):
    raise ValueError('Call to scipy.signal not allowed!')

class CNNTest(TensorTestCase):
    class SolConvNet:
        def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
                   hidden_dim=100, num_classes=10, dtype=np.float32):
            self.params = {}
            self.dtype = dtype
            (self.C, self.H, self.W) = input_dim
            self.filter_size = filter_size
            self.num_filters_1 = num_filters_1
            self.num_filters_2 = num_filters_2
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes

            k = 1 / (self.C * self.filter_size**2)
            self.params['W1'] = np.random.uniform(-np.sqrt(k), np.sqrt(k), (num_filters_1, self.C, filter_size, filter_size))
            H_2 = (1+self.H-self.filter_size) // 2
            W_2 = (1+self.W-self.filter_size) // 2
            k2 = 1 / (num_filters_2 * self.filter_size**2)
            self.params['W2'] = np.random.uniform(-np.sqrt(k2), np.sqrt(k2), (num_filters_2, num_filters_1, filter_size, filter_size))
            H_3 = (1+H_2-self.filter_size) // 2
            W_3 = (1+W_2-self.filter_size) // 2
            k3 = 1 / (num_filters_2 * H_3 * W_3)
            self.params['W3'] = np.random.uniform(-np.sqrt(k3), np.sqrt(k3), (num_filters_2 * H_3 * W_3, hidden_dim))
            self.params['b3'] = np.random.uniform(-np.sqrt(k3), np.sqrt(k3), (hidden_dim,))
            self.params['W4'] = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, num_classes))
            self.params['b4'] = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (num_classes,))

            for k, v in self.params.items():
                self.params[k] = v.astype(dtype)

        def loss(self, X, y=None):
            W1 = self.params['W1']
            W2 = self.params['W2']
            W3, b3 = self.params['W3'], self.params['b3']
            W4, b4 = self.params['W4'], self.params['b4']

            # pass pool_param to the forward pass for the max-pooling layer
            pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

            scores = None

            out_1, cache_1 = conv_forward(X, W1)
            out_2, cache_2 = relu_forward(out_1)
            out_3, cache_3 = max_pool_forward(out_2, pool_param)
            out_4, cache_4 = conv_forward(out_3, W2)
            out_5, cache_5 = relu_forward(out_4)
            out_6, cache_6 = max_pool_forward(out_5, pool_param)
            out_6 = out_6.reshape((out_6.shape[0], -1))
            out_7, cache_7 = fc_forward(out_6, W3, b3)
            out_8, cache_8 = relu_forward(out_7)
            out_9, cache_9 = fc_forward(out_8, W4, b4)
            scores = out_9

            if y is None:
                return scores

            loss, grads = 0, {}

            loss, dx = softmax_loss(scores, y)
            dx_9, grads['W4'], grads['b4'] = fc_backward(dx, cache_9)
            dx_8 = relu_backward(dx_9, cache_8)
            dx_7, grads['W3'], grads['b3'] = fc_backward(dx_8, cache_7)
            H_ = (1+self.H-self.filter_size) // 2
            W_ = (1+self.W-self.filter_size) // 2
            H__ = (1+H_-self.filter_size) // 2
            W__ = (1+W_-self.filter_size) // 2
            dx_7 = dx_7.reshape((dx_7.shape[0], self.num_filters_2, H__, W__))
            dx_6 = max_pool_backward(dx_7, cache_6)
            dx_5 = relu_backward(dx_6, cache_5)
            dx_4, grads['W2'] = conv_backward(dx_5, cache_4)
            dx_3 = max_pool_backward(dx_4, cache_3)
            dx_2 = relu_backward(dx_3, cache_2)
            dx_1, grads['W1'] = conv_backward(dx_2, cache_1)

            return loss, grads

    def _sol_conv_forward(self, x, w):
        out = None

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride = 1
        pad = 0
        # Check for parameter sanity
        assert (H + 2 * pad - HH) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
        assert (W + 2 * pad - WW) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'
        H_prime = 1 + (H + 2 * pad - HH) // stride
        W_prime = 1 + (W + 2 * pad - WW) // stride
        # Padding
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
        # Construct output
        out = np.zeros((N, F, H_prime, W_prime))
        for j in range(0, H_prime):
            for i in range(0, W_prime):
               tmp_w = w
               tmp_w = tmp_w[np.newaxis,:]
               tmp_w = np.repeat(tmp_w, N, axis=0)
               tmp_x = x_pad[:, :, j * stride:j * stride + HH, i * stride:i * stride + WW]
               tmp_x = tmp_x[:,np.newaxis]
               tmp_x = np.repeat(tmp_x, F, axis=1)
               out[:,:,j,i] = np.sum(np.sum(np.sum(tmp_x*tmp_w, axis=-1), axis=-1), axis=-1)
        cache = (x, w)
        return out, cache

    def _sol_conv_backward(self, dout, cache):
        dx, dw = None, None
        x, w = cache
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride = 1
        pad = 0
        # Padding
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
        H_prime = 1 + (H + 2 * pad - HH) // stride
        W_prime = 1 + (W + 2 * pad - WW) // stride
        # Construct output
        dx_pad = np.zeros_like(x_pad)
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)

        for j in range(0, H_prime):
            for i in range(0, W_prime):
                tmp_dout = dout[:, :, j, i]
                tmp_dout = tmp_dout[:,:,np.newaxis]
                tmp_dout = np.repeat(tmp_dout, C, axis=2)
                tmp_dout = tmp_dout[:,:,:,np.newaxis]
                tmp_dout = np.repeat(tmp_dout, HH, axis=3)
                tmp_dout = tmp_dout[:,:,:,:, np.newaxis]
                tmp_dout = np.repeat(tmp_dout, WW, axis=4)
                tmp_x = x_pad[:, :, j * stride:j * stride + HH, i * stride:i * stride + WW]
                tmp_x = tmp_x[:,np.newaxis]
                tmp_x = np.repeat(tmp_x, F, axis=1)
                dw += np.sum(tmp_x * tmp_dout, axis=0)
                tmp_w = w
                tmp_w = tmp_w[:,np.newaxis]
                tmp_w = np.repeat(tmp_w, N, axis=0)
                dx_pad[:,:,j * stride:j * stride + HH, i * stride:i * stride + WW] += np.sum(w*tmp_dout, axis=1)
        dx = dx_pad[:, :, pad:pad + H, pad:pad + W]

        return dx, dw

    def _sol_max_pool_forward(self, x, pool_param):
        out = None
        (N, C, H, W) = x.shape

        p_H = pool_param.get('pool_height', 3)
        p_W = pool_param.get('pool_width', 3)
        stride = pool_param.get('stride', 1)
        H_out = 1 + (H - p_H) / stride
        W_out = 1 + (W - p_W) / stride

        S = (N, C, math.floor(H_out), math.floor(W_out))
        out = np.zeros(S)

        for x1 in range(math.floor(H_out)):
            for y in range(math.floor(W_out)):
                out[:, :, x1, y] = np.amax(np.amax(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W], axis=-1), axis=-1)
        cache = (x, pool_param)
        return out, cache

    def _sol_max_pool_backward(self, dout, cache):
        dx = None

        (x, pool_param) = cache
        (N, C, H, W) = x.shape
        p_H = pool_param.get('pool_height', 3)
        p_W = pool_param.get('pool_width', 3)
        stride = pool_param.get('stride', 1)
        H_out = 1 + (H - p_H) / stride
        W_out = 1 + (W - p_W) / stride

        dx = np.zeros(x.shape)
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

    @patch('scipy.signal.convolve', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.correlate', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.oaconvolve', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.convolve2d', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.correlate2d', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.choose_conv_method', side_effect=fail_with_message, autospec=True)
    def test_conv_forward(self, *args):
        target_fn = cnn_layers_student.conv_forward
        gt_fn = conv_forward

        self.reset_seed()
        tol = 1e-6
        num_trials = 10
        rng = np.random.default_rng(42)
        for trial in range(num_trials):
            X = rng.standard_normal((7, 3, 19, 19))
            K = rng.standard_normal((5, 3, 4, 4))

            gt_out, gt_cache = gt_fn(X, K)

            with self.assertUnchanged(X, K):
                your_out, your_cache = target_fn(X, K)

            self.assertTensorClose(gt_out, your_out, tol, 'your output is far from expected result')
            self.assertTrue(len(gt_cache) == len(your_cache), 'your number of returns in cache is differ from the spec')
            for gt_item, your_item in zip(gt_cache, your_cache):
                self.assertTensorClose(gt_item, your_item, tol, 'your cache value is far from expected result')

    @patch('scipy.signal.convolve', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.correlate', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.oaconvolve', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.convolve2d', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.correlate2d', side_effect=fail_with_message, autospec=True)
    @patch('scipy.signal.choose_conv_method', side_effect=fail_with_message, autospec=True)
    def test_conv_backward(self, *args):
        target_fn = cnn_layers_student.conv_backward
        gt_fn = conv_backward

        self.reset_seed()
        tol = 1e-6
        num_trials = 10
        rng = np.random.default_rng(42)
        for trial in range(num_trials):
            X = rng.standard_normal((7, 3, 19, 19))
            K = rng.standard_normal((5, 3, 4, 4))

            gt_dout, gt_test_cache = conv_forward(X, K)
            gt_dX, gt_dK = gt_fn(gt_dout, gt_test_cache)

            with self.assertUnchanged(gt_dout):
                your_dX, your_dK = target_fn(gt_dout, gt_test_cache)

            self.assertTensorClose(gt_dX, your_dX, tol, 'your output is far from expected result')
            self.assertTensorClose(gt_dK, your_dK, tol, 'your output is far from expected result')

    def test_cnn_initialization(self):
        self.reset_seed()
        tol = 1e-6

        rng = np.random.default_rng(42)
        N, C, H, W, K = 5, 3, 29, 29, 5
        num_filters_1, num_filters_2, filter_size, hidden_dim = 4, 5, 3, 19
        X = rng.standard_normal((N, C, H, W))
        y = rng.integers(K, size=N)

        your_cnn = cnn_student.ConvNet(input_dim=(C, H, W), num_filters_1=num_filters_1, num_filters_2=num_filters_2, filter_size=filter_size, hidden_dim=hidden_dim, num_classes=K)
        gt_cnn = self.SolConvNet(input_dim=(C, H, W), num_filters_1=num_filters_1, num_filters_2=num_filters_2, filter_size=filter_size, hidden_dim=hidden_dim, num_classes=K)
        self.assertTrue(len(your_cnn.params) == len(gt_cnn.params), 'wrong number of parameters')
        for p in gt_cnn.params:
            self.assertTrue(gt_cnn.params[p].shape == your_cnn.params[p].shape, 'parameter shape mismatch')

    @patch('cnn.conv_forward', side_effect=conv_forward, autospec=True)
    @patch('cnn.conv_backward', side_effect=conv_backward, autospec=True)
    def test_cnn_loss_scores(self, *patches):
        self.reset_seed()
        tol = 1e-6

        rng = np.random.default_rng(42)
        N, C, H, W, K = 5, 3, 29, 29, 5
        num_filters_1, num_filters_2, filter_size, hidden_dim = 4, 5, 3, 19

        num_trials = 10
        for trial in range(num_trials):
            your_cnn = cnn_student.ConvNet(input_dim=(C, H, W), num_filters_1=num_filters_1, num_filters_2=num_filters_2, filter_size=filter_size, hidden_dim=hidden_dim, num_classes=K)
            gt_cnn = self.SolConvNet(input_dim=(C, H, W), num_filters_1=num_filters_1, num_filters_2=num_filters_2, filter_size=filter_size, hidden_dim=hidden_dim, num_classes=K)

            # copy over gt model params
            for param, val in gt_cnn.params.items():
                your_cnn.params[param] = gt_cnn.params[param]

            X = rng.standard_normal((N, C, H, W))

            gt_scores = gt_cnn.loss(X)
            with self.assertUnchanged(X):
                your_scores = your_cnn.loss(X)

            self.assertTensorClose(gt_scores, your_scores, tol, 'your output is far from expected result')

    @patch('cnn.conv_forward', side_effect=conv_forward, autospec=True)
    @patch('cnn.conv_backward', side_effect=conv_backward, autospec=True)
    def test_cnn_loss_rest(self, *patches):
        self.reset_seed()
        tol = 1e-6

        rng = np.random.default_rng(42)
        N, C, H, W, K = 5, 3, 29, 29, 5
        num_filters_1, num_filters_2, filter_size, hidden_dim = 4, 5, 3, 19

        num_trials = 10
        for trial in range(num_trials):
            your_cnn = cnn_student.ConvNet(input_dim=(C, H, W), num_filters_1=num_filters_1, num_filters_2=num_filters_2, filter_size=filter_size, hidden_dim=hidden_dim, num_classes=K)
            gt_cnn = self.SolConvNet(input_dim=(C, H, W), num_filters_1=num_filters_1, num_filters_2=num_filters_2, filter_size=filter_size, hidden_dim=hidden_dim, num_classes=K)

            # copy over gt model params
            for param, val in gt_cnn.params.items():
                your_cnn.params[param] = gt_cnn.params[param]

            X = rng.standard_normal((N, C, H, W))
            y = rng.integers(K, size=N)

            gt_loss, gt_grads = gt_cnn.loss(X, y)
            with self.assertUnchanged(X, y):
                your_loss, your_grads = your_cnn.loss(X, y)

            self.assertTensorClose(gt_loss, your_loss, tol, 'your output is far from expected result')
            self.assertTrue(len(gt_grads) == len(your_grads), 'your number of returns in cache is differ from the spec')
            for gt_key, gt_item in gt_grads.items():
                your_item = your_grads[gt_key]
                self.assertTensorClose(gt_item, your_item, tol, 'your cache value is far from expected result')

if __name__ == '__main__':
    #unittest.main()
    TestProgramWithCode()
