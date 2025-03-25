import random
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
from layers import fc_backward, fc_forward
from rnn import CaptioningRNN
import rnn_layers
from rnn_layers import (rnn_backward, rnn_forward, rnn_step_forward,
                        temporal_softmax_loss, word_embedding_backward,
                        word_embedding_forward)
from test_utils import TensorTestCase, TestProgramWithCode

def temporal_fc_forward(x, w, b):
    out, cache = None, None
    out = x @ w + b
    cache = (x, w, b, out)
    return out, cache

def temporal_fc_backward(dout, cache):
    dx, dw, db = None, None, None
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db

class RNNTest(TensorTestCase):

    class SolCaptioningRNN:
        def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                     hidden_dim=128, cell_type='rnn', dtype=np.float32):
            if cell_type not in {'rnn'}:
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

            # Initialize word vectors
            self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
            self.params['W_embed'] /= 100

            # Initialize CNN -> hidden state projection parameters
            self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
            self.params['W_proj'] /= np.sqrt(input_dim)
            self.params['b_proj'] = np.zeros(hidden_dim)

            # Initialize parameters for the RNN
            self.params['Wx'] = np.random.randn(wordvec_dim, hidden_dim)
            self.params['Wx'] /= np.sqrt(wordvec_dim)
            self.params['Wh'] = np.random.randn(hidden_dim, hidden_dim)
            self.params['Wh'] /= np.sqrt(hidden_dim)
            self.params['b'] = np.zeros(hidden_dim)

            # Initialize output to vocab weights
            self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
            self.params['W_vocab'] /= np.sqrt(hidden_dim)
            self.params['b_vocab'] = np.zeros(vocab_size)

            # Cast parameters to correct dtype
            for k, v in self.params.items():
                self.params[k] = v.astype(self.dtype)


        def loss(self, features, captions):
            captions_in = captions[:, :-1]
            captions_out = captions[:, 1:]

            mask = (captions_out != self._null)

            W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

            W_embed = self.params['W_embed']

            Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

            W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

            loss, grads = 0.0, {}

            # Forward Pass
            # (1)
            h0, cache_affine = fc_forward(x = features, w= W_proj, b=b_proj)
            # (2)
            embedded_captions_in, cache_embed_in = word_embedding_forward(captions_in, W_embed)
            embedded_captions_in = np.transpose(embedded_captions_in, (1,0,2))
            # (3)
            h, cache_rnn = rnn_forward(embedded_captions_in, h0, Wx, Wh, b)
            h = np.transpose(h, (1,0,2))
            # (4)
            y, cache_temporal = temporal_fc_forward(h, W_vocab, b_vocab)
            # (5)
            loss, dout = temporal_softmax_loss(y, captions_out, mask)
            # Gradients
            # (4)
            dout, grads['W_vocab'], grads['b_vocab'] = temporal_fc_backward(dout, cache_temporal)
            dout = np.transpose(dout, (1,0,2))
            # (3)
            dout, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dout, cache_rnn)
            dout = np.transpose(dout, (1,0,2))
            # (2)
            grads['W_embed'] = word_embedding_backward(dout, cache_embed_in)
            # (1)
            _, grads['W_proj'], grads['b_proj'] = fc_backward(dh0, cache_affine)

            return loss, grads

        def sample(self, features, max_length=30):
            N = features.shape[0]
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Unpack parameters
            W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
            W_embed = self.params['W_embed']
            Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
            W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

            h0 = features.dot(W_proj) + b_proj
            start = (self._start * np.ones(N)).astype(np.int32)
            x = W_embed[start, :]
            h = h0
            for t in range(max_length):
                h, _ = rnn_step_forward(x, h, Wx, Wh, b)
                y = h.dot(W_vocab) + b_vocab
                captions[:, t] = np.argmax(y, axis=1)
                x = W_embed[captions[:, t], :]

            return captions

    def test_temporal_fc_forward(self):
        target_fn = rnn_layers.temporal_fc_forward
        gt_fn = temporal_fc_forward

        self.reset_seed()
        tol = 1e-8
        num_trials = 10
        rng = np.random.default_rng(42)
        for trial in range(num_trials):
            x = rng.standard_normal((7, 6, 7))
            w = rng.standard_normal((7, 5))
            b = rng.standard_normal((5,))

            gt_out, gt_cache = gt_fn(x, w, b)

            with self.assertUnchanged(x, w, b):
                your_out, your_cache = target_fn(x, w, b)

            self.assertTensorClose(gt_out, your_out, tol, 'your output is far from expected result')
            self.assertTrue(len(gt_cache) == len(your_cache), 'your number of returns in cache is differ from the spec')
            for gt_item, your_item in zip(gt_cache, your_cache):
                self.assertTensorClose(gt_item, your_item, tol, 'your cache value is far from expected result')

    def test_temporal_fc_backward(self):
        target_fn = rnn_layers.temporal_fc_backward
        gt_fn = temporal_fc_backward

        self.reset_seed()
        tol = 1e-6
        num_trials = 10
        rng = np.random.default_rng(42)
        for trial in range(num_trials):
            x = rng.standard_normal((7, 6, 7))
            w = rng.standard_normal((7, 5))
            b = rng.standard_normal((5,))

            gt_dout, gt_test_cache = temporal_fc_forward(x, w, b)
            gt_dx, gt_dw, gt_db = gt_fn(gt_dout, gt_test_cache)

            with self.assertUnchanged(gt_dout):
                your_dx, your_dw, your_db = target_fn(gt_dout, gt_test_cache)

            self.assertTensorClose(gt_dx, your_dx, tol, 'your output is far from expected result')
            self.assertTensorClose(gt_dw, your_dw, tol, 'your output is far from expected result')
            self.assertTensorClose(gt_db, your_db, tol, 'your output is far from expected result')

    @patch('rnn.temporal_fc_forward', side_effect=temporal_fc_forward, autospec=True)
    @patch('rnn.temporal_fc_backward', side_effect=temporal_fc_backward, autospec=True)
    def test_rnn_loss(self, *patches):
        self.reset_seed()
        tol = 1e-6

        rng = np.random.default_rng(545)
        num_trials = 10
        for trial in range(num_trials):
            wtoi = {f'{i}': i for i in range(10)}# dummy word to index
            wtoi['<NULL>'] = 10
            your_rnn_model = CaptioningRNN(
                cell_type='rnn',
                word_to_idx=wtoi,
                input_dim=5,
                hidden_dim=4,
                wordvec_dim=5,
            )
            gt_rnn_model = self.SolCaptioningRNN(
                cell_type='rnn',
                word_to_idx=wtoi,
                input_dim=5,
                hidden_dim=4,
                wordvec_dim=5,
            )

            X = rng.standard_normal((2, 5))
            y = rng.integers(0, 10, size=(2, 4))

            # copy over gt model params
            for param, val in gt_rnn_model.params.items():
                your_rnn_model.params[param] = gt_rnn_model.params[param]

            gt_loss, gt_grads = gt_rnn_model.loss(X, y)

            with self.assertUnchanged(X, y):
                your_loss, your_grads = your_rnn_model.loss(X, y)

            self.assertTensorClose(gt_loss, your_loss, tol, 'your output is far from expected result')
            self.assertTrue(len(gt_grads) == len(your_grads), 'your number of returns in cache is differ from the spec')

            for gt_key, gt_item in gt_grads.items():
                your_item = your_grads[gt_key]
                self.assertTensorClose(gt_item, your_item, tol, 'your cache value is far from expected result')

    @patch('rnn.temporal_fc_forward', side_effect=temporal_fc_forward, autospec=True)
    @patch('rnn.temporal_fc_backward', side_effect=temporal_fc_backward, autospec=True)
    def test_rnn_sample(self, *patches):
        self.reset_seed()
        tol = 1e-6

        rng = np.random.default_rng(42)
        num_trials = 10
        for trial in range(num_trials):
            wtoi = {f'{i}': i for i in range(10)}# dummy word to index
            wtoi['<NULL>'] = 10
            wtoi['<START>'] = 11
            wtoi['<END>'] = 12
            your_rnn_model = CaptioningRNN(
                cell_type='rnn',
                word_to_idx=wtoi,
                input_dim=5,
                hidden_dim=4,
                wordvec_dim=5,
            )
            gt_rnn_model = self.SolCaptioningRNN(
                cell_type='rnn',
                word_to_idx=wtoi,
                input_dim=5,
                hidden_dim=4,
                wordvec_dim=5,
            )
            X = rng.standard_normal((2, 5))
            #y = rng.integers(0, 10, size=(2, 4))

            # copy over gt model params
            for param, val in gt_rnn_model.params.items():
                your_rnn_model.params[param] = gt_rnn_model.params[param]

            gt_caption = gt_rnn_model.sample(X, max_length=15)
            with self.assertUnchanged(X):
                your_caption = your_rnn_model.sample(X, max_length=15)

            assert np.issubdtype(your_caption.dtype, np.integer)
            your_caption = your_caption.astype(gt_caption.dtype)# convert to same int type
            self.assertTensorEqual(gt_caption, your_caption, 'your caption is not equal to the expected result')

if __name__ == '__main__':
    #unittest.main()
    TestProgramWithCode()
