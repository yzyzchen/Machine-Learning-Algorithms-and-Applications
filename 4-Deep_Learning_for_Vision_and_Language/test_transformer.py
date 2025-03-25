import random
import pytest
import unittest
from unittest.mock import MagicMock, call, patch
from pdb import set_trace as st

import numpy as np

class DUMMY_MODULE:
    def __getattr__(self, name):
        raise NotImplementedError
    
try:
    from transformer import MaskedAttention as attention_student
except ModuleNotFoundError:
    linear_regression_student = DUMMY_MODULE()

from test_utils import TensorTestCase, TestProgramWithCode

import torch
import torch.nn as nn
from torch.nn import functional as F

class SolMaskedAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.attention = nn.Linear(self.n_embd, 3*self.n_embd) #key, query, and value
        self.fc = nn.Linear(self.n_embd, self.n_embd)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def split(self, values):
        B, T, C = values.shape
        return values.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    
    def fill_mask(self, attention):
        T = attention.shape[-1]
        attention = attention.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        return attention
    
    def forward(self, x):
        Q, K, V  = self.attention(x).split(self.n_embd, dim=2)
        Q, K, V = self.split(Q), self.split(K), self.split(V)
        attention = (Q @ K.transpose(-2, -1) / np.sqrt(K.size(-1)))
        attention = self.fill_mask(attention)
        attention = F.softmax(attention, dim=-1)
        y = self.drop1(attention) @ V
        y = y.transpose(1, 2).contiguous().view(x.size())
        y = self.drop2(self.fc(y))
        return y
    
class TransformerTest(TensorTestCase):

    def test_attention_small(self):
        self.reset_seed(True)
        target_fn = attention_student(48, 3, 3)
        target_fn.eval()
        self.reset_seed(True)
        gt_fn = SolMaskedAttention(48, 3, 3)
        gt_fn.eval()

        self.reset_seed(True)
        tol = 1e-3
        num_trials = 5
        rng = np.random.default_rng(42)
        for trial in range(num_trials):
            x = rng.standard_normal((2, 3, 48))
            x = torch.tensor(x).float()
            x1 = torch.clone(x)
            with torch.no_grad():
                self.reset_seed(True)
                gt_out = gt_fn(x)
                self.reset_seed(True)
                your_out = target_fn(x1)

            self.assertTensorClose(gt_out, your_out, tol, 'your output is far from expected result')

    def test_attention_large(self):
        self.reset_seed(True)
        target_fn = attention_student(256, 16, 20)
        target_fn.eval()

        self.reset_seed(True)
        gt_fn = SolMaskedAttention(256, 16, 20)
        gt_fn.eval()

        tol = 1e-3
        num_trials = 5
        self.reset_seed(True)
        rng = np.random.default_rng(42)
        for trial in range(num_trials):
            x = rng.standard_normal((2, 20, 256))
            x = torch.tensor(x).float()
            x1 = torch.clone(x)
            with torch.no_grad():
                self.reset_seed(True)
                gt_out = gt_fn(x)
                self.reset_seed(True)
                your_out = target_fn(x1)

            self.assertTensorClose(gt_out, your_out, tol, 'your output is far from expected result')

if __name__ == '__main__':
    #unittest.main()
    TestProgramWithCode()
