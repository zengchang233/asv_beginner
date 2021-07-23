from __future__ import with_statement, print_function, absolute_import

import math
import numpy as np

import torch
# import conv
from torch import nn
from torch.nn import functional as F


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    m = conv.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


def ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs):
    freq_axis_kernel_size = kernel_size[0]
    m = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
    m.weight.data.fill_(1.0 / freq_axis_kernel_size)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Conv1d1x1(in_channels, out_channels, bias=True):
    """1-by-1 convolution layer
    """
    return Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                  dilation=1, bias=bias)


def _conv1x1_forward(conv, x, is_incremental):
    """Conv1x1 forward
    """
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit
    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 skip_out_channels=None, dropout=1 - 0.95, 
                 padding=None, dilation=1, causal=True, bias=True):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = nn.Conv1d(residual_channels, gate_channels, kernel_size = kernel_size,
                              padding = padding, dilation = dilation,
                              bias = bias)

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = nn.Conv1d(gate_out_channels, residual_channels, kernel_size = 1, bias = bias)
        self.conv1x1_skip = nn.Conv1d(gate_out_channels, skip_out_channels, kernel_size = 1, bias = bias)

    def forward(self, x, c=None, g=None):
        return self._forward(x)

    def _forward(self, x):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not
        Returns:
            Tensor: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)

        splitdim = 1
        x = self.conv(x)
        # remove future time steps
        x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        x = torch.tanh(a) * torch.sigmoid(b)

        # For skip connection
        s = self.conv1x1_skip(x)

        # For residual connection
        x = self.conv1x1_out(x)

        x = (x + residual) * math.sqrt(0.5)
        return x, s