from __future__ import with_statement, print_function, absolute_import

import math
import sys
sys.path.insert(0, '../../')
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from libs.nnet.module import Embedding
from libs.nnet.module import Conv1d1x1, ResidualConv1dGLU, ConvTranspose2d
from libs.components import pooling

def receptive_field_size(total_layers, num_cycles, kernel_size,
                         dilation=lambda x: 2**x):
    """Compute receptive field size
    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.
    Returns:
        int: receptive field size in sample
    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


class WaveNet(nn.Module):
    """The WaveNet model that supports local and global conditioning.
    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        n_speakers (int): Number of speakers. Used only if global conditioning
          is enabled.
        upsample_conditional_features (bool): Whether upsampling local
          conditioning features by transposed convolution layers or not.
        upsample_scales (list): List of upsample scale.
          ``np.prod(upsample_scales)`` must equal to hop size. Used only if
          upsample_conditional_features is enabled.
        freq_axis_kernel_size (int): Freq-axis kernel_size for transposed
          convolution layers for upsampling. If you only care about time-axis
          upsampling, set this to 1.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
        use_speaker_embedding (Bool): Use speaker embedding or Not. Set to False
          if you want to disable embedding layer and use external features
          directly.
    """

    def __init__(self, opts):
        super(WaveNet, self).__init__()
        out_channels = opts['input_dim'] # 256
        layers = opts['layers'] # 20
        stacks = opts['stacks'] # 2
        residual_channels = opts['residual_channels'] # 512
        gate_channels = opts['gate_channels'] # 512
        skip_out_channels = opts['skip_out_channels'] # 512
        kernel_size = opts['kernel_size'] # 3
        dropout = opts['dropout'] # 1 - 0.95
        scalar_input = opts['scalar_input'] # False
        attention_hidden_size = opts['attention_hidden_size']
        num_head = opts['num_head']
        embedding_dim = opts['embedding_dim']
        self.out_channels = out_channels
        assert layers % stacks == 0
        layers_per_stack = layers // stacks # dilation times
        if scalar_input:
            self.first_conv = nn.Conv1d(1, residual_channels, kernel_size = 1,
                                        stride = 1, padding = 0, bias = True)
        else:
            self.first_conv = nn.Conv1d(out_channels, residual_channels, kernel_size = 1,
                                        stride = 1, padding = 0, bias = True)

        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualConv1dGLU(
                residual_channels, gate_channels,
                kernel_size=kernel_size,
                skip_out_channels=skip_out_channels,
                bias=True,  # magenda uses bias, but musyoku doesn't
                dilation=dilation, dropout=dropout
                )
            self.conv_layers.append(conv)
        self.receptive_field = receptive_field_size(layers, stacks, kernel_size)
        
        # pooling method selection
        if opts['pooling'] == 'STAT':
            self.pooling = pooling.STAT()
        elif opts['pooling'] == 'TAP':
            self.pooling = pooling.TAP()
        elif opts['pooling'] == 'ASP':
            self.pooling = pooling.AttentiveStatPooling(attention_hidden_size, skip_out_channels)
            #  self.pooling = pooling.AttentiveStatisticsPooling(hidden_dim[-1], hidden_size = attention_hidden_size)
        elif opts['pooling'] == 'multi_head_ffa':
            self.pooling = pooling.MultiHeadFFA(skip_out_channels, attention_hidden_size)
        elif opts['pooling'] == 'multi_head_attention':
            self.pooling = pooling.MultiHeadAttentionPooling(skip_out_channels, num_head = num_head)
        elif opts['pooling'] == 'multi_resolution_attention':
            self.pooling = pooling.MultiResolutionMultiHeadAttentionPooling(skip_out_channels, num_head = num_head)
        else:
            raise NotImplementedError('Other pooling method has not implemented.')
        
        if opts['pooling'] == 'STAT' \
          or opts['pooling'] == 'ASP' \
          or opts['pooling'] == 'multi_head_attention' \
          or opts['pooling'] == 'multi_resolution_attention':
            self.fc1 = nn.Linear(skip_out_channels * 2, skip_out_channels)
        elif opts['pooling'] == 'TAP' or opts['pooling'] == 'multi_head_ffa':
            self.fc1 = nn.Linear(skip_out_channels, skip_out_channels)
        else:
            raise ValueError("pooling method is wrong!")

        self.bn1 = nn.BatchNorm1d(skip_out_channels)
        self.fc2 = nn.Linear(skip_out_channels, embedding_dim)
        
        self.activation = nn.ReLU()

    def extract_embedding(self, x):
        # Feed data to network
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        # return x, x
        x = self.pooling(x)
        x.squeeze_(-1)
        x_a = self.fc1(x)
        x = self.activation(x)
        x = self.bn1(x_a)
        x_b = self.fc2(x)
        return x_b, x_a

    def forward(self, x):
        """Forward step
        Args:
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
            g (Tensor): Global conditioning features,
              shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
              Note that ``self.use_speaker_embedding`` must be False when you
              want to disable embedding layer and use external features
              directly (e.g., one-hot vector).
              Also type of input tensor must be FloatTensor, not LongTensor
              in case of ``self.use_speaker_embedding`` equals False.
            softmax (bool): Whether applies softmax or not.
        Returns:
            Tensor: output, shape B x out_channels x T
        """
        x, _ = self.extract_embedding(x)
        return x