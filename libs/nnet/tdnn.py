import torch
import torch.nn as nn
import torch.nn.functional as F

import asv_beginner.libs.components.conv as conv
import asv_beginner.libs.components.loss as loss
import asv_beginner.libs.components.pooling as pooling

class XVector(nn.Module):
    def __init__(self, opts):
        super(XVector, self).__init__()
        opts = opts[opts['arch']]
        context = opts['context']
        input_dim = opts['input_dim']
        hidden_dim = opts['hidden_dim']
        layers_num = opts['tdnn_layers']
        embedding_dim = opts['embedding_dim']
        attention_hidden_size = opts['attention_hidden_size']
        self.bn_first = opts['bn_first']
        self.activation = nn.ReLU(negative_slope = 0.2)
        layers = []

        for i in range(layers_num):
            layers.append(conv.TDNNLayer(input_dim, hidden_dim[i], dilation = context[i], stride = 1, bn_first = self.bn_first))
            input_dim = hidden_dim[i]

        self.frame_level = nn.Sequential(*layers)

        # pooling method selection
        if opts['pooling'] == 'STAT':
            self.pooling = pooling.STAT()
        elif opts['pooling'] == 'TAP':
            self.pooling = pooling.TAP()
        elif opts['pooling'] == 'multi_head_sap':
            self.pooling = pooling.MultiHeadSAP(hidden_dim[-1], attention_hidden_size)
        elif opts['pooling'] == 'multi_head_ffa':
            self.pooling = pooling.MultiHeadFFA(hidden_dim[-1], attention_hidden_size)
        else:
            raise NotImplementedError('Other pooling method has not implemented.')

        # first fc layer
        if opts['pooling'] == 'STAT' or opts['pooling'] == 'multi_head_sap':
            self.fc1 = nn.Linear(hidden_dim[-1] * 2, embedding_dim)
        elif opts['pooling'] == 'TAP' or opts['pooling'] == 'multi_head_ffa':
            self.fc1 = nn.Linear(hidden_dim[-1], embedding_dim)
        else:
            raise ValueError("pooling method is wrong!")

        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

    def extract_embedding(self, x):
        x = self.tdnn(x)
        x = self.pooling(x)
        x.squeeze_(1)
        x_a = self.fc1(x)
        if self.bn_first:
            x = self.bn1(x_a)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.bn1(x_a)
        x_b = self.fc2(x)
        return x_b, x_a
        
    def forward(self, x):
        x, _ = self.extract_embedding(x)
        if self.bn_first:
            x = self.bn2(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.bn2(x)
        return x
