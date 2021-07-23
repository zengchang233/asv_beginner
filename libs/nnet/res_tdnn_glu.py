import sys
sys.path.insert(0, "../../")

import torch
import torch.nn as nn
import torch.nn.functional as F

import libs.components.conv as conv
import libs.components.loss as loss
import libs.components.pooling as pooling

class XVector(nn.Module):
    def __init__(self, opts):
        super(XVector, self).__init__()
        context = opts['context']
        input_dim = opts['input_dim']
        hidden_dim = opts['hidden_dim']
        layers_num = opts['tdnn_layers']
        embedding_dim = opts['embedding_dim']
        attention_hidden_size = opts['attention_hidden_size']
        num_head = opts['num_head']
        self.activation = nn.ReLU()
        layers = []
        
        self.first_conv = nn.Conv1d(input_dim, 512, kernel_size = 1, padding = 0, stride = 1, dilation = 1)
        self.first_bn = nn.BatchNorm1d(512)
        self.first_activation = nn.ReLU()
        input_dim = 512

        for i in range(layers_num):
            layers.append(conv.ResidualTDNNGLU(input_dim, hidden_dim[i], context = context[i], stride = 1))
            input_dim = hidden_dim[i]

        self.frame_level = nn.Sequential(*layers)

        # pooling method selection
        if opts['pooling'] == 'STAT':
            self.pooling = pooling.STAT()
        elif opts['pooling'] == 'TAP':
            self.pooling = pooling.TAP()
        elif opts['pooling'] == 'ASP':
            self.pooling = pooling.AttentiveStatPooling(attention_hidden_size, hidden_dim[-1])
            #  self.pooling = pooling.AttentiveStatisticsPooling(hidden_dim[-1], hidden_size = attention_hidden_size)
        elif opts['pooling'] == 'multi_head_ffa':
            self.pooling = pooling.MultiHeadFFA(hidden_dim[-1], attention_hidden_size)
        elif opts['pooling'] == 'multi_head_attention':
            self.pooling = pooling.MultiHeadAttentionPooling(hidden_dim[-1], num_head = num_head)
        elif opts['pooling'] == 'multi_resolution_attention':
            self.pooling = pooling.MultiResolutionMultiHeadAttentionPooling(hidden_dim[-1], num_head = num_head)
        else:
            raise NotImplementedError('Other pooling method has not implemented.')

        # first fc layer
        if opts['pooling'] == 'STAT' or opts['pooling'] == 'ASP' \
                or opts['pooling'] == 'multi_head_attention' \
                or opts['pooling'] == 'multi_resolution_attention':
            self.fc1 = nn.Linear(hidden_dim[-1] * 2, embedding_dim)
        elif opts['pooling'] == 'TAP' or opts['pooling'] == 'multi_head_ffa':
            self.fc1 = nn.Linear(hidden_dim[-1], embedding_dim)
        else:
            raise ValueError("pooling method is wrong!")

        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        #  self.bn2 = nn.BatchNorm1d(embedding_dim)

    def extract_embedding(self, x):
        x = self.first_conv(x)
        x = self.first_activation(x)
        x = self.first_bn(x)
        x = self.frame_level(x)
        x = self.pooling(x)
        x.squeeze_(-1)
        x_a = self.fc1(x)
        x = self.activation(x)
        x = self.bn1(x_a)
        x_b = self.fc2(x)
        return x_b, x_a
        
    def forward(self, x):
        x, _ = self.extract_embedding(x)
        #  x = self.activation(x)
        #  x = self.bn2(x)
        return x

        
if __name__ == '__main__':
    import yaml
    from yaml import CLoader
    from torchsummary import summary
    f = open('../../conf/model/tdnn.yaml', 'r')
    opts = yaml.load(f, Loader = CLoader)
    f.close()
    net = XVector(opts)
    summary(net.cuda(), (30, 300))
