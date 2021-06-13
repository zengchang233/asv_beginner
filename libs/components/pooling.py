import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TAP(nn.Module):
    def __init__(self):
        super(TAP, self).__init__()

    def forward(self, feature_map):
        '''
        params:
            feature_map: (B x C x T)
        returns:
            embedding: (B x C)
        '''
        mean = torch.mean(feature_map, dim = 2)
        return mean

class GAP(nn.Module):
    def __init__(self, output_size):
        super(GAP, self).__init__()
        assert (type(output_size) == int or len(output_size) >= 1), 'output_size must be int or list (tuple) type'
        self.global_average_pooling = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, feature_map):
        '''
        params:
            feature_map: (B x C x F x T)
        returns:
            embedding: (B x C)
        '''
        feature_map = self.global_average_pooling(feature_map)
        return feature_map

class STAT(nn.Module):
    """
    Mean and Standard deviation pooling
    """
    def __init__(self):
        super(STAT, self).__init__()
        pass

    def forward(self, feature_map):
        '''
        params:
            feature_map: (B x C x T)
        returns:
            embedding: (B x C)
        '''
        mean = torch.mean(feature_map, dim=2)
        std = torch.std(feature_map, dim=2)
        return torch.cat([mean, std], dim=1)

class MultiHeadSAP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiHeadSAP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, feature_map):
        '''
        params:
            feature_map: (B x C x T)
        returns:
            embeddings: (B x 2C)
        '''
        pass

class LDE(nn.Module):
    def __init__(self, input_dim):
        super(LDE, self).__init__()

    def forward(self, feature_map):
        pass

class MultiHeadFFA(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiHeadFFA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim


    def forward(self, feature_map):
        '''
        params:
            feature_map: (B x C x T)
        returns:
            embeddings: (B x C)
        '''
        pass
