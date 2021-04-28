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
    def __init__(self):
        super(GAP, self).__init__()

    def forward(self, feature_map):
        '''
        params:
            feature_map: (B x C x F x T)
        returns:
            embedding: (B x C)
        '''
        pass

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
    def __init__(self):
        super(MultiHeadSAP, self).__init__()

    def forward(self, feature_map):
        '''
        params:
            feature_map: (B x C x T)
        returns:
            embeddings: (B x 2C)
        '''
        pass

class LDE(nn.Module):
    def __init__(self):
        super(LDE, self).__init__()

    def forward(self, feature_map):
        pass

class MultiHeadFFA(nn.Module):
    def __init__(self):
        super(MultiHeadFFA, self).__init__()

    def forward(self, feature_map):
        pass
