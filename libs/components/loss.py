import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, embeddings, labels):
        pass

class ASoftmax(nn.Module):
    def __init__(self):
        super(ASoftmax, self).__init__()

    def forward(self, embeddings, labels):
        pass

class AMSoftmax(nn.Module):
    def __init__(self):
        super(AMSoftmax, self).__init__()

    def forward(self, embeddings, labels):
        pass

class LMCL(AMSoftmax):
    def __init__(self):
        super(LMCL, self).__init__()

    def forward(self, embeddings, labels):
        super.forward(embeddings, labels)

class AAMSoftmax(nn.Module):
    def __init__(self):
        super(AAMSoftmax, self).__init__()

    def forward(self, embeddings, labels):
        pass

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, embeddings, labels):
        pass

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, embeddings, labels):
        pass
