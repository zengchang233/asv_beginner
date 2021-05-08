import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(CrossEntropy, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, embeddings, labels):
        logits = self.fc(embeddings)
        loss = F.cross_entropy(logits + eps, labels)
        return loss, logits

class ASoftmax(nn.Module):
    def __init__(self):
        super(ASoftmax, self).__init__()

    def forward(self, embeddings, labels):
        pass

class AMSoftmax(nn.Module):
    def __init__(self, embedding_size, num_classes, s, margin):
        super(AMSoftmax, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.margin = margin
        self.weights = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.weights)

    def forward(self, embeddings, labels):
        logits = F.linear(F.normalize(embeddings), F.normalize(self.weights))
        margin = torch.zeros_like(logits)
        margin.scatter_(1, labels.view(-1,1), self.margin)
        m_logits = self.s * (logits - margin)
        loss = F.cross_entropy(m_logits, labels)
        return loss, logits

class LMCL(AMSoftmax):
    def __init__(self, embedding_size, num_classes, s, margin):
        super(LMCL, self).__init__(embedding_size, num_classes, s, margin)

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
