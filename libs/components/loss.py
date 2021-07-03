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
        loss = F.cross_entropy(logits, labels)
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
        super().forward(embeddings, labels)

class AAMSoftmax(nn.Module):
    def __init__(self):
        super(AAMSoftmax, self).__init__()

    def forward(self, embeddings, labels):
        pass

class OnlineTripletLoss(nn.Module):
    def __init__(self, centers, margin, selector = 'hardest', cpu=False):
        super(OnlineTripletLoss, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.centers = F.normalize(centers)
        self.centers.requires_grad = False
        self.selector = selector

    def forward(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        cos_matrix = F.linear(embeddings, self.centers)# cos_matrix batch_size * 1211
        rows = torch.arange(embeddings.size(0))
        positive_cos = cos_matrix[rows, labels].view(-1,1) # 32 * 1
        idx = torch.ones((embeddings.size(0), self.centers.size(0)), dtype = rows.dtype) # 32 * 1211
        idx[rows, labels] = 0
        negative_cos_matrix = cos_matrix[idx > 0].view(embeddings.size(0), -1) # 32 * 1210
        loss_values = negative_cos_matrix + self.margin - positive_cos # 求出所有的loss 32 * 1210
        if self.selector == 'hardest': # 挑选出最大的loss
            loss_value, _ = torch.max(loss_values, dim = 1)
        if self.selector == 'hard':
            pass
        if self.selector == 'semihard':
            pass
        losses = F.relu(loss_value.view(-1,1))
        return losses.mean(), (loss_value > 0).sum().item()

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
