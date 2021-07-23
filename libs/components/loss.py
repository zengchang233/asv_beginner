import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self, input_dim, num_classes = 1, affine = True, reduction = 'mean'):
        super(BCELoss, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.affine = affine
        if self.affine:
            self.logistic = nn.Linear(input_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(reduction = reduction)

    def get_prob(self, inputs):
        if self.affine:
            logits = self.logistic(inputs)
            prob = self.sigmoid(logits)
            return logits, prob
        else:
            prob = self.sigmoid(inputs)
            return inputs, prob

    def forward(self, inputs, labels):
        labels = labels.view(-1, 1).float()
        _, prob = self.get_prob(inputs)
        loss = self.bce_loss(prob, labels)
        return loss, prob

class CrossEntropy(nn.Module):
    def __init__(self, embedding_size, num_classes, reduction = 'mean'):
        super(CrossEntropy, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.reduction = reduction
        self.fc = nn.Linear(embedding_size, num_classes)

    def get_prob(self, inputs):
        logits = self.fc(inputs)
        scores = F.softmax(logits, dim = 1)
        return scores[:, 1], logits

    def forward(self, embeddings, labels):
        logits = self.fc(embeddings)
        loss = F.cross_entropy(logits, labels, reduction = self.reduction)
        return loss, logits

class ASoftmax(nn.Module):
    def __init__(self, embedding_size, num_classes, margin):
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

#  class OnlineTripletLoss(nn.Module):
    #  def __init__(self, centers, margin, selector = 'hardest'):
    #      super(OnlineTripletLoss, self).__init__()
    #      self.margin = margin
    #      self.centers = F.normalize(centers)
    #      self.centers.requires_grad = False
    #      self.selector = selector
    #
    #  def forward(self, embeddings, labels):
    #      embeddings = embeddings.cpu()
    #      cos_matrix = F.linear(embeddings, self.centers)# cos_matrix batch_size * 1211
    #      rows = torch.arange(embeddings.size(0))
    #      positive_cos = cos_matrix[rows, labels].view(-1,1) # 32 * 1
    #      idx = torch.ones((embeddings.size(0), self.centers.size(0)), dtype = rows.dtype) # 32 * 1211
    #      idx[rows, labels] = 0
    #      negative_cos_matrix = cos_matrix[idx > 0].view(embeddings.size(0), -1) # 32 * 1210
    #      loss_values = negative_cos_matrix + self.margin - positive_cos # 求出所有的loss 32 * 1210
    #      if self.selector == 'hardest': # 挑选出最大的loss
    #          loss_value, _ = torch.max(loss_values, dim = 1)
    #      if self.selector == 'hard':
    #          pass
    #      if self.selector == 'semihard':
    #          pass
    #      losses = F.relu(loss_value.view(-1,1))
#          return losses.mean(), (loss_value > 0).sum().item()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if embeddings.is_cuda:
            triplets = triplets.cuda()
        ap_cos = F.cosine_similarity(embeddings[triplets[:,0]], embeddings[triplets[:,1]])  # .pow(.5)
        an_cos = F.cosine_similarity(embeddings[triplets[:,0]], embeddings[triplets[:,2]]) # .pow(.5)
        losses = F.relu(an_cos - ap_cos + self.margin)
        return losses.mean(), len(triplets)

class OnlineContrastiveLoss(nn.Module):
    def __init__(self, margin, pairs_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pairs_selector = pairs_selector

    def forward(self, embeddings, target):
        pass

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, embeddings, labels):
        pass

class OCAngleLayer(nn.Module):
    """ Output layer to produce activation for one-class softmax

    Usage example:
     batchsize = 64
     input_dim = 10
     class_num = 2

     l_layer = OCAngleLayer(input_dim)
     l_loss = OCSoftmaxWithLoss()

     data = torch.rand(batchsize, input_dim, requires_grad=True)
     target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
     target = target.to(torch.long)

     scores = l_layer(data)
     loss = l_loss(scores, target)

     loss.backward()
    """
    def __init__(self, in_planes, w_posi=0.9, w_nega=0.2, alpha=20.0):
        super(OCAngleLayer, self).__init__()
        self.in_planes = in_planes
        self.w_posi = w_posi # m_0
        self.w_nega = w_nega # m_1
        self.out_planes = 1
        
        self.weight = nn.Parameter(torch.Tensor(in_planes, self.out_planes))
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2,1,1e-5).mul_(1e5)

        self.alpha = alpha

    def forward(self, input, flag_angle_only=False):
        """
        Compute oc-softmax activations
        
        input:
        ------
          input tensor (batchsize, input_dim)

        output:
        -------
          tuple of tensor ((batchsize, output_dim), (batchsize, output_dim))
        """
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        # w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, 1)
        inner_wx = input.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)
                
        if flag_angle_only:
            pos_score = cos_theta
            neg_score = cos_theta
        else:
            pos_score = self.alpha * (self.w_posi - cos_theta)
            neg_score = -1 * self.alpha * (self.w_nega - cos_theta)
        
        #
        return pos_score, neg_score

    
class OCSoftmaxWithLoss(nn.Module):
    """
    OCSoftmaxWithLoss()
    
    """
    def __init__(self, reduction):
        super(OCSoftmaxWithLoss, self).__init__()
        self.m_loss = nn.Softplus()
        self.reduction = reduction

    def forward(self, inputs, target):
        """ 
        input:
        ------
          input: tuple of tensors ((batchsie, out_dim), (batchsie, out_dim))
                 output from OCAngle
                 inputs[0]: positive class score
                 inputs[1]: negative class score
          target: tensor (batchsize)
                 tensor of target index
        output:
        ------
          loss: scalar
        """
        # Assume target is binary, positive = 1, negaitve = 0
        # 
        # Equivalent to select the scores using if-elese
        # if target = 1, use inputs[0]
        # else, use inputs[1]
        output = inputs[0] * target.view(-1, 1) + \
                 inputs[1] * (1-target.view(-1, 1))
        loss = self.m_loss(output)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
   
class OCSoftmax(nn.Module):
    def __init__(self, input_dim, w_posi=0.9, w_nega=0.2, alpha=20.0, reduction = 'mean'):
        super(OCSoftmax, self).__init__()
        self.oc_layer = OCAngleLayer(input_dim, w_posi = w_posi, w_nega = w_nega, alpha = alpha)
        self.loss = OCSoftmaxWithLoss(reduction)

    def get_prob(self, inputs):
        logits = self.fc(inputs)
        scores = F.softmax(logits, dim = 1)
        return scores[:, 1], logits   
    
    def forward(self, x, target):
        pos_score, neg_score = self.oc_layer(x)
        loss = self.loss((pos_score, neg_score), target)
        return loss, pos_score

if __name__ == '__main__':
    ocsoftmax = OCSoftmax(64)
    inputs = torch.randn(4, 64)
    target = torch.tensor([0,0,1,0], dtype = torch.int64)
    output = ocsoftmax(inputs, target)
    print(output)
