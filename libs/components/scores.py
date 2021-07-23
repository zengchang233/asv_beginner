import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineScore(nn.Module):
    def __init__(self):
        super(CosineScore, self).__init__()
    
    def forward(self, enroll, eval):
        '''
        inputs:
            enroll: (batch_size, embedding_dim)
            eval: (batch_size, embedding_dim)
        returns:
            cosine_matrix: (batch_size, batch_size)
        '''
        cosine_matrix = F.linear(F.normalize(eval), F.normalize(enroll))
        return cosine_matrix
    
class PLDALikeScore(nn.Module):
    def __init__(self, input_dim):
        super(PLDALikeScore, self).__init__()
        self.a = nn.Parameter(torch.randn(input_dim, input_dim))
        self.b = nn.Parameter(torch.randn(1, 1))
        # self._initialize()
        
    def _initialize(self):
        nn.init.kaiming_normal_(self.a)
        nn.init.kaiming_normal_(self.b)
    
    def forward(self, enroll, eval):
        enroll = F.normalize(enroll)
        eval = F.normalize(eval)
        diag = torch.eye(enroll.size(1)).to(enroll.device)
        # S = torch.diag(self.a.view(-1)) + diag
        S = self.a.matmul(self.a.T) + diag
        part1 = F.linear(eval, enroll)
        part2 = torch.chain_matmul(enroll, S, enroll.T)
        part3 = torch.chain_matmul(eval, S, eval.T)
        plda_matrix = part1 - part2 - part3 + self.b
        return plda_matrix
    
class MhaScore(nn.Module):
    def __init__(self):
        super(MhaScore, self).__init__()
        
    def forward(self, enroll, eval):
        pass
