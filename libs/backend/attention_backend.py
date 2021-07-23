import sys
sys.path.insert(0, '../../')

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.components.attention import AttentionAlphaComponent
from libs.components.pooling import MultiHeadAttentionPooling
from libs.components import scores

def mean(x, dim = 0):
    return torch.mean(x, dim = dim)

class AttentionAggregation(nn.Module):
    def __init__(self, opts):
        super(AttentionAggregation, self).__init__()
        self.head = opts['head']
        self.d_model = opts['d_model']
        self.pooling = opts['pooling']
        self.score_method = opts['scores']
        '''
        torch自带的mha的输入是(q,k,v)，格式分别如下：
            q: (length, batch_size, dimension)
            k: (length, batch_size, dimension)
            v: (length, batch_size, dimension)
        输出的是attention_mapping和attention_weight:
            attention_mapping: (length, batch_size, dimension)
            attention_weight: 格式暂时不知
        '''
        self.attention_transformation = nn.MultiheadAttention(self.d_model, self.head)
        if self.pooling == 'mean':
            self.aggregation = mean
        elif self.pooling == 'mha':
            '''
            AttentionAlphaComponent的输入是(x)，格式如下：
                x: (batch_size, dimension, length)
            输出是attention_weight:
                attention_weight: (batch_size, head, length)
            '''
            self.aggregation = AttentionAlphaComponent(self.d_model, self.head)
            #  self.aggregation = MultiHeadAttentionPooling(self.d_model, num_head = self.head, stddev = False)
        
        #  self.fc = nn.Linear(self.d_model * 2, self.d_model)

        if self.score_method == 'cosine':
           self.score = scores.CosineScore()
        elif self.score_method == 'plda':
           self.score = scores.PLDALikeScore(self.d_model)
        elif self.score_method == 'mha':
           self.score = scores.MhaScore()
        else:
           raise NotImplementedError('Not available!')

    def merge(self, x):
        '''
        params:
            x: (N, M, D), N speakers, M utterances per speaker, D dimension
        '''
        assert len(x.size()) == 3
        n_spks, n_utts, _ = x.size()
        x = x.repeat(1, n_utts, 1) # 重复n_utts次
        mask = torch.logical_not(torch.eye(n_utts)).repeat(n_spks, 1).view(-1).to(x.device) # mask掉本身
        masked_x = x.view(-1, self.d_model)[mask].contiguous().view(n_spks * n_utts, -1, self.d_model) # 每n_spks个划分为一组，分别对应mask掉的部分
        masked_x = masked_x.transpose(0, 1).contiguous() # n_utts - 1, n_spks * n_utts, dimension
        x, _ = self.attention_transformation(masked_x, masked_x, masked_x, None) # n_utts - 1, n_spks * n_utts, dimension
        x = x + masked_x
        if self.pooling == 'mean':
            aggregation = mean(x) # (n_spks * (n_utts - 1), dimension), n_spks * (n_utts - 1)
        else:
            x = x.permute(1, 2, 0) # n_spks * n_utts, dimension, n_utts - 1
            alpha = self.aggregation(x)
            aggregation = x.view(n_spks * n_utts, self.head, self.d_model // self.head, -1).matmul(alpha.unsqueeze(-1)).view(n_spks * n_utts, self.d_model)
            #  aggregation = self.aggregation(x).squeeze(-1)

        #  aggregation = self.fc(aggregation)

        return aggregation

    def forward(self, x):
        '''
        params:
            x: (N, M, D), N speakers, M utterances per speaker, D dimension
        '''
        n_spks, n_utts, _ = x.size()
        aggregation = self.merge(x).view(n_spks * n_utts, -1) # 得到n_spks * n_utts个中心, x_n^e
        x = x.view(-1, self.d_model) # x_n^t

        score_matrix = self.score(aggregation, x) # n_utts * n_spks, n_utts * n_spks

        mask = torch.eye(n_utts, dtype = torch.bool).repeat(n_spks, n_spks).to(x.device) # mask
        ground_truth_matrix = torch.eye(mask.size(0), dtype = torch.long).to(x.device) # 候选单位阵
        scores = score_matrix.view(-1)[mask.view(-1)].view(-1, 1) # n_utts * n_spks * n_spks
        ground_truth = ground_truth_matrix.view(-1)[mask.view(-1)] # .view(-1, 1)

        return scores, ground_truth

    def enroll(self, x):
        x = x.transpose(0, 1).contiguous() # n_utts, 1, dimension
        output, _ = self.attention_transformation(x, x, x, None)
        x += output # n_utts, 1, dimension
        if self.pooling == 'mean':
            aggregation = mean(x) # (n_spks * (n_utts - 1), dimension), n_spks * (n_utts - 1)
        else:
            x = x.permute(1, 2, 0) # 1, dimension, n_utts 
            alpha = self.aggregation(x)
            aggregation = x.view(x.size(0), self.head, self.d_model // self.head, -1).matmul(alpha.unsqueeze(-1)).view(x.size(0), self.d_model)
            #  aggregation = self.aggregation(x).squeeze(-1)

        #  aggregation = self.fc(aggregation)

        return aggregation
        
    def test(self, aggregation, evaluation, criterion):
        score = self.score(aggregation, evaluation)
        calibration_score, _ = criterion.get_prob(score.view(-1, 1))
        return calibration_score 

if __name__ == '__main__':
    import torch
    opts = {}
    opts['utterance'] = {}
    opts['utterance']['d_model'] = 256
    opts['utterance']['dropout'] = 0.1
    opts['utterance']['head'] = 8
    opts['utterance']['pooling'] = 'mha'

    model = AttentionAggregation(opts['utterance'])
    x = torch.randn(128, 5, 256)
    output = model(x)
    print(output[0])
