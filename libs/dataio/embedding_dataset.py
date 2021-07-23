import os
import random
import sys
sys.path.insert(0, '../../')

from torch.utils.data import Dataset
import numpy as np
import torch

from libs.utils import utils

class EmbeddingTrainDataset(Dataset):
    def __init__(self, opts):
        path = opts['train_path']
        self.dataset = []
        self.count = 0
        self.labels = []
        spk_idx = 0
        for speaker in os.listdir(path):
            speaker_path = os.path.join(path, speaker)
            embeddings = os.listdir(speaker_path)
            if len(embeddings) > 10:
                for embedding in embeddings:
                    self.dataset.append(os.path.join(speaker_path, embedding))
                    self.labels.append(spk_idx)
                spk_idx += 1

    def __len__(self):
        return len(self.dataset) * 5
    
    def __getitem__(self, idx):
        embedding_path = self.dataset[idx]
        sid = self.labels[idx]
        embedding = np.load(embedding_path)
        return embedding, sid

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    opt = {}
    opt['path'] = "/home/smg/zengchang/data/cnceleb_xv/train"
    trainset = EmbeddingTrainDataset(opt)
    labels = trainset.labels
    speech_number = len(trainset)
    n_spks = 5
    n_speech = 4
    balance_sampler = utils.BalancedBatchSampler(labels, len(trainset), n_spks, n_speech)
    train_loader = DataLoader(trainset, batch_sampler = balance_sampler, num_workers = 1, pin_memory = True)
    trainiter = iter(train_loader)
    embedding, label = next(trainiter)
    print(embedding.shape)
    print(label)
