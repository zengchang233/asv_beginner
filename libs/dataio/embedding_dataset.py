import os
import random
import sys
sys.path.insert(0, '../../')

from torch.utils.data import Dataset, DataLoader, BatchSampler
import numpy as np
import torch

from libs.utils import utils

class EmbeddingTrainDataset(Dataset):
    def __init__(self, opts):
        path = opts['path']
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

class EmbeddingEnrollDataset(Dataset):
    def __init__(self, path = None):
        if path is None:
            path = '../../data/cnceleb_xv/enroll_mean'
        self.dataset = []
        self.count = 0
        self.labels = []
        spk_idx = 0
        for speaker in os.listdir(path):
            speaker_path = os.path.join(path, speaker)
            self.dataset.append(speaker_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        speaker_path = self.dataset[idx]
        embeddings = os.listdir(speaker_path)
        all_embeddings = []
        for embedding in embeddings:
            embedding_path = os.path.join(speaker_path, embedding)
            embedding = np.load(embedding_path)
            all_embeddings.append(embedding.reshape(-1))
        embedding = torch.from_numpy(np.array(all_embeddings))
        return embedding.unsqueeze(0), os.path.basename(speaker_path)

    def __call__(self):
        idx = 0
        while idx < len(self.dataset):
            embedding, spk = self.__getitem__(idx)
            yield embedding, spk
            idx += 1

if __name__ == '__main__':
    opt = {}
    opt['path'] = "../exp/Mon_Feb_22_09:43:37_2021/train_xv"
    #  trainset = EmbeddingTrainDataset(opt)
    #  labels = trainset.labels
    #  speech_number = len(trainset)
    #  n_spks = 5
    #  n_speech = 4
    #  balance_sampler = BalancedBatchSampler(labels, n_spks, n_speech)
    #  train_loader = DataLoader(trainset, batch_sampler = balance_sampler, num_workers = 1, pin_memory = True)
    #  trainiter = iter(train_loader)
    #  embedding, label = next(trainiter)
    enrollset = EmbeddingEnrollDataset()
    for idx, (embedding, spk) in enumerate(enrollset()):
        print(embedding.shape)
        print(idx, spk)
