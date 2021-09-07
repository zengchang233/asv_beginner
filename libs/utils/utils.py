from itertools import combinations

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler

def read_config(file_path): 
    f = open(file_path, 'r')
    config = yaml.load(f, Loader = yaml.CLoader)
    f.close()
    return config

def save_config():
    pass

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, all_speech, n_classes, n_samples):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = all_speech
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class BalancedBatchSamplerV2(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, n_classes, n_samples, classes_per_batch, samples_per_class):
        self.count = 0
        if type(n_classes) == int:
            self.n_classes = list(range(0, n_classes))
        elif type(n_classes) == list:
            self.n_classes = list(set(n_classes))
        self.n_samples = n_samples 
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batch_size = self.samples_per_class * self.classes_per_batch

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_samples:
            classes = np.random.choice(self.n_classes, self.classes_per_batch, replace=False)
            indices = []
            for class_ in classes:
                indices += [class_ for i in range(self.samples_per_class)]
            yield indices
            self.count += self.classes_per_batch * self.samples_per_class

    def __len__(self):
        return self.n_samples // self.batch_size
