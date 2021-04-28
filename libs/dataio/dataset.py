import os
import random
import logging
import warnings
warnings.filterwarnings('ingore')

from torch.utils.data import Dataset
import torchaudio as ta
from torchaudio.transforms import *
from torchaudio.compliance.kaldi import *

class SpeechTrainDataset(Dataset):
    def __init__(self, opts):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def collate_fn(self, batch):
        pass

    def _load_audio(self, path, offset = 0, length = None):
        pass
