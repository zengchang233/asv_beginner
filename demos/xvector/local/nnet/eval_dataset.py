import sys
import os
sys.path.insert(0, '../../')

import torch
from torch.utils.data import Dataset
import torchaudio as ta

from libs.utils.utils import read_config

class SpeechEvalDataset(Dataset):
    def __init__(self, opts):
        '''
        default sample rate is 16kHz

        '''
        self.root_path = opts['test_root']
        path = opts['test_manifest']
        feat_type = opts['feat_type']
        win_length = opts['win_len']
        hop_length = opts['win_shift']
        rate = opts['rate']
        self.utts = []
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utts.append(line[1])
                self.utts.append(line[2])
        self.utts = list(set(self.utts))
        if 'kaldi' in feat_type:
            from libs.dataio.feature import KaldiFeatureExtractor as FeatureExtractor
        else:
            from libs.dataio.feature import FeatureExtractor
        try:
            feature_opts = read_config("conf/data/{}.yaml".format(feat_type))
        except:
            feature_opts = read_config("../conf/data/{}.yaml".format(feat_type)) # for test
        self.feature_extractor = FeatureExtractor(rate, feat_type.split("_")[-1], feature_opts)

    def _load_audio(self, path, offset = 0, length = -1):
        y, sr = ta.load(path, frame_offset = offset, num_frames = length)
        return y[0], sr

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = self.utts[idx]
        utt_path = os.path.join(self.root_path, utt)
        data, rate = self._load_audio(utt_path)
        feat = self.feature_extractor(data)
        return feat, utt
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    opts = read_config("./conf/data.yaml")
    test_dataset = SpeechEvalDataset(opts)
    feature, uttid = test_dataset[0]
    print(feature.shape)
    print(uttid)