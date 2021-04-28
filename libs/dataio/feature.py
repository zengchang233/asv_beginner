import torch
import torch.nn as nn
import torchaudio as ta
from torchaudio.functional import *
from torchaudio.transforms import *

class FeatureExtractor(nn.Module):
    def __init__(self, opts):
        super(FeatureExtractor, self).__init__()
        transform = []
        self.rate = opts['rate']
        self.feat_type = opts['feat_type']
        self.opts = opts[self.feat_type] # can choose mfcc or fbank as input feat
        if self.feat_type == 'spectrogram': # torchaudio spectrogram
            feat = Spectrogram(n_fft = self.opts['n_fft'],
                               hop_length = int(self.rate * self.opts['win_shift']),
                               win_length = int(self.rate * self.opts['win_len']))
        elif self.feat_type == 'mel_spectrogram': # torchaudio fbank
            feat = MelSpectrogram(sample_rate = self.rate,
                                  n_fft = self.opts['n_fft'],
                                  n_mels = self.opts['num_bin'],
                                  hop_length = int(self.rate * self.opts['win_shift']),
                                  win_length = int(self.rate * self.opts['win_len']))
        elif self.feat_type == 'mfcc': # transforms.MFCC() torchaudio mfcc
            feat = MFCC(sample_rate = self.rate,
                        n_mfcc = self.opts['num_cep'],
                        n_fft = self.opts['n_fft'],
                        n_mel = self.opts['num_bin'],
                        hop_length = int(self.rate * self.opts['win_shift']),
                        win_length = int(self.rate * self.opts['win_len']))
        else:
            raise NotImplementedError("Other features are not implemented!")
        transform.append(feat)
        if self.opts['normalize']:
            cmvn = SlidingWindowCmn(self.opts['cmvn_window'], norm_vars = False)
            transform.append(cmvn)
        if self.opts['delta']:
            delta = ComputeDeltas()
            transform.append(delta)
        self.transform = nn.Sequential(*transform)

    def _parse_opts(self, opts):
        pass

    def forward(self, wave):
        '''
        Params:
            wave: wave data, shape is (B, T)
        Returns:
            feature: specified feature, shape is (B, C, T)
        '''
        feature = self.transform(wave)
        return feature

if __name__ == "__main__":
    import yaml
    f = open("../conf/config.yaml")
    config = yaml.load(f, Loader = yaml.CLoader)
    f.close()
    feature_extractor = FeatureExtraxtor(config['feature'])
    a = torch.randn(4, 45000)
    a = a / a.abs().max()
    feature = feature_extractor(a)
    print(feature.shape)
