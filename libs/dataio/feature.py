from functools import partial

import torch
import torch.nn as nn
import torchaudio as ta
from torchaudio.functional import *
from torchaudio.transforms import ComputeDeltas, SlidingWindowCmn, Spectrogram, MelSpectrogram, MFCC
from torchaudio.compliance.kaldi import *

class FeatureExtractor(nn.Module):
    def __init__(self, rate, feat_type, opts):
        '''
        params: 
            rate: sample rate of speech
            feat_type: feature type, spectrogram, fbank, mfcc
            opts: detail configuration for feature extraction
        '''
        super(FeatureExtractor, self).__init__()
        transform = []
        if feat_type == 'spectrogram': # torchaudio spectrogram
            feat = Spectrogram(n_fft = opts['n_fft'],
                               hop_length = opts['hop_length'],
                               win_length = opts['win_length'])
        elif feat_type == 'fbank': # torchaudio fbank
            feat = MelSpectrogram(sample_rate = rate,
                                  n_fft = opts['n_fft'],
                                  n_mels = opts['n_mels'],
                                  hop_length = opts['hop_length'],
                                  win_length = opts['win_length'])
        elif feat_type == 'mfcc': # transforms.MFCC() torchaudio mfcc
            feat = MFCC(sample_rate = rate,
                        n_mfcc = opts['num_cep'],
                        log_mels = opts['log_mels'], 
                        melkwargs = opts['fbank'])
        else:
            raise NotImplementedError("Other features are not implemented!")
        transform.append(feat)
        if opts['normalize']:
            cmvn = SlidingWindowCmn(opts['cmvn_window'], norm_vars = False)
            transform.append(cmvn)
        if opts['delta']:
            delta = ComputeDeltas()
            transform.append(delta)
        self.transform = nn.Sequential(*transform)

    def forward(self, wave):
        '''
        Params:
            wave: wave data, shape is (B, T)
        Returns:
            feature: specified feature, shape is (B, C, T)
        '''
        feature = self.transform(wave)
        return feature

class KaldiFeatureExtractor(nn.Module): 
    def __init__(self, rate, feat_type, opts): 
        super(KaldiFeatureExtractor, self).__init__()
        self.rate = rate
        self.feat_type = feat_type
        self.opts = opts
        if self.feat_type == 'spectrogram': # torchaudio spectrogram
            self.feat = partial(spectrogram, frame_length = self.opts['frame_length'],
                           frame_shift = self.opts['frame_shift'],
                           sample_frequency = self.rate)
        elif self.feat_type == 'fbank': # torchaudio fbank
            self.feat = partial(fbank, sample_frequency = self.rate,
                           frame_length = self.opts['frame_length'],
                           frame_shift = self.opts['frame_shift'],
                           num_mel_bins = self.opts['num_mel_bins'],
                           use_energy = self.opts['use_energy'],
                           use_log_fbank = self.opts['use_log_fbank'])
        elif self.feat_type == 'mfcc': # transforms.MFCC() torchaudio mfcc
            self.feat = partial(mfcc, sample_frequency = self.rate,
                           frame_length = self.opts['frame_length'],
                           frame_shift = self.opts['frame_shift'],
                           num_ceps = self.opts['num_ceps'],
                           num_mel_bins = self.opts['num_mel_bins'],
                           use_energy = self.opts['use_energy'])
        else:
            raise NotImplementedError("Other features are not implemented!")

    def forward(self, wave): 
        '''
        Params:
            wave: wave data, shape is (B, T)
        Returns:
            feature: specified feature, shape is (B, C, T)
        '''
        feature = []
        for _ in range(wave.size(0)):
            feature.append(self.feat(wave).unsqueeze(0))
        feature = torch.cat(feature, dim = 0)
        return feature.transpose(2,1).contiguous()

if __name__ == "__main__":
    import yaml
    f = open("../../conf/data/kaldi_mfcc.yaml")
    config = yaml.load(f, Loader = yaml.CLoader)
    f.close()
    feature_extractor = KaldiFeatureExtractor(16000, 'mfcc', config)
    a = torch.randn(4, 45000)
    a = a / a.abs().max()
    feature = feature_extractor(a)
    print(feature.shape)
