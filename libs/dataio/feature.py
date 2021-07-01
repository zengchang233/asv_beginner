from functools import partial

import torch
import torch.nn as nn
import torchaudio as ta
from torchaudio.functional import *
from torchaudio.transforms import ComputeDeltas, SlidingWindowCmn, Spectrogram, MelSpectrogram, MFCC, MuLawEncoding
import torchaudio.compliance.kaldi as kaldi

from librosa import stft, magphase
from numpy import log1p
from python_speech_features import mfcc, fbank, logfbank, delta

def normalize(feat):
    '''
    params:
        feat: F, T
    '''
    return (feat - feat.mean(axis = 1, keepdims = True)) / (feat.std(axis = 1, keepdims = True) + 2e-12)

def librosa_stft(data, n_fft, hop_length, win_length):
    '''
    return:
        feat: F, T
    '''
    feat = stft(data, n_fft = n_fft, hop_length = hop_length, win_length = win_length)
    feat, _ = magphase(feat)
    feat = log1p(feat)
    feat = normalize(feat)
    return feat

def python_speech_features_mfcc(data, samplerate, n_fft, hop_length, win_length, n_cep, n_mel_bin):
    '''
    return:
        feat: F, T
    '''
    feat = mfcc(data, samplerate = samplerate, nfft = n_fft, winstep = hop_length, winlen = win_length, numcep = n_cep, nfilt = n_mel_bin)
    feat = feat.transpose()
    feat = normalize(feat)
    return feat

def python_speech_features_fbank(data, samplerate, n_fft, hop_length, win_length, n_mel_bin):
    '''
    return:
        feat: F, T
    '''
    # fbank returns fbank matrix and energy vector(one frame has one value)
    feat, energy = fbank(data, samplerate = samplerate, nfft = n_fft, winstep = hop_length, winlen = win_length, nfilt = n_mel_bin)
    feat = np.hstack([feat, energy.reshape(-1, 1)])
    feat = feat.transpose()
    feat = normalize(feat)
    return feat

def python_speech_features_logfbank(data, samplerate, n_fft, hop_length, win_length, n_mel_bin):
    '''
    return:
        feat: F, T
    '''
    feat = logfbank(data, samplerate = samplerate, nfft = n_fft, winstep = hop_length, winlen = win_length, nfilt = n_mel_bin)
    feat = feat.transpose()
    feat = normalize(feat)
    return feat

class FeatureExtractor(nn.Module):
    def __init__(self, rate, feat_type, opts):
        '''
        params: 
            rate: sample rate of speech
            feat_type: feature type, spectrogram, fbank, mfcc
            opts: detail configuration for feature extraction
        '''
        super(FeatureExtractor, self).__init__()
        if feat_type == 'spectrogram': # torchaudio spectrogram
            #  feat = CustomizedSpec(n_fft = opts['n_fft'],
                               #  hop_length = opts['hop_length'],
                               #  win_length = opts['win_length'],
                               #  power = 1)
            #  transform.append(feat)
            self.feat = partial(librosa_stft, n_fft = opts['n_fft'],
                                hop_length = opts['hop_length'],
                                win_length = opts['win_length'])
        elif feat_type == 'fbank': # torchaudio fbank
            #  feat = MelSpectrogram(sample_rate = rate,
                                  #  n_fft = opts['n_fft'],
                                  #  n_mels = opts['n_mels'],
                                  #  hop_length = opts['hop_length'],
            #                        win_length = opts['win_length'])
            self.feat = partial(python_speech_features_fbank, samplerate = rate,
                           n_fft = opts['n_fft'], hop_length = opts['hop_length'],
                           win_length = opts['win_length'], n_mel_bin = opts['n_mels'])
        elif feat_type == 'mfcc': # transforms.MFCC() torchaudio mfcc
            #  feat = MFCC(sample_rate = rate,
                        #  n_mfcc = opts['num_cep'],
                        #  log_mels = opts['log_mels'],
            #              melkwargs = opts['fbank'])
            self.feat = partial(python_speech_features_mfcc, samplerate = rate,
                           n_fft = opts['n_fft'], hop_length = opts['hop_length'],
                           win_length = opts['win_length'], n_mel_bin = opts['n_mels'],
                           n_cep = opts['num_cep'])
        else:
            raise NotImplementedError("Other features are not implemented!")
        #  transform.append(feat)
        #  if opts['normalize']:
        #      cmvn = SlidingWindowCmn(opts['cmvn_window'], norm_vars = False)
        #      transform.append(cmvn)
        #  if opts['delta']:
        #      delta = ComputeDeltas()
        #      transform.append(delta)
        #  self.transform = nn.Sequential(*transform)

    def forward(self, wave):
        '''
        Params:
            wave: wave data, shape is (B, T)
        Returns:
            feature: specified feature, shape is (B, C, T)
        '''
        feature = []
        for i in range(len(wave)):
            feature.append(torch.from_numpy(self.feat(wave[i])).unsqueeze(0))
        feature = torch.cat(feature, dim = 0)
        return feature
        #  feature = self.transform(wave)
        #  feature = self.feat(wave)
        #  feature = self.transform(feature)
        #  return feature

class KaldiFeatureExtractor(nn.Module): 
    def __init__(self, rate, feat_type, opts): 
        super(KaldiFeatureExtractor, self).__init__()
        self.rate = rate
        self.feat_type = feat_type
        self.opts = opts
        if self.feat_type == 'spectrogram': # torchaudio spectrogram
            self.feat = partial(kaldi.spectrogram, frame_length = self.opts['frame_length'],
                                frame_shift = self.opts['frame_shift'],
                                sample_frequency = self.rate)
        elif self.feat_type == 'fbank': # torchaudio fbank
            self.feat = partial(kaldi.fbank, sample_frequency = self.rate,
                                frame_length = self.opts['frame_length'],
                                frame_shift = self.opts['frame_shift'],
                                num_mel_bins = self.opts['num_mel_bins'],
                                use_energy = self.opts['use_energy'],
                                use_log_fbank = self.opts['use_log_fbank'])
        elif self.feat_type == 'mfcc': # torchaudio mfcc
            self.feat = partial(kaldi.mfcc, sample_frequency = self.rate,
                                frame_length = self.opts['frame_length'],
                                frame_shift = self.opts['frame_shift'],
                                num_ceps = self.opts['num_ceps'],
                                num_mel_bins = self.opts['num_mel_bins'],
                                use_energy = self.opts['use_energy'])
        else:
            raise NotImplementedError("Other features are not implemented!")

    def _normalize(self, feature):
        return (feature - feature.mean(dim = 0, keepdims = True)) / (feature.std(axis = 0, keepdims = True) + 2e-12)

    def forward(self, wave): 
        '''
        Params:
            wave: wave data, shape is (B, T)
        Returns:
            feature: specified feature, shape is (B, C, T)
        '''
        feature = []
        for i in range(len(wave)):
            kaldi_feature = self.feat(torch.from_numpy(wave[i].reshape(1, -1)))
            #  print(kaldi_feature.shape)
            feature.append(self._normalize(kaldi_feature).unsqueeze(0))
        feature = torch.cat(feature, dim = 0)
        return feature.transpose(2,1).contiguous()

class RawWaveform(nn.Module):
    def __init__(self, rate, feat_type, opts):
        super(RawWaveform).__init__()
        self.rate = rate
        mu_law_encoding = opts['mu_law_encoding']
        if mu_law_encoding:
            self.transform = MuLawEncoding()
        else:
            self.transform = torch.tensor

    def forward(self, x):
        wave = [self.transform(torch.from_numpy(data.reshape(1, -1))) for data in wave]
        return torch.cat(wave, dim = 0)

if __name__ == "__main__":
    import numpy as np
    import yaml
    f = open("../../conf/data/kaldi_spectrogram.yaml")
    config = yaml.load(f, Loader = yaml.CLoader)
    f.close()
    feature_extractor = KaldiFeatureExtractor(16000, 'spectrogram', config)
    #  a = torch.randn(1, 45000)
    a = np.random.randn(1, 45000)
    a = a / np.abs(a).max()
    #  a = a / a.abs().max()
    feature = feature_extractor(a)
    print(feature)
    print(feature.shape)
