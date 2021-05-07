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
        pass

    def _parse_opts(self, opts):
        pass
    
    def forward(self, wave): 
        '''
        Params:
            wave: wave data, shape is (B, T)
        Returns:
            feature: specified feature, shape is (B, C, T)
        '''
        pass

if __name__ == "__main__":
    import yaml
    f = open("../conf/config.yaml")
    config = yaml.load(f, Loader = yaml.CLoader)
    f.close()
    feature_extractor = FeatureExtractor(config['feature'])
    a = torch.randn(4, 45000)
    a = a / a.abs().max()
    feature = feature_extractor(a)
    print(feature.shape)
