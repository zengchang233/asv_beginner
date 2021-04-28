import os
import random
import logging
import warnings
warnings.filterwarnings('ingore')

from torch.utils.data import Dataset
import torchaudio as ta
from torchaudio.transforms import *
from torchaudio.compliance.kaldi import *

from asv_beginner.libs.dataio.feature import FeatureExtractor

class SpeechTrainDataset(Dataset):
    def __init__(self, opts):
        frame_range = opts['frames'] # frame number range in training
        self.lower_frame_num = frame_range[0]
        self.higher_frame_num = frame_range[1]
        TRAIN_MANIFEST = opts['train_manifest']
        self.rate = opts['rate']
        self.opts = opts
        self.dataset = []
        current_sid = -1
        total_duration = 0
        with open(TRAIN_MANIFEST, 'r') as f:
            reader = csv.reader(f)
            for sid, aid, filename, duration, samplerate in reader:
                if sid != current_sid:
                    self.dataset.append([])
                    current_sid = sid
                self.dataset[-1].append((filename, float(duration), int(samplerate)))
                total_duration += eval(duration)
        self.n_spk = len(self.dataset)
        self.split_train_dev(opts['dev_number'])
        total_duration -= self.dev_total_duration
        mean_duration_per_utt = (np.mean(frame_range) - 1) * self.opts['win_shift'] + self.opts['win_len']
        self.count = math.floor(total_duration / mean_duration_per_utt) # make sure each sampling point in data will be used
        self.feature_extractor = feature.FeatureExtractor(opts[opts['data_format']])

    def split_train_dev(self, dev_number = 1000):
        self.dev = []
        self.dev_total_duration = 0
        self.dev_number = dev_number
        i = 0
        while i < dev_number:       
            spk = random.randint(0, self.n_spk - 1)
            if len(self.dataset[spk]) <= 1:
                continue 
            utt_idx = random.randint(0, len(self.dataset[spk]) - 1)
            utt = self.dataset[spk][utt_idx]
            self.dev.append((utt, spk))
            self.dev_total_duration += utt[1]
            del self.dataset[spk][utt_idx]
            i += 1

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        idx = idx % self.n_spk
        return idx

    def collate_fn(self, batch):
        frame = random.randint(self.lower_frame_num, self.higher_frame_num) # random select a frame number in uniform distribution
        duration = (frame - 1) * self.opts['win_shift'] + self.opts['win_len'] # duration in time of one training speech segment
        samples_num = int(duration * self.rate) # duration in sample point of one training speech segment
        wave = []
        for sid in batch:
            speaker = self.dataset[sid]
            y = []
            n_samples = 0
            while n_samples < samples_num:
                aid = random.randrange(0, len(speaker))
                audio = speaker[aid]
                t, sr = audio[1], audio[2]
                samples_len = int(t * sr)
                start = int(random.uniform(0, t) * sr) # random select start point of speech
                _y, _ = self._load_audio(audio[0], start = start, stop = samples_len) # read speech data from start point to the end
                if _y is not None:
                    y.append(_y)
                    n_samples += len(_y)
            y = torch.cat(y, dim = 0)[:samples_num]
            y = y.unsqueeze(0)
            wave.append(y)
        wave = torch.cat(wave, dim = 0)
        labels = torch.tensor(batch)
        feats = self.feature_extractor(wave)
        return feats, labels

    def _load_audio(self, path, offset = 0, length = -1):
        y, sr = ta.load(path, frame_offset = offset, num_frames = length)
        return y[0], sr

    def get_dev_data(self):
        idx = 0
        while idx < self.dev_number:
            (wav_path, duration, rate), spk = self.dev[idx]
            data, _ = self._load_audio(wav_path)
            feat = self.feature_extractor(data)
            yield feat.unsqueeze(0), torch.LongTensor([spk])
            idx += 1

    def __call__(self):
        idx = 0
        wavlist = []
        spk = []
        for ind, i in enumerate(self.dataset):
            wavlist.extend(i)
            spk.extend([ind] * len(i))
        while idx < len(wavlist):
            wav_path, duration, rate = wavlist[idx]
            data, _ = self.load(wav_path)
            feat = self.feature_extractor(data)
            yield feat, spk[idx], os.path.basename(wav_path).replace('.wav', '.npy')
            #  yield feat, wav_path
            idx += 1 
