import csv
import os

import soundfile as sf
from tqdm import tqdm
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import sys
import numpy as np
from multiprocessing import Pool, Manager

SAMPLE_RATE = 16000
MANIFEST_DIR = "/home/smg/zengchang/data/manifest/{}_manifest.csv"
os.makedirs(os.path.dirname(MANIFEST_DIR), exist_ok = True)

def read_manifest(dataset, start = 0):
    n_speakers = 0
    rows = []
    with open(MANIFEST_DIR.format(dataset), 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in reader:
            rows.append([int(sid) + start, aid, filename, duration, samplerate])
            n_speakers = int(sid) + 1
    return n_speakers, rows

def save_manifest(dataset, rows):
    rows.sort()
    with open(MANIFEST_DIR.format(dataset), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def create_manifest_librispeech():
    dataset = 'SLR12'
    n_speakers = 0
    log = []
    sids = dict()
    for m in ['train-clean-100', 'train-clean-360']:
        train_dataset = '/home/zeng/zeng/datasets/librispeech/{}'.format(m)
        for speaker in tqdm(os.listdir(train_dataset), desc = dataset):
            speaker_dir = os.path.join(train_dataset, speaker)
            if os.path.isdir(speaker_dir):
                speaker = int(speaker)
                if sids.get(speaker) is None:
                    sids[speaker] = n_speakers
                    n_speakers += 1
                for task in os.listdir(speaker_dir):
                    task_dir = os.path.join(speaker_dir, task)
                    aid = 0
                    for audio in os.listdir(task_dir):
                        if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1):
                            filename = os.path.join(task_dir, audio)
                            info = sf.info(filename)
                            log.append((sids[speaker], aid, filename, info.duration, info.samplerate))
                        aid += 1
    save_manifest(dataset, log)

def make_speaker_list():
    train_dataset = '/home/smg/zengchang/data/voxceleb1/dev/wav'
    speakers_path_list = []
    count = 0
    for speaker in os.listdir(train_dataset):
        speaker_dir = os.path.join(train_dataset, speaker)
        speakers_path_list.append(speaker_dir)
        count += 1

    n = 5 
    res = [speakers_path_list[i:i+n] for i in range(0, len(speakers_path_list), n)] 
    for i in range(len(res)):
        res[i].append(i*n)
    return res

def create_manifest_voxceleb1(speakers_path):
    dataset = 'voxceleb1'
    n_speakers = speakers_path[-1]
    global log
    train_dataset = '/home/smg/zengchang/data/voxceleb1/dev/wav'
    
    for speaker in tqdm(speakers_path[:-1], desc = dataset):
        #  speaker_dir = os.path.join(train_dataset, speaker)
        aid = 0
        for sub_speaker in os.listdir(speaker):
            sub_speaker_path = os.path.join(speaker, sub_speaker)
            if os.path.isdir(sub_speaker_path):
                for audio in os.listdir(sub_speaker_path):
                    if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1):
                        filename = os.path.join(sub_speaker_path, audio)
                        info = sf.info(filename)
                        log.append((n_speakers, aid, filename, info.duration, info.samplerate))
                        aid += 1
        n_speakers += 1
    #  save_manifest(dataset, log)
    
def create_manifest_voxceleb2(speakers_path):
    dataset = 'voxceleb2'
    n_speakers = speakers_path[-1]
    global log
    train_dataset = '/home/smg/zengchang/data/voxceleb2/dev/wav'
    for speaker in tqdm(speakers_path[:-1], desc = dataset):
        #  speaker_dir = os.path.join(train_dataset, speaker)
        aid = 0
        for sub_speaker in os.listdir(speaker):
            sub_speaker_path = os.path.join(speaker, sub_speaker)
            if os.path.isdir(sub_speaker_path):
                for audio in os.listdir(sub_speaker_path):
                    if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1):
                        filename = os.path.join(sub_speaker_path, audio)
                        info = sf.info(filename)
                        log.append((n_speakers, aid, filename, info.duration, info.samplerate))                    
                        aid += 1
        n_speakers += 1
    #  save_manifest(dataset, log)

def merge_manifest(datasets, dataset):
    rows = []
    n = len(datasets)
    start = 0
    for i in range(n):
        n_speakers, temp = read_manifest(datasets[i], start = start)
        rows.extend(temp)
        start += n_speakers
    with open(MANIFEST_DIR.format(dataset), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if __name__ == '__main__':
    log = Manager().list()
    speakers_path_list = make_speaker_list()
    #  create_manifest_voxceleb1()
    with Pool(40) as p:
        p.map(create_manifest_voxceleb1, speakers_path_list)
    save_manifest("voxceleb1_augment", log)
    #  create_manifest_voxceleb2()
    #  merge_manifest(['aug_voxceleb2_dev', 'aug_voxceleb2_test'], 'aug_voxceleb2')
