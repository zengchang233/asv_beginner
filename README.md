# ASV_Beginner

To newcomer: common tools built by pure PyTorch for automatic speaker verification

## Introduction

ASV_Beginner provides a toolkit, which is built by pure PyTorch, for automatic speaker verification to newcomer in this field.

Thanks to friendly interface of PyTorch, this toolkit can be built in a concise way.

In this toolkit, it contains three parts:

1. `libs`: it provides all functions to users.

2. `conf`: it contains `nnet.yaml` file for configuration of frontend(nnet) module and `backend.yaml` file for configuration of backend module.

3. `demos`: it provides examples of ASV pipeline for some popular architectures, such as X-Vector(TDNN, ETDNN, FTDNN), RawNet and so on.

## Requirements

```
pytorch>=1.8.0
torchaudio>=0.8.0
tqdm
scipy
scikit-learn
librosa
numpy
python_speech_features
```

## Implementation

### dataio

- [x] customized dataset implementation
- [x] librosa style feature and kaldi style feature (implemented by the api of torchaudio)

### components

- [ ] linear transformation components (TDNN layer, FTDNN layer, conv layer ...)
  - [x] tdnn
  - [ ] ftdnn
  - [x] conv
- [ ] pooling components
  - [x] statistic pooling
  - [x] temporal average pooling
  - [ ] attentive statistic pooling
  - [ ] learnable dictionary pooling
- [ ] loss components
  - [x] cross entropy
  - [ ] a-softmax
  - [x] am-softmax(LMCL)
  - [ ] aam-softmax
  - [x] triplet loss

### models (front-end)

- [x] TDNN
- [x] E-TDNN
- [ ] F-TDNN
- [ ] ECAPA-TDNN
- [x] ResNet (18, 34)
- [ ] RawNet
- [ ] Wav2Spk

### back-end

- [ ] Attention Backend
- [ ] PLDA
- [ ] Neural PLDA

### trainer

- [x] trainer basic class

## Demos

### VoxCeleb

- X-Vector recipe
  - [x] TDNN front-end
  - [ ] PLDA back-end
  - [ ] NPLDA back-end
  - [ ] Attention back-end
- ResNet recipe
  - [x] ResNet front-end
  - [ ] PLDA back-end
  - [ ] NPLDA back-end
  - [ ] Attention back-end

### CN-Celeb

- [ ] X-Vector recipe
  - [ ] TDNN front-end
  - [ ] PLDA back-end
  - [ ] NPLDA back-end
  - [ ] Attention back-end
- [ ] ResNet recipe
  - [ ] ResNet front-end
  - [ ] PLDA back-end
  - [ ] NPLDA back-end
  - [ ] Attention back-end

### Results

- VoxCeleb1 (no repeat)
    - TDNN: log fbank (80 dims)
        - clean: 5.10%
        - aug: 4.07%
    - ResCNN
        - log fbank (80 dims) + energy
            - clean: 4.46%
            - aug: %
        - log stft (161 dims)
            - clean: 4.57%
            - aug: &

- CN-Celeb

## TODO

- [ ] Better log information output
- [ ] ...

## Contributors

[Chang ZENG](https://nii-yamagishilab.github.io/)

Meng LIU

## Citation


## Reference


## Licence
