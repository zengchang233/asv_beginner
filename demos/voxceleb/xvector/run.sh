#! /bin/bash

alias python=`which python3`

stage=1
# train a frontend module
if [ $stage -le 0 ]; then
    # options:
    # --feat-type {python,kaldi}_{fbank,mfcc,spectrogram}
    # --input-dim 1 for resnet, feature dimension for xvector
    # --arch resnet, tdnn, etdnn, ... take ../../../libs/nnet/*.py as reference
    # --loss CrossEntropy, AMSoftmax, TripletLoss, ...
    # --bs batch size
    # --resume resume path
    # --device gpu or cpu
    # --mode depreciated
    python local/nnet/trainer.py --feat-type kaldi_fbank --arch tdnn --input-dim 80 \
        --device cuda --bs 64 --loss AMSoftmax # --resume exp/Sun_Jul_11_10_48_55_2021/net_7.pth
    echo "frontend training done!"
    exit 0;
fi

##### Result #####
# model config: 
    # TDNN layers: [512,512,512,512,1500], [[-2,-1,0,1,2],[-2,0,2],[-3,0,3],[0],[0]]
    # FC layers: fc1+activation+bn, fc2 (without activation and bn), fc3 + softmax
# feature: 80 dims kaldi fbank (log = true)
# data repeat: false
# loss: AMSoftmax (s=20,m=0.2)
# voxceleb1 dev as training set, voxceleb1 test as test set
    # no augmentation, no repeat: EER: 5.10% (xvector in kaldi EER is 5.302% as reference)
    # no augmentation, repeat: EER: 4.97% (xvector in kaldi EER is 5.302% as reference)
    # augmentation, no repeat: EER: 4.07% (xvector in kaldi EER is 5.302% as reference)
    # augmentation, repeat: EER: % (xvector in kaldi EER is 5.302% as reference)
    # augmentation, ASP (implemented by myself), no repeat: 3.94% (using asv subtools implementation, the result is 4.05%)
    # augmentation, MultiHeadAttentionPooling, no repeat: 3.87%
    # augmentation, MultiResolutionAttentionPooling, no repeat: 3.99%

# evaluation on test set without backend (or using cosine backend)
if [ $stage -le 1 ]; then
    expdir=$1
    start=$2
    stop=$3
    # for x in `seq $start $stop`; do
    #     python local/evaluation.py -e $expdir -m net_${x}.pth -d cuda -l far
        python local/evaluation.py -e $expdir -m best_dev_model.pth -d cuda -l far
    # done
    echo "scoring with only frontend done!"
    exit 0;
fi

# train a backend module
# if [ $stage -le 1 ]; then
#     python local/backend/backend_trainer.py
# fi

# evaluation on test set
if [ $stage -le 2 ]; then
    python local/evaluation.py -e Sat_May__8_11_24_59_2021 -m best_dev_model.pth -d gpu -l far
fi 
