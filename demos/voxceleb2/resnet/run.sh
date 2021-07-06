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
    # --device gpu or cpu
    # --resume resume path
    # --mode depreciated
    python local/nnet/trainer.py \
        --feat-type kaldi_fbank \
        --input-dim 1 \
        --arch resnet \
        --loss AMSoftmax \
        --bs 64 \
        --device cuda \
        --resume exp/Mon_Jul__5_01_27_53_2021/net_7.pth
    echo "frontend training done!"
    exit 0;
fi

##### Result #####
# model config: [64,128,256], [1,1,1]
# feature: 161 dims stft
# loss: AMSoftmax (s=20,m=0.3)
# voxceleb1 dev as training set, voxceleb1 test as test set
    # no augmentation, no repeat: EER: 4.57%
    # no augmentation, repeat: EER: 4.35%
    # augmentation, no repeat: EER: 3.82% (xvector in kaldi EER is 5.302% as reference)
    # augmentation, repeat: EER: % (xvector in kaldi EER is 5.302% as reference)
# voxceleb2 dev and test as training set, voxceleb1 test as test set
    # no augmentation, no repeat: EER: 2.87%
    # 

# evaluation on test set without backend (or using cosine backend)
if [ $stage -le 1 ]; then
    expdir=$1
    start=$2
    stop=$3
    for x in `seq $start $stop`; do
        model=`basename $x`
        python local/evaluation.py -e $expdir -m net_${x}.pth -d cuda -l far
        # python local/evaluation.py -e $expdir -m best_dev_model.pth -d cuda -l far
    done
fi

echo "scoring with only frontend done!"
exit 0;

# train a backend module
# if [ $stage -le 2 ]; then
#     python local/backend/backend_trainer.py
# fi

# evaluation on test set
if [ $stage -le 3 ]; then
    python local/evaluation.py -e Sat_May__8_11_24_59_2021 -m best_dev_model.pth -d gpu -l far
fi 
