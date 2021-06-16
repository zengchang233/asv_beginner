#! /bin/bash

alias python=`which python3`

stage=0
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
    python local/nnet/trainer.py --device cuda --arch tdnn --bs 64 --feat-type kaldi_mfcc --input-dim 23 \
        --loss AMSoftmax # --resume exp/Tue_Jun__8_03_00_40_2021/net_7.pth
fi

echo "frontend training done!"
exit 0;

# evaluation on test set without backend (or using cosine backend)
if [ $stage -le 1 ]; then
    expdir=$1
    start=$2
    stop=$3
    for x in `seq $start $stop`; do
        python local/evaluation.py -e $expdir -m net_${x}.pth -d cuda -l far
    done
fi

echo "scoring with only frontend done!"
exit 0;

# train a backend module
# if [ $stage -le 1 ]; then
#     python local/backend/backend_trainer.py
# fi

# evaluation on test set
if [ $stage -le 2 ]; then
    python local/evaluation.py -e Sat_May__8_11_24_59_2021 -m best_dev_model.pth -d gpu -l far
fi 
