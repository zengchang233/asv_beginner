#! /bin/bash

alias python=`which python3`

stage=0
# train a frontend module
if [ $stage -le 0 ]; then
    python local/nnet/trainer.py --arch resnet --device cuda --bs 64 --feat-type kaldi_fbank --input-dim 80 # --resume exp/Sun_Jun_13_12_36_33_2021/net_29.pth
fi

##### Result #####
# model config: [64,128,256], [1,1,1]
# feature: 161 dims stft
# loss: AMSoftmax (s=20,m=0.3)
# voxceleb1 dev as training set, voxceleb1 test as test set
    # no augmentation: EER: 4.35%
    # augmentation with MUSAN and RIRS: EER: 3.82% (xvector in kaldi EER is 5.302% as reference)
echo "frontend training done!"
exit 0;

# evaluation on test set without backend (or using cosine backend)
if [ $stage -le 1 ]; then
    expdir=$1
    start=$2
    stop=$3
    for x in `seq $start $stop`; do
        # model=`basename $x`
        python local/evaluation.py -e $expdir -m net_${x}.pth -d cuda -l far
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
