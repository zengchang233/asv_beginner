#! /bin/bash

alias python=`which python3`

stage=1
# train a frontend module
if [ $stage -le 0 ]; then
    python local/nnet/xv_trainer.py --device gpu --bs 128 --feat-type python_fbank --input-dim 80 --resume exp/Tue_Jun__8_03_00_40_2021/net_7.pth
fi

echo "frontend training done!"

# evaluation on test set without backend (or using cosine backend)
if [ $stage -le 2 ]; then
    python local/evaluation.py -e Tue_Jun__8_03_00_40_2021 -m net_50.pth -d cuda -l far
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
