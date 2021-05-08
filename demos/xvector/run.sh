#! /bin/bash

# ln -s `which python3` ../../../python3
# export PYTHONPATH=../../../:$PYTHONPATH
alias python3=/home/acc12416pz/anaconda3/envs/torch1.7/bin/python3

stage=0
# train a frontend module
if [ $stage -le 0 ]; then
    python3 local/nnet/xv_trainer.py --device gpu
fi

# train a backend module
# if [ $stage -le 1 ]; then
#     python3 local/backend/backend_trainer.py
# fi

# evaluation on test set
# if [ $stage -le 2 ]; then
#     python3 evaluation.py
# fi 