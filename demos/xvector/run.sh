#! /bin/bash

# train a frontend module
python3 local/nnet/xv_trainer.py

# train a backend module
python3 local/backend/backend_trainer.py

# evaluation on test set
python3 evaluation.py
