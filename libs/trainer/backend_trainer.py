import sys
import time
import logging
import yaml
import os
sys.path.insert(0, "../../")

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

from libs.trainer import base_trainer
from libs.utils.utils import read_config
from libs.utils.config_parser import ArgParser

class BackendTrainer(base_trainer.BaseTrainer):
    def __init__(self, data_opts = None, model_opts = None, train_opts = None, args = None):
        self.data_opts = data_opts
        self.model_opts = model_opts
        self.train_opts = train_opts
        self.mode = args['mode']
        if "resume" in args:
            self.train_opts['resume'] = args["resume"]
        if os.path.exists(self.train_opts['resume']):
            self.log_time = self.train_opts['resume'].split('/')[1]
        else:
            self.log_time = time.asctime(time.localtime(time.time())).replace(' ', '_').replace(':', '_')
        logging.info(self.log_time)

        self.current_epoch = 0
        self.epoch = self.train_opts['epoch']
        logging.info("Total train {} epochs".format(self.epoch))
        self.grad_clip = self.train_opts['grad_clip']
        self.grad_clip_threshold = self.train_opts['grad_clip_threshold']
        self.hard_negative_mining = self.train_opts['hard_negative_mining']
        super(BackendTrainer, self).__init__()

    def build_model(self):
        '''
        You MUST overwrite this function.
        And you have to set two attributions in this function.
        1. embedding_dim
        2. model
        '''
        # you MUST overwrite this function
        if __name__ == '__main__':
            print("build model")
        else:
            raise NotImplementedError("Please implement this function by yourself!")

    def build_criterion(self):
        '''
        You can overwrite this function.
        If so, you have to set the following attribution in this function.
        1. criterion
        '''
        if __name__ == '__main__':
            print('build criterion')
        else:
            raise NotImplementedError("Please implement this function by yourself!")

    def build_optimizer(self): 
        '''
        You can overwrite this function.
        If so, you have to set the following attribution in this function.
        1. optim
        2. lr_scheduler
        '''
        if __name__ == '__main__':
            print('build optimizer')
        else:
            raise NotImplementedError("Please implement this function by yourself!")

    def build_dataloader(self): 
        '''
        You can overwrite this function.
        If so, you have to set the following attribution in this function.
        1. trainloader 
        '''
        if __name__ == '__main__':
            print('build dataloader')
        else:
            raise NotImplementedError("Please implement this function by yourself!")
        

    def train_epoch(self): 
        # you MUST overwrite this function
        raise NotImplementedError("Please implement this function by yourself!")

    def _dev(self): 
        raise NotImplementedError("Please implement this function by yourself!")

if __name__ == "__main__":
    trainer = BackendTrainer()
    trainer.train()
