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

import libs.dataio.dataset as dataset
from libs.utils.utils import read_config
from libs.utils.config_parser import ArgParser
from libs.components import loss

class BaseTrainer(object):
    def __init__(self):
        # build model
        self.build_model()

        # mv to device
        self._move_to_device()

        # build dataloader
        self.build_dataloader()

        # build loss
        self.build_criterion()

        # build optimizer
        self.build_optimizer()

        # resume model from saved path
        if os.path.exists(self.train_opts['resume']):
            self.load(self.train_opts['resume'])

    def build_model(self):
        '''
        You MUST overwrite this function.
        And you have to set two attributions in this function.
        1. embedding_dim
        2. model
        '''
        # you MUST overwrite this function
        raise NotImplementedError("Please implement this function by yourself!")

    def build_criterion(self):
        '''
        You can overwrite this function.
        If so, you have to set the following attribution in this function.
        1. criterion
        '''
        raise NotImplementedError("Please implement this function by yourself!")

    def build_optimizer(self): 
        '''
        You can overwrite this function.
        If so, you have to set the following attribution in this function.
        1. optim
        2. lr_scheduler
        '''
        raise NotImplementedError("Please implement this function by yourself!")

    def build_dataloader(self): 
        '''
        You can overwrite this function.
        If so, you have to set the following attribution in this function.
        1. trainloader 
        '''
        raise NotImplementedError("Please implement this function by yourself!")
    
    def train_epoch(self): 
        # you MUST overwrite this function
        raise NotImplementedError("Please implement this function by yourself!")

    def _dev(self): 
        raise NotImplementedError("Please implement this function by yourself!")

    def _move_to_device(self):
        if self.train_opts['device'] == 'cuda':
            print("using gpus")
            self.device = torch.device('cuda')
            if self.mode == 'test':
                device_ids = [0]
                device_num = 1
            else:
                device_ids = self.train_opts['gpus_id']
                device_num = torch.cuda.device_count()
                if device_num >= len(device_ids):
                    device_num = len(device_ids)
                else:
                    logging.warn('There are only {} GPU cards in this machine, using all of them'.format(device_num))
                    device_ids = list(range(device_num))
            self.model = torch.nn.DataParallel(self.model.to(self.device), device_ids = device_ids)
            logging.info("Using GPU: {}".format(device_ids))
            self.device_num = device_num
        else:
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
            logging.info("Using CPU")
            self.device_num = 1

    def model_average(self, avg_num = 4):
        model_state_dict = {}
        for i in range(avg_num):
            suffix = self.current_epoch - i
            ckpt = torch.load('exp/{}/net_{}.pth'.format(self.log_time, suffix))
            state_dict = ckpt['state_dict']
            for k, v in state_dict.items():
                if k in model_state_dict:
                    model_state_dict[k] += v
                else:
                    model_state_dict[k] = v
        for k, v in model_state_dict.items():
            model_state_dict[k] = v / avg_num
        torch.save({'epoch': 0, 'state_dict': model_state_dict,
                    'optimizer': ckpt['optimizer']},
                    'exp/{}/net_avg.pth'.format(self.log_time))
        self.model.load_state_dict(model_state_dict)

    def save(self, filename = None):
        model = self.model.module # DO NOT save DataParallel wrapper
        if filename is None:
            torch.save({'epoch': self.current_epoch, 'state_dict': model.state_dict(), 'criterion': self.criterion.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict(), 'optimizer': self.optim.state_dict()},
                        'exp/{}/net_{}.pth'.format(self.log_time, self.current_epoch))
        else:
            torch.save({'epoch': self.current_epoch, 'state_dict': model.state_dict(), 'criterion': self.criterion.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict(), 'optimizer': self.optim.state_dict()},
                        'exp/{}/{}'.format(self.log_time, filename))

    def load(self, resume):
        ckpt = torch.load(resume)
        if self.train_opts['device'] == 'cuda':
            self.model.module.load_state_dict(ckpt['state_dict'])
        else:
            self.model.load_state_dict(ckpt['state_dict'])
        if 'criterion' in ckpt and isinstance(ckpt['criterion'], dict):
            # print("criterion state dict")
            self.criterion.load_state_dict(ckpt['criterion'])
        else:
            # print("criterion object")
            self.criterion = ckpt['criterion']
        if 'lr_scheduler' in ckpt:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        self.optim.load_state_dict(ckpt['optimizer'])
        self.current_epoch = ckpt['epoch']

    def train(self):
        start_epoch = self.current_epoch
        self.best_dev_epoch = self.current_epoch
        self.best_dev_loss = 1000
        self.count = 0
        self.dev_check_count = 0
        for epoch in range(start_epoch + 1, self.epoch + 1):
            self.current_epoch = epoch
            logging.info("Epoch {}".format(self.current_epoch))
            stop = self.train_epoch()
            self.save()
            if stop == -1:
                break 

    def _reset_opts(self, module, opts):
        if module == 'data':
            for k, v in opts.items():
                self.data_opts[k] = v
        elif module == 'model':
            for k, v in opts.items():
                self.model_opts[k] = v
        elif module == 'train':
            for k, v in opts.items():
                self.train_opts[k] = v

    def __call__(self):
        os.makedirs('exp/{}/conf'.format(self.log_time), exist_ok = True)
        if not os.path.exists("exp/{}/conf/data.yaml".format(self.log_time)):
            with open("exp/{}/conf/data.yaml".format(self.log_time), 'w') as f:
                yaml.dump(self.data_opts, f)
        if not os.path.exists("exp/{}/conf/model.yaml".format(self.log_time)):
            with open("exp/{}/conf/model.yaml".format(self.log_time), 'w') as f:
                yaml.dump(self.model_opts, f)
        if not os.path.exists("exp/{}/conf/train.yaml".format(self.log_time)):
            with open("exp/{}/conf/train.yaml".format(self.log_time), 'w') as f:
                yaml.dump(self.train_opts, f)
        logging.info("start training")
        self.train()

if __name__ == "__main__":
    def main():
        parser = ArgParser()
        args = parser.parse_args()
        args = vars(args)
        data_config = read_config("../../conf/data.yaml")
        model_config = read_config("../../conf/model.yaml")
        train_config = read_config("../../conf/train.yaml")
        trainer = NNetTrainer(data_config, model_config, train_config, args)

    main()
