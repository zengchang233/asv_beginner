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
import libs.dataio.dataset as dataset
from libs.utils.utils import read_config
from libs.utils.config_parser import ArgParser
from libs.components import loss

class NNetTrainer(base_trainer.BaseTrainer):
    def __init__(self, data_opts = None, model_opts = None, train_opts = None, args = None):
        # set configuration according to options parsed by argparse module
        # print(args)
        self.mode = args['mode']
        data_opts['feat_type'] = args['feat_type']
        model_opts['arch'] = args["arch"]
        self.input_dim = args['input_dim']
        train_opts['loss'] = args["loss"]
        train_opts['bs'] = args['bs']
        train_opts['device'] = args['device']
        if "resume" in args:
            train_opts['resume'] = args["resume"]

        # if the path corresponding to resume option exists, resume training process from exp dir
        if os.path.exists(train_opts['resume']):
            self.log_time = train_opts['resume'].split('/')[1]
        else:
            self.log_time = time.asctime(time.localtime(time.time())).replace(' ', '_').replace(':', '_')
        logging.info(self.log_time)

        self.train_opts = train_opts
        self.model_opts = model_opts
        self.data_opts = data_opts

        # initialize training and dev dataset
        self.trainset = dataset.SpeechTrainDataset(self.data_opts)
        logging.info("using {} to extract features".format(args['feat_type'].split('_')[0]))
        logging.info("Using {} feature".format(args['feat_type'].split('_')[-1]))
        self.n_spk = self.trainset.n_spk

        self.epoch = self.train_opts['epoch']
        logging.info("Total train {} epochs".format(self.epoch))
        self.current_epoch = 0
        super(NNetTrainer, self).__init__()

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
        #  raise NotImplementedError("Please implement this function by yourself!")

    def build_criterion(self):
        '''
        You can overwrite this function.
        If so, you have to set the following attribution in this function.
        1. criterion
        '''
        if self.train_opts['loss'] == 'CrossEntropy':
            self.criterion = loss.CrossEntropy(self.embedding_dim, self.n_spk).to(self.device)
        elif self.train_opts['loss'] == 'AMSoftmax':
            self.criterion = loss.AMSoftmax(self.embedding_dim, self.n_spk, self.train_opts['scale'], self.train_opts['margin']).to(self.device)
        elif self.train_opts['loss'] == 'TripletLoss':
            from libs.utils.utils import RandomNegativeTripletSelector, SemihardNegativeTripletSelector, HardestNegativeTripletSelector
            self.margin = self.train_opts['margin']
            if self.train_opts['selector'] == 'hardest':
                selector = HardestNegativeTripletSelector(self.margin)
            elif self.train_opts['selector'] == 'semihard':
                selector = SemihardNegativeTripletSelector(self.margin)
            elif self.train_opts['selector'] == 'randomhard':
                selector = RandomNegativeTripletSelector(self.margin)
            else:
                raise NotImplementedError("Other triplet selector has not been implemented yet!")
            self.criterion = loss.OnlineTripletLoss(self.margin, selector).to(self.device)
        else:
            raise NotImplementedError("Other loss function has not been implemented yet!")
        logging.info('Using {} loss function'.format(self.train_opts['loss']))

    def build_optimizer(self): 
        '''
        You can overwrite this function.
        If so, you have to set the following attribution in this function.
        1. optim
        2. lr_scheduler
        '''
        param_groups = [{'params': self.model.parameters()}, {'params': self.criterion.parameters()}]
        if self.train_opts['type'] == 'sgd':
            optim_opts = self.train_opts['sgd']
            self.optim = optim.SGD(param_groups, optim_opts['init_lr'], momentum = optim_opts['momentum'], weight_decay = optim_opts['weight_decay'])
        elif self.train_opts['type'] == 'adam':
            optim_opts = self.train_opts['adam']
            self.optim = optim.Adam(param_groups, optim_opts['init_lr'], weight_decay = optim_opts['weight_decay'])
        elif self.train_opts['type'] == 'adamw':
            optim_opts = self.train_opts['adamw']
            self.optim = optim.AdamW(param_groups, optim_opts['init_lr'], betas = (optim_opts['beta1'], optim_opts['beta2']), weight_decay = optim_opts['weight_decay'])
        logging.info("Using {} optimizer".format(self.train_opts['type']))

        # you can define your scheduler according to your preference
        if self.train_opts['lr_scheduler'] == 'step':
            self.lr_scheduler = lr_scheduler.StepLR(self.optim, step_size = self.train_opts['step_size'], gamma = self.train_opts['lr_decay'])
        elif self.train_opts['lr_scheduler'] == 'multi_step':
            self.lr_scheduler = lr_scheduler.MultiStepLR(self.optim, milestones = self.train_opts['milestones'], gamma = self.train_opts['lr_decay'])
        elif self.train_opts['lr_scheduler'] == 'cosineannealingwarmup':
            self.lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0 = self.train_opts['T_0'], 
                                                                         T_mult = self.train_opts['T_mult'], eta_min = 4e-8)
        elif self.train_opts['lr_scheduler'] == 'reducep':
            self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode = self.train_opts['lr_scheduler_mode'],
                                                               factor = self.train_opts['lr_decay'], patience = self.train_opts['patience'],
                                                               min_lr = self.train_opts['min_lr'], verbose = True)
        elif self.train_opts['lr_scheduler'] == 'cyclic':
            self.lr_scheduler = lr_scheduler.CyclicLR(self.optim, base_lr = self.train_opts['base_lr'], 
                                                      max_lr = self.train_opts['max_lr'], step_size_up = self.train_opts['step_size_up'],
                                                      mode = self.train_opts['mode'])
        # self.lr_scheduler = ...

    def build_dataloader(self): 
        '''
        You can overwrite this function.
        If so, you have to set the following attribution in this function.
        1. trainloader 
        '''
        train_collate_fn = self.trainset.collate_fn
        if self.train_opts['loss'] == 'TripletLoss':
            from libs.dataio.dataset import BalancedBatchSampler
            utt_per_spk = self.train_opts['utt_per_spk']
            spk_per_batch = self.train_opts['spk_per_batch']
            batch_sampler = BalancedBatchSampler(self.n_spk, self.trainset.count, spk_per_batch, utt_per_spk)
            self.trainloader = DataLoader(self.trainset, collate_fn = train_collate_fn, batch_sampler = batch_sampler, 
                                          num_workers = 16, pin_memory = True)
        else:
            batch_sampler = None
            self.trainloader = DataLoader(self.trainset, shuffle = True, collate_fn = train_collate_fn, batch_sampler = batch_sampler, 
                                          batch_size = self.train_opts['bs'] * self.device_num, num_workers = 16, pin_memory = True)
    
    def train_epoch(self): 
        # you MUST overwrite this function
        if __name__ == '__main__':
            print("train epoch")
        else:
            raise NotImplementedError("Please implement this function by yourself!")

    def _dev(self): 
        if __name__ == '__main__':
            print("")
        else:
            raise NotImplementedError("Please implement this function by yourself!")

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
