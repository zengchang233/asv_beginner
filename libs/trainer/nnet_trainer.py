import sys
import time
import logging
import yaml
import os
sys.path.insert(0, "../../")

import torch
import libs.dataio.dataset as dataset
from libs.utils.utils import read_config
from libs.utils.config_parser import ArgParser

class NNetTrainer(object):
    def __init__(self, data_opts = None, model_opts = None, train_opts = None, args = None):
        # set configuration according to options parsed by argparse module
        # print(args)
        self.mode = args['mode']
        data_opts['feat_type'] = args['feat_type']
        model_opts['arch'] = args["arch"]
        # model_opts['input_dim'] = args['input_dim']
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

        # build model
        self.build_model()
        # if self.model_opts['arch'] == 'tdnn' or self.model_opts['arch'] == 'etdnn':
        #     import components.models.tdnn as tdnn
        #     model = tdnn.SpeakerEmbNet(self.model_opts)
        # elif self.model_opts['arch'] == 'resnet':
        #     import components.models.resnet as resnet
        #     model = resnet.SpeakerEmbNet(self.model_opts)
        # else:
        #     raise NotImplementedError("Other models are not implemented!")
        # logging.info("Using {} neural network".format(self.model_opts['arch']))
        # self.embedding_dim = self.model_opts[self.model_opts['arch']]['embedding_dim']
        # logging.info("Dimension of speaker embedding is {}".format(self.embedding_dim))

        # resume model from saved path
        if os.path.exists(self.train_opts['resume']):
            self.load(self.train_opts['resume'])

        # mv to device
        self._move_to_device()
        
        # build dataloader
        self.build_dataloader()
        # train_collate_fn = self.trainset.collate_fn
        # self.trainloader = DataLoader(self.trainset, shuffle = True, collate_fn = train_collate_fn, batch_size = self.train_opts['bs'] * device_num, num_workers = 32, pin_memory = True)
        # self.voxtestloader = DataLoader(self.voxtestset, batch_size = 1, shuffle = False, num_workers = 8, pin_memory = True)

        # build loss
        self.build_criterion()
        # if self.train_opts['loss'] == 'CrossEntropy':
        #     self.criterion = CrossEntropy(self.embedding_dim, n_spk).to(self.device)
        # elif self.train_opts['loss'] == 'LMCL':
        #     margin_range = self.train_opts['margin']
        #     self.init_margin = margin_range[0]
        #     self.end_margin = margin_range[1]
        #     if self.model_opts['arch'] in ['tdnn', 'etdnn']:
        #         self.criterion = LMCL(self.embedding_dim, n_spk, self.train_opts['scale'], self.init_margin).to(self.device)
        #     elif self.model_opts['arch'] == 'resnet':
        #         self.criterion = LMCL(self.embedding_dim, n_spk, self.train_opts['scale'], self.end_margin).to(self.device)
        # elif self.train_opts['loss'] == 'LMCL_Uniform':
        #     margin_range = self.train_opts['margin']
        #     self.init_margin = margin_range[0]
        #     self.end_margin = margin_range[1]
        #     self.criterion = LMCLUniformLoss(self.embedding_dim, n_spk, self.train_opts['scale'], self.end_margin).to(self.device)
        # else:
        #     raise NotImplementedError("Other loss function has not been implemented yet!")
        # logging.info('Using {} loss function'.format(self.train_opts['loss']))

        # build optimizer
        self.build_optimizer()
        # param_groups = [{'params': self.model.parameters()}, {'params': self.criterion.parameters()}]
        # if self.train_opts['type'] == 'sgd':
        #     optim_opts = self.train_opts['sgd']
        #     self.optim = optim.SGD(param_groups, optim_opts['init_lr'], momentum = optim_opts['momentum'], weight_decay = optim_opts['weight_decay'])
        # elif self.train_opts['type'] == 'adam':
        #     optim_opts = self.train_opts['adam']
        #     self.optim = optim.Adam(param_groups, optim_opts['init_lr'], weight_decay = optim_opts['weight_decay'])
        # logging.info("Using {} optimizer".format(self.train_opts['type']))
        # self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode = self.train_opts['lr_scheduler_mode'], 
        #                                                    factor = self.train_opts['lr_decay'], patience = self.train_opts['patience'], 
        #                                                    min_lr = self.train_opts['min_lr'], verbose = True)

    def build_model(self):
        raise NotImplementedError("Please implement this function by yourself!")

    def build_criterion(self):
        raise NotImplementedError("Please implement this function by yourself!")

    def build_optimizer(self): 
        raise NotImplementedError("Please implement this function by yourself!")

    def build_dataloader(self): 
        raise NotImplementedError("Please implement this function by yourself!")
    
    def train_epoch(self): 
        raise NotImplementedError("Please implement this function by yourself!")

    def _move_to_device(self):
        if self.train_opts['device'] == 'gpu':
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
            torch.save({'epoch': self.current_epoch, 'state_dict': model.state_dict(), 'criterion': self.criterion,
                        'lr_scheduler': self.lr_scheduler.state_dict(), 'optimizer': self.optim.state_dict()},
                        'exp/{}/net_{}.pth'.format(self.log_time, self.current_epoch))
        else:
            torch.save({'epoch': self.current_epoch, 'state_dict': model.state_dict(), 'criterion': self.criterion,
                        'lr_scheduler': self.lr_scheduler.state_dict(), 'optimizer': self.optim.state_dict()},
                        'exp/{}/{}'.format(self.log_time, filename))

    def load(self, resume):
        ckpt = torch.load(resume)
        self.model.load_state_dict(ckpt['state_dict'])
        if 'criterion' in ckpt:
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

    def __call__(self):
        os.makedirs('exp/{}'.format(self.log_time), exist_ok = True)
        
        if not os.path.exists("exp/{}/config.yaml".format(self.log_time)):
            with open("exp/{}/config.yaml".format(self.log_time), 'w') as f:
                yaml.dump(self.data_opts, f)
                yaml.dump(self.model_opts, f)
                yaml.dump(self.train_opts, f)
        # train, test 
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