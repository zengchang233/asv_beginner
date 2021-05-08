import logging
logging.basicConfig(level = logging.INFO, filename = 'train.log', filemode = 'w', format = "%(asctime)s [%(filename)s:%(lineno)d - %(levelname)s ] %(message)s")
import sys
sys.path.insert(0, "../../")
import math

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from libs.trainer import nnet_trainer
from libs.dataio import dataset
from libs.nnet import tdnn
from libs.components import loss
from libs.utils.config_parser import ArgParser
from libs.utils.utils import read_config
from eval_dataset import SpeechEvalDataset

class XVTrainer(nnet_trainer.NNetTrainer):
    def __init__(self, data_opts, model_opts, train_opts, args):
        super(XVTrainer, self).__init__(data_opts, model_opts, train_opts, args)

    def build_model(self):
        model_config = read_config("conf/model/{}.yaml".format(self.model_opts['arch']))
        model_config['input_dim'] = self.input_dim
        self.embedding_dim = model_config['embedding_dim']
        self.model = tdnn.XVector(model_config)
    
    def build_criterion(self):
        if self.train_opts['loss'] == 'CrossEntropy':
            self.criterion = loss.CrossEntropy(self.embedding_dim, self.n_spk).to(self.device)
        elif self.train_opts['loss'] == 'AMSoftmax':
            self.criterion = loss.AMSoftmax(self.embedding_dim, self.n_spk, self.train_opts['scale'], self.train_opts['margin']).to(self.device)
        else:
            raise NotImplementedError("Other loss function has not been implemented yet!")
        logging.info('Using {} loss function'.format(self.train_opts['loss']))

    def build_dataloader(self):
        train_collate_fn = self.trainset.collate_fn
        self.trainloader = DataLoader(self.trainset, shuffle = True, collate_fn = train_collate_fn, batch_size = self.train_opts['bs'] * self.device_num, num_workers = 16, pin_memory = True)
        self.eval_dataset = SpeechEvalDataset(self.data_opts)
        self.evalloader = DataLoader(self.eval_dataset, batch_size = 1, shuffle = False, num_workers = 8, pin_memory = True)
    
    def build_optimizer(self):
        param_groups = [{'params': self.model.parameters()}, {'params': self.criterion.parameters()}]
        if self.train_opts['type'] == 'sgd':
            optim_opts = self.train_opts['sgd']
            self.optim = optim.SGD(param_groups, optim_opts['init_lr'], momentum = optim_opts['momentum'], weight_decay = optim_opts['weight_decay'])
        elif self.train_opts['type'] == 'adam':
            optim_opts = self.train_opts['adam']
            self.optim = optim.Adam(param_groups, optim_opts['init_lr'], weight_decay = optim_opts['weight_decay'])
        logging.info("Using {} optimizer".format(self.train_opts['type']))
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode = self.train_opts['lr_scheduler_mode'], 
                                                           factor = self.train_opts['lr_decay'], patience = self.train_opts['patience'], 
                                                           min_lr = self.train_opts['min_lr'], verbose = True)

    def train_epoch(self):
        self.model.train()
        self.criterion.train()
        sum_samples, correct = 0, 0
        sum_am_loss, sum_uniform_loss = 0, 0
        min_distance = 0
        progress_bar = tqdm(self.trainloader)
        for batch_idx, (wave, targets_label) in enumerate(progress_bar):
            self.dev_check_count += 1
            self.optim.zero_grad() # zero_grad function only zero parameters which will be updated by optimizer

            wave = wave.to(self.device)
            targets_label = targets_label.to(self.device)
            #  print(wave.shape)
            output = self.model(wave) # output of 2nd fc layer
            loss, logits = self.criterion(output, targets_label)

            sum_samples += len(wave)
            _, prediction = torch.max(logits, dim = 1)
            correct += (prediction == targets_label).sum().item()
            
            loss.backward()

            backbone_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_opts['grad_clip_threshold']
            )
            logging.info("backbone grad norm={}".format(backbone_grad_norm))
            loss_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.criterion.parameters(), self.train_opts['grad_clip_threshold']
            )
            logging.info("criterion grad norm={}".format(loss_grad_norm))

            if math.isnan(backbone_grad_norm) or math.isnan(loss_grad_norm):
                logging.warning("grad norm is nan. Do not update model.")
            else: 
                self.optim.step()

            sum_am_loss += loss.item() * len(targets_label)

            progress_bar.set_description(
                    'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] AMLoss: {:.4f} Acc: {:.4f}%'.format(
                        self.current_epoch, batch_idx + 1, len(self.trainloader),
                        100. * (batch_idx + 1) / len(self.trainloader),
                        sum_am_loss / sum_samples, 100. * correct / sum_samples
                        )
                    )
            if self.dev_check_count % self.train_opts['check_interval'] == 0:
                # dev hyper-parameter adjust
                dev_loss, dev_acc = self._dev()
                self.lr_scheduler.step(dev_loss)
                self.criterion.train()
                self.model.train()
                logging.info("Epoch {} Dev loss {:.8f} Dev acc {:.4%} LR {:.8f}".format(self.current_epoch, dev_loss, dev_acc, self.optim.state_dict()['param_groups'][0]['lr']))
                if self.best_dev_loss >= dev_loss:
                    self.best_dev_loss = dev_loss
                    self.best_dev_epoch = self.current_epoch
                    logging.info("Best dev loss is {:.8f} at epoch {}".format(self.best_dev_loss, self.best_dev_epoch))
                    self.count = 0
                    self.save("best_dev_model.pth")
                else:
                    self.count += 1
                if self.count >= 8:
                    logging.info("Trigger early stop strategy at epoch {}, stop training".format(self.current_epoch))
                    return -1

        self.save()

    def _dev(self):
        parallel_model = self.model
        self.model = self.model.module
        self.model.eval()
        self.criterion.eval()
        dev_loss = 0
        dev_correct = 0
        with torch.no_grad():
            for wave, spk in self.trainset.get_dev_data():
                wave = wave.to(self.device)
                spk = spk.to(self.device) 
                output = self.model(wave) 
                am_loss, logits = self.criterion(output, spk)
                _, prediction = torch.max(logits, dim = 1)
                dev_correct += (prediction == spk).sum().item()
                dev_loss += am_loss.item()
        self.model = parallel_model
        return dev_loss / self.trainset.dev_number, dev_correct / self.trainset.dev_number

    def extract_embedding(self):
        pass

def main():
    parser = ArgParser()
    args = parser.parse_args()
    args = vars(args)
    data_config = read_config("conf/data.yaml")
    model_config = read_config("conf/model.yaml")
    train_config = read_config("conf/train.yaml")
    trainer = XVTrainer(data_config, model_config, train_config, args)
    trainer()

if __name__ == "__main__":
    main()