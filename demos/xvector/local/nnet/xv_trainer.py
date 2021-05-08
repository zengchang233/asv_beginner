import os
import logging
logging.basicConfig(level = logging.INFO, filename = 'train.log', filemode = 'w', format = "%(asctime)s [%(filename)s:%(lineno)d - %(levelname)s ] %(message)s")
import sys
sys.path.insert(0, "../../")
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from libs.trainer import nnet_trainer
from libs.dataio import dataset
from libs.nnet import tdnn
from libs.components import loss
from libs.utils.config_parser import ArgParser
from libs.utils.utils import read_config
from libs.utils.performance_eval import compute_eer

class XVTrainer(nnet_trainer.NNetTrainer):
    def __init__(self, data_opts, model_opts, train_opts, args):
        super(XVTrainer, self).__init__(data_opts, model_opts, train_opts, args)

    def build_model(self):
        model_config = read_config("conf/model/{}.yaml".format(self.model_opts['arch']))
        model_config['input_dim'] = self.input_dim
        self.embedding_dim = model_config['embedding_dim']
        self.model = tdnn.XVector(model_config)
    
    def build_criterion(self):
        super().build_criterion()

    def build_dataloader(self):
        super().build_dataloader()    
    
    def build_optimizer(self):
        super().build_optimizer()
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode = self.train_opts['lr_scheduler_mode'], 
                                                           factor = self.train_opts['lr_decay'], patience = self.train_opts['patience'], 
                                                           min_lr = self.train_opts['min_lr'], verbose = True)

    def train_epoch(self):
        self.model.train()
        self.criterion.train()
        sum_samples, sum_loss, correct = 0, 0, 0
        progress_bar = tqdm(self.trainloader)
        for batch_idx, (feature, targets_label) in enumerate(progress_bar):
            self.dev_check_count += 1
            self.optim.zero_grad() # zero_grad function only zero parameters which will be updated by optimizer

            feature = feature.to(self.device)
            targets_label = targets_label.to(self.device)
            output = self.model(feature) # output of 2nd fc layer
            loss, logits = self.criterion(output, targets_label)

            sum_samples += len(feature)
            _, prediction = torch.max(logits, dim = 1)
            correct += (prediction == targets_label).sum().item()
            
            loss.backward()

            backbone_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_opts['grad_clip_threshold']
            )
            logging.info("backbone grad norm = {}".format(backbone_grad_norm))
            loss_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.criterion.parameters(), self.train_opts['grad_clip_threshold']
            )
            logging.info("criterion grad norm = {}".format(loss_grad_norm))

            if math.isnan(backbone_grad_norm) or math.isnan(loss_grad_norm):
                logging.warning("grad norm is nan. Do not update model.")
            else: 
                self.optim.step()

            sum_loss += loss.item() * len(targets_label)

            progress_bar.set_description(
                    'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] AMLoss: {:.4f} Acc: {:.4f}%'.format(
                        self.current_epoch, batch_idx + 1, len(self.trainloader),
                        100. * (batch_idx + 1) / len(self.trainloader),
                        sum_loss / sum_samples, 100. * correct / sum_samples
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

    def extract_embedding(self, feature): 
        feature = feature.to(self.device)
        if self.train_opts['loss'] == 'CrossEntropy': 
            _, xv = self.model.extract_embedding(feature)
        else: 
            xv, _ = self.model.extract_embedding(feature)
            xv = F.normalize(xv)
        return xv

    def compute_cosine_score(self): 
        y_true = []
        y_pred = []
        with open('./task.txt', 'r') as f:
            for line in f:
                line = line.rstrip()
                true_score, test_utt1, test_utt2 = line.split(' ')
                y_true.append(eval(true_score))
                utt1_feat = np.load(os.path.join('exp/{}/test_xv'.format(self.log_time), test_utt1.replace('.wav', '.npy')))
                utt2_feat = np.load(os.path.join('exp/{}/test_xv'.format(self.log_time), test_utt2.replace('.wav', '.npy')))
                score = cosine_similarity(utt1_feat.reshape(1, -1), utt2_feat.reshape(1, -1)).reshape(-1)
                y_pred.append(score)
        return y_true, y_pred

    def test(self):
        if os.path.exists('exp/{}/best_dev_model.pth'.format(self.log_time)):
            self.load('exp/{}/best_dev_model.pth'.format(self.log_time))
        self.model.eval()
        os.makedirs('exp/{}/test_xv'.format(self.log_time), exist_ok = True)
        with torch.no_grad():
            for feature, utt in self.evalloader:
                utt = utt[0]
                xv = self.extract_embedding(feature)
                # feature = feature.to(self.device)
                # if self.train_opts['loss'] == 'CrossEntropy':
                #     _, xv = self.model.extract_embedding(feature) # cross entropy loss use 1st fc layer output as speaker embedding
                # # elif self.train_opts['loss'] == 'LMCL' or self.train_opts['loss'] == 'LMCL_Uniform':
                # else:
                #     xv, _ = self.model.extract_embedding(feature) # lmcl use 2nd fc layer output as speaker embedding
                #     xv = F.normalize(xv)
                xv = xv.cpu().numpy()
                test_spk_dir = os.path.join('exp/{}/test_xv'.format(self.log_time), os.path.dirname(utt))
                os.makedirs(test_spk_dir, exist_ok = True)
                np.save(os.path.join(test_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), xv)
        y_true, y_pred = self.compute_cosine_score()
        eer, threshold = compute_eer(y_true, y_pred)
        print("EER: {:.4%}".format(eer))

def main():
    parser = ArgParser()
    args = parser.parse_args()
    args = vars(args)
    data_config = read_config("conf/data.yaml")
    model_config = read_config("conf/model.yaml")
    train_config = read_config("conf/train.yaml")
    trainer = XVTrainer(data_config, model_config, train_config, args)
    # trainer()


if __name__ == "__main__":
    main()