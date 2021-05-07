import argparse

import yaml
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def read_config(file_path): 
    f = open(file_path, 'r')
    config = yaml.load(f, Loader = yaml.CLoader)
    f.close()
    return config

class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description = "specify some important arguments")
        parser.add_argument("--data-opts", type = str, default = "data", dest = "data_opts", help = "data config file")
        parser.add_argument("--model-opts", type = str, default = "model", dest = "model_opts", help = "model config file")
        parser.add_argument("--train-opts", type = str, default = "train", dest = "train_opts", help = "train config file")
        parser.add_argument("--feat-type", type = str, default = 'python_mfcc', dest = "feat_type", help = 'input feature')
        parser.add_argument("--input-dim", type = int, default = 30, dest = "input_dim", help = "dimension of input feature")
        parser.add_argument("--arch", type = str, default = "tdnn", choices = ["resnet", "tdnn", "etdnn", "ftdnn", "rawnet", "wav2spk"], help = "specify model architecture")
        parser.add_argument("--loss", type = str, default = "AMSoftmax", choices = ["AMSoftmax", "CrossEntropy", "ASoftmax", "TripletLoss"], help = "specify loss function")
        parser.add_argument("--bs", type = int, default = 64, help = "specify batch size for training")
        parser.add_argument("--collate", type = str, default = "length_varied", choices = ["kaldi", "length_varied"], help = "specify collate function in DataLoader")
        parser.add_argument("--resume", type = str, help = "if you give a ckpt path to this argument and if the ckpt file exists, it will resume training based on this ckpt file. Otherwise, it will start a new training process")
        parser.add_argument("--device", default = 'gpu', choices = ['gpu', 'cpu'], help = 'designate the device on which the model will run')
        self.parser = parser

    def add_argument(self, option, arg_type, default, dest, help):
        self.parser.add_argument(option, type = arg_type, default = default, dest = dest, help = help)

    def parse_args(self):
        args = self.parser.parse_args()
        return args

def compute_eer(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

if __name__ == "__main__":
    arg_parser = ArgParser()
    arg_parser.add_argument('--epochs', int, 40, 'epochs', "the number of training epochs")
    args = arg_parser.parse_args()
    print(args)