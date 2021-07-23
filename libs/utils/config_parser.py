import argparse

class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description = "specify some important arguments")
        #  parser.add_argument("--feat-type", type = str, default = 'python_mfcc', dest = "feat_type", help = 'input feature')
        #  parser.add_argument("--input-dim", type = int, default = 30, dest = "input_dim", help = "dimension of input feature")
        #  parser.add_argument("--arch", type = str, default = "tdnn", choices = ["resnet", "tdnn", "etdnn", "ftdnn", "rawnet", "wav2spk", "wavenet"], help = "specify model architecture")
        #  parser.add_argument("--loss", type = str, default = "AMSoftmax", choices = ["AMSoftmax", "CrossEntropy", "ASoftmax", "TripletLoss"], help = "specify loss function")
        #  parser.add_argument("--bs", type = int, default = 64, help = "specify batch size for training")
        #  # parser.add_argument("--collate", type = str, default = "length_varied", choices = ["kaldi", "length_varied"], help = "specify collate function in DataLoader")
        #  parser.add_argument("--resume", type = str, default = 'none', help = "if you give a ckpt path to this argument and if the ckpt file exists, it will resume training based on this ckpt file. Otherwise, it will start a new training process")
        #  parser.add_argument("--device", default = 'gpu', choices = ['cuda', 'cpu'], help = 'designate the device on which the model will run')
        #  parser.add_argument("--mode", default = 'train', choices = ['train', 'test'], help = 'train or test mode')
        #  self.parser = parser

    def add_argument(self, option, arg_type, default, dest, help):
        self.parser.add_argument(option, type = arg_type, default = default, dest = dest, help = help)

    def parse_args(self):
        args = self.parser.parse_args()
        return args

if __name__ == "__main__":
    arg_parser = ArgParser()
    arg_parser.add_argument('--epochs', int, 40, 'epochs', "the number of training epochs")
    args = arg_parser.parse_args()
    print(args)
