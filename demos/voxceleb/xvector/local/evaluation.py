import os
import sys
sys.path.insert(0, "../../../")
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from eval_dataset import SpeechEvalDataset
from libs.utils.performance_eval import compute_eer
from libs.utils.utils import read_config
from libs.nnet import tdnn

class Evaluator(object): 
    def __init__(self, data_opts, model_opts, args): 
        eval_dataset = SpeechEvalDataset(data_opts)
        self.input_dim = eval_dataset[0][0].size(0)
        self.evalloader = DataLoader(eval_dataset, batch_size = 1, shuffle = False, num_workers = 8, pin_memory = True)
        self.build_model(model_opts)
        self.load_model(args)
        self.data_opts = data_opts
        self.model_opts = model_opts
        self.location = args['l']

    def build_model(self, model_opts):
        #  model_config = read_config("conf/model/{}.yaml".format(model_opts['arch']))
        model_config = model_opts
        model_config['input_dim'] = self.input_dim
        self.embedding_dim = model_config['embedding_dim']
        self.model = tdnn.XVector(model_config)

    def load_model(self, args): 
        self.device = torch.device(args['d']) 
        self.exp_dir = args['e']
        model_file_name = args['m']
        ckpt = torch.load('exp/{}/{}'.format(self.exp_dir, model_file_name), map_location = self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(self.device)

    def extract_embedding(self, feature): 
        feature = feature.to(self.device)
        if self.location == 'near': 
            _, xv = self.model.extract_embedding(feature)
        elif self.location == 'far': 
            xv, _ = self.model.extract_embedding(feature)
            #  xv = self.model(feature)
            xv = F.normalize(xv)
        return xv

    def compute_cosine_score(self): 
        y_true = []
        y_pred = []
        score_file = open('exp/{}/scores.txt'.format(self.exp_dir), 'w')
        with open('./task.txt', 'r') as f:
            for line in tqdm(f):
                line = line.rstrip()
                true_score, test_utt1, test_utt2 = line.split(' ')
                y_true.append(eval(true_score))
                utt1_feat = np.load(os.path.join('exp/{}/test_xv'.format(self.exp_dir), test_utt1.replace('.wav', '.npy')))
                utt2_feat = np.load(os.path.join('exp/{}/test_xv'.format(self.exp_dir), test_utt2.replace('.wav', '.npy')))
                score = cosine_similarity(utt1_feat.reshape(1, -1), utt2_feat.reshape(1, -1)).reshape(-1)
                y_pred.append(score)
                score_file.write(str(score[0]) + ' ' + test_utt1 + ' ' + test_utt2 + '\n')
        score_file.close()
        return y_true, y_pred

    def evaluate(self):
        self.model.eval()
        os.makedirs('exp/{}/test_xv'.format(self.exp_dir), exist_ok = True)
        with torch.no_grad():
           for feature, utt in tqdm(self.evalloader):
               utt = utt[0]
               feature = feature.to(self.device)
               xv = self.extract_embedding(feature)
               xv = xv.cpu().numpy()
               test_spk_dir = os.path.join('exp/{}/test_xv'.format(self.exp_dir), os.path.dirname(utt))
               os.makedirs(test_spk_dir, exist_ok = True)
               np.save(os.path.join(test_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), xv)
        y_true, y_pred = self.compute_cosine_score()
        eer, threshold = compute_eer(y_true, y_pred)
        print("EER       : {:.4%}".format(eer))
        print("Threshold : {:.4f}".format(threshold))

def main(): 
    parser = argparse.ArgumentParser("options for evaluation of speaker verification")
    parser.add_argument("-e", "--exp-dir", dest = 'e', type = str, help = "the name of experiment directory")
    parser.add_argument("-m", "--model", dest = 'm', type = str, help = "the evaluation model")
    parser.add_argument("-d", "--device", dest = 'd', type = str, choices = ['cuda', 'cpu'], help = "device name")
    parser.add_argument("-l", "--location", dest = 'l', type = str, choices = ['near', 'far'], help = "which to extract embedding")
    args = parser.parse_args()
    args = vars(args)
    data_config = read_config("exp/{}/conf/data.yaml".format(args['e']))
    model_config = read_config("exp/{}/conf/model.yaml".format(args['e']))
    evaluator = Evaluator(data_config, model_config, args)
    evaluator.evaluate()

if __name__ == "__main__": 
    main()
