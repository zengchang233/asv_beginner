import torch
import torch.nn.functional as F
import torch.optim as optim

class BackendTrainer(object):
    def __init__(self):
        pass

    def save(self, filepath = None):
        pass

    def load(self):
        pass

    def train(self):
        pass

    def train_epoch(self):
        pass

    def _dev(self):
        pass

    def _move_to_device(self):
        pass

if __name__ == "__main__":
    trainer = BackendTrainer()
    trainer.train()