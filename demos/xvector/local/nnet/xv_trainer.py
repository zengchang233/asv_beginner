from asv_beginner.libs.trainer import nnet_trainer

class XVTrainer(nnet_trainer.NNetTrainer):
    def __init__(self):
        super(XVTrainer, self).__init__()

    def train_epoch(self):
        pass

    def _dev(self):
        pass

    def extract_embedding(self):
        pass
