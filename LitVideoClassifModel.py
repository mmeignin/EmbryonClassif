import pytorch_lightning as pl
from LitVideo import LitVideo
import torch.nn as nn
import torch
from ipdb import set_trace
from argparse import ArgumentParser
import pytorchvideo

class LitVideoClassifModel(pl.LightningModule) :
    def __init__(self, criterion_name, **kwargs) :
        super().__init__()
        self.video_model = LitVideo(**kwargs)
        self.criterion_name = criterion_name

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        print(f'Optimizer : {optimizer}')
        return optimizer

    def prediction(self, batch) :
        """
        Produce a prediction for a segmentation using the model and the batch
        """
        batch['Pred'] = self.video_model.forward(batch)
        return batch

    def Criterion(self, batch) :
        """"
        Comput loss for the batch
        """
        if self.criterion_name == 'bce' :
            losses = nn.functional.cross_entropy(batch['Pred'], batch['Class'], reduction='none')
        if self.criterion_name == 'bce_balanced' :
            class_weight = torch.tensor([1.2500, 1.1842, 1.1250, 1.6071, 1.1250, 0.8654, 0.9000, 0.5921]).to(batch['Class'].device)
            losses = nn.functional.cross_entropy(batch['Pred'], batch['Class'], reduction='none', weight=class_weight)
        return {'losses' : losses}

    def Evaluations(self, batch, evals) :
        """
        Run evaluations of the binary masks and add the key to evals
        Args : batch with at least keys
                    'Pred' (b, L, I, J) : Mask proba predictions with sofmax over dim=1
                    'GtMask' binary (b, I, J) : ground truth foreground mask
               evals with a least keys
                    'losses' (b) : Loss per image
        """
        evals['loss'] = evals['losses'].mean() # Loss is necessary for backprop in Pytorch Lightning
        a = batch['Pred'].argmax(axis=-1) #(Nvideos, 1)
        evals['preds'] =  a
        evals['accs'] = (a == batch['Class']).to(torch.float)
        evals['Class'] =  batch['Class']
        for i in range(8) :
            evals[f'preds_{i}'] = (a == i).to(torch.float)


    def Logging(self, evals, step_label) :
        bs = evals['losses'].shape[0]
        log = lambda k : self.log(f'{step_label}/{k}', evals[k].mean(), on_epoch=True, on_step=False)

        for k in ['losses', 'accs'] + [f'preds_{i}' for i in range(8)]:
            evals[k] = evals[k].detach()
            log(k)

    def step(self, batch, step_label):
        batch = self.prediction(batch)
        evals = self.Criterion(batch)
        self.Evaluations(batch, evals)
        self.Logging(evals, step_label)
        return evals, batch

    def training_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'train')
        return evals

    def validation_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'val')
        return evals

    def test_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'test')
        return evals

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LitVideo.add_specific_args(parser)
        parser.add_argument('--criterion_name', type=str, choices=['bce', 'bce_balanced'], default='bce_balanced')
        return parser
