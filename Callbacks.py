import pytorch_lightning as pl
import flowiz
import torch
import wandb
from ipdb import set_trace
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
#from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResultsLogger(pl.Callback) :
    def __init__(self, keys=['losses', 'accs', 'preds', 'Class'], filepath=None):
        super().__init__()
        self.fp = filepath
        self.keys = keys

    def setup(self, trainer, pl_module, stage) :
        if self.fp is None :
            self.fp = os.path.join(trainer.log_dir, trainer.logger.name, trainer.logger.experiment.id, 'results.csv')
        print(f'Save results in {self.fp}')
        with open(self.fp, 'w') as f :
            f.write(f'epoch,step_label,file_name,'+','.join(self.keys)+'\n')

    @torch.no_grad()
    def write_results(self, imps, outputs, epoch, step_label) :
        with open(self.fp, 'a') as f :
            for i, imn in enumerate(imps) :
                f.write(f'{epoch},{step_label},{imn.strip(os.environ["PWD"])},'+','.join([f'{outputs[j][i].item():.3f}' for j in self.keys if j in outputs.keys()])+'\n')

    def batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, step_label):
        key_path = 'VideoName'
        self.write_results(batch[key_path], outputs, trainer.current_epoch, step_label)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'test')

    def on_test_end(self, trainer, pl_module) :
        self.summary_path = self.fp.replace('.csv', '_summary.tsv')
        dfr = pd.read_csv(self.fp)
        dfr.groupby(['step_label','epoch']).mean()[['accs','losses']].to_csv(self.summary_path.replace("_.tsv",".csv"), sep='\t')
        print(f'Summary saved at : {self.summary_path}')
