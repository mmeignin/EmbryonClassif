from LitClassificationModel import LitClassificationModel
from csvflowdatamodule.CsvDataset import CsvDataModule
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch
from argparse import ArgumentParser
from pathlib import Path
from Callbacks import ResultsLogger
import json

checkpoint = "results/EmbryonClassif_on_serv/306otnua/checkpoints/last.ckpt"
pathtosummary=os.getcwd()+"/results/wandb/run-20220901_175828-306otnua/files/wandb-metadata.json"


with open(pathtosummary, 'r') as f:
  data = json.load(f)
arglist={}
for i in range(len(data['args'])-1):
    if "-" in data['args'][i]:
        arglist[data['args'][i]] = data['args'][i+1]

model = LitClassificationModel(backbone=arglist['-bb'],head=arglist['-he'],criterion_name=arglist['--criterion_name'],framestep=int(arglist['--framestep']),embedding='False',img_size=int(256),NBClass=8)

model = LitClassificationModel.load_from_checkpoint(checkpoint,backbone=arglist['-bb'],head=arglist['-he'],criterion_name=arglist['--criterion_name'],framestep=int(arglist['--framestep']),embedding='False',img_size=int(256),NBClass=8)
dm_test = CsvDataModule(request=['Image', 'Class','t0'], batch_size = int(1),framestep = int(arglist['--framestep']) ,subsample_train = int(1), img_size = [int(256),int(256)] ,base_dir = os.environ['PWD'] ,data_file =arglist['--data_file'],data_path='NewDataSplit/'+arglist['--data_file']+'_test.csv',augmentation='')
print(model)
callback=ResultsLogger(filepath="results/EmbryonClassif_on_serv/306otnua/results.csv")


trainer = pl.Trainer(callbacks=callback)
trainer.test(model,dm_test)
# ------------
# logger and callbacks
# --------
