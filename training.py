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

parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser = LitClassificationModel.add_specific_args(parser)
parser = CsvDataModule.add_specific_args(parser)
args = parser.parse_args()
args.gpus = -1 if torch.cuda.is_available() else 0

model = LitClassificationModel(**vars(args))

args.data_path  = f'NewDataSplit/{args.data_file}_'+'{}.csv'
args.base_dir=os.environ['PWD']
dm = CsvDataModule(request=['Image', 'Class','t0'], **vars(args))

# ------------
# logger and callbacks
# ------------


args.save_dir = os.path.join(os.environ['PWD'], 'results/')
args.logger = WandbLogger(project='EmbryonClassif_on_serv',
                          save_dir=args.save_dir,
                          log_model=False)
model.hparams.project_name = args.logger.experiment.project_name()
model.hparams.experiment_id =  args.logger.experiment.id
model.hparams.experiment_name =  args.logger.experiment.name
wandb.run.log_code(".")


# ------------
# log model
# ------------
path_save_model = os.path.join(args.save_dir, args.logger.name, args.logger.experiment.id)
Path(path_save_model+'/checkpoints/').mkdir(parents=True, exist_ok=True)
mck = pl.callbacks.ModelCheckpoint(path_save_model+'/checkpoints/',
                                   monitor='val/losses',
                                   filename='epoch_{epoch}_val_loss_{val/losses:.3f}',
                                   save_last=True,
                                   mode='min',
                                   save_top_k=3)

args.callbacks = [ResultsLogger(), mck]
args.max_epochs = 120
args.accumulate_grad_batches=16
trainer = pl.Trainer.from_argparse_args(args)
trainer.logger.log_hyperparams(args)

trainer.fit(model, dm)
trainer.test(model,dm)


