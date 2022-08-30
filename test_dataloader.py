from csvflowdatamodule.CsvDataset import CsvDataModule
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from visualisation import displayvideo

#class_names=['A','B','C','D','E','F','G','H'])})

hflipper = transforms.RandomHorizontalFlip(p=1)
vflipper = transforms.RandomVerticalFlip(p=1)

dm = CsvDataModule(data_path='DataSplit/EmbryonBinaryRaw_RandomSplit_{}.csv', base_dir=os.environ['PWD'], batch_size=1, request=['Image', 'Class','t0'], img_size=[256,256], augmentation=['randombrightness'], framestep=1)

dm.setup('fit')

dm.dtrain.framestep
dtrain = dm.train_dataloader()

idtrain = iter(dtrain)

ret = idtrain.next()

print(f"Video shape: {ret['Video'].shape}") # VideoName,frames,channels,w,h
#print(f"Video Name: {ret['VideoName']}",f"Number of Frames in Video {ret['Video'].shape[1]}")
#print(f"max: {torch.max(ret['Video'][0,:])}" , f"min: {torch.min(ret['Video'][0,:])}",f"mean: {torch.mean(ret['Video'][0,:])}")

print(ret)
fig1=displayvideo(ret,5)

plt.show()