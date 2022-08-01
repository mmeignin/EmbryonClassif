from csvflowdatamodule.CsvDataset import CsvDataModule
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from visualisation import displayvideo

hflipper = transforms.RandomHorizontalFlip(p=0.5)
vflipper = transforms.RandomVerticalFlip(p=0.5)

dm = CsvDataModule(data_path='DataSplit/Embryon_RandomSplit_{}.csv', base_dir=os.environ['PWD'], batch_size=1, request=['Image', 'Class'], img_size=[256,256], augmentation='', framestep=1)

dm.setup('fit')

dm.dtrain.framestep
dtrain = dm.train_dataloader()

idtrain = iter(dtrain)

ret = idtrain.next()



print(ret['Video'].shape) # VideoName,frames,channels,w,h
print(ret['VideoName'], ret['Video'].shape[1])
fig1=displayvideo(ret,5)





#ret['Video']=hflip(ret['Video'])
ret['Video']= transforms.functional.vflip(ret['Video'])
fig2=displayvideo(ret,5)



plt.show()