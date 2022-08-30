from csvflowdatamodule.CsvDataset import CsvDataModule
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from visualisation import displayvideo
<<<<<<< HEAD
#class_names=['A','B','C','D','E','F','G','H'])})

hflipper = transforms.RandomHorizontalFlip(p=1)
vflipper = transforms.RandomVerticalFlip(p=1)

dm = CsvDataModule(data_path='DataSplit/EmbryonBinaryRaw_RandomSplit_{}.csv', base_dir=os.environ['PWD'], batch_size=1, request=['Image', 'Class','t0'], img_size=[256,256], augmentation=['randombrightness'], framestep=1)
=======

hflipper = transforms.RandomHorizontalFlip(p=0.5)
vflipper = transforms.RandomVerticalFlip(p=0.5)

dm = CsvDataModule(data_path='DataSplit/Embryon_RandomSplit_{}.csv', base_dir=os.environ['PWD'], batch_size=1, request=['Image', 'Class'], img_size=[256,256], augmentation='', framestep=1)
>>>>>>> c5b426a3415e5d68823f62d2649f8da31a66a0e4

dm.setup('fit')

dm.dtrain.framestep
dtrain = dm.train_dataloader()

idtrain = iter(dtrain)

ret = idtrain.next()

<<<<<<< HEAD
print(f"Video shape: {ret['Video'].shape}") # VideoName,frames,channels,w,h
#print(f"Video Name: {ret['VideoName']}",f"Number of Frames in Video {ret['Video'].shape[1]}")


#print(f"max: {torch.max(ret['Video'][0,:])}" , f"min: {torch.min(ret['Video'][0,:])}",f"mean: {torch.mean(ret['Video'][0,:])}")

print(ret)
fig1=displayvideo(ret,5)
=======


print(ret['Video'].shape) # VideoName,frames,channels,w,h
print(ret['VideoName'], ret['Video'].shape[1])
fig1=displayvideo(ret,5)





#ret['Video']=hflip(ret['Video'])
ret['Video']= transforms.functional.vflip(ret['Video'])
fig2=displayvideo(ret,5)



>>>>>>> c5b426a3415e5d68823f62d2649f8da31a66a0e4
plt.show()