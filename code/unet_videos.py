from baseline_aug import get_unet, input_shape
from cv2 import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
import numpy as np
from tqdm import tqdm
import cv2
import os
from keras.layers import ReLU
from PIL import Image
from random import sample
from read_frame import batch,get_number_frames,sub_sample_frames
import imageio
import pandas as pd

##---------------------------------------------------------------
batchsize = 64
#change path to your directories
pathIn="../data/tifvideo/"
#pathToReconstructed="../data/reconstructed_films/"
pathToMask='../outputs/masks/'
pathToTagged='../outputs/tagged_films/'
pathToReplaced='../outputs/replaced_films/'
##----------------------------------------------------------------




if __name__ == '__main__':
	model_name = "baseline_unet_aug_do_0.0_activation_ReLU_"

	if not os.path.exists(pathToMask):
		# Create a new directory because it does not exist 
		os.makedirs(pathToMask)
		print("Masks will be available in the following dir :"+str(pathToMask) )

	if not os.path.exists(pathToReplaced):
		# Create a new directory because it does not exist 
		os.makedirs(pathToReplaced)
		print("Tagged Videos will be available in the following dir :"+str(pathToReplaced) )

	if not os.path.exists(pathToTagged):
		# Create a new directory because it does not exist 
		os.makedirs(pathToTagged)
		print("Segmented Videos will be available in the following dir :"+str(pathToTagged) )

	filenames = os.listdir(pathIn)
	model = get_unet(do=0.1, activation=ReLU)
	file_path = model_name + "weights.best.hdf5"
	model.load_weights(file_path) #loading model and printing model architecture
	
	for i in filenames:	#reading file from dir
		print(i)
		if i.endswith(".tif"):
			img = Image.open(pathIn + i)		
			list_frames = sub_sample_frames(img)	
			imgs = [resize(x[:, :, np.newaxis]/255, input_shape) for x in list_frames] #reading .tif to get desired output shape

			

			imgs = np.array(imgs)
			pred = model.predict(imgs) #predicting the segmented
			pred = np.clip(pred, 0, 1)

			#reconstructed mp4 film from tif file
			#		out = np.concatenate((imgs, imgs, imgs), axis=-1)*255
			#		out = out.astype(np.uint8)
			#		imageio.mimwrite(pathToReconstructed+ i.replace(".tif","")+"_reconstructed.mp4", out, fps=30)

			pred = (pred>0.5)

			#writing mask into depository
			imageio.mimwrite(pathToMask+ i.replace(".tif","")+"_mask.mp4", img_as_ubyte(pred), fps=30)

			#image where mask value are 1
			img_modified = np.array(imgs)
			img_modified[pred] = 1

			out_tagged = (np.concatenate((imgs, imgs, img_modified), axis=-1)*255)
			out_tagged = out_tagged.astype(np.uint8)

			#writing mask into depository
			imageio.mimwrite(pathToTagged+ i.replace(".tif","")+"_tagged.mp4", out_tagged, fps=30)

			#image where backgrounnd = 0 and pixel are those of the image on the mask area
			img_replaced = np.array(imgs)
			img_replaced[np.logical_not(pred)] = 0

			out_replaced = (np.concatenate((img_replaced, img_replaced, img_replaced), axis=-1)*255)
			out_replaced = out_replaced.astype(np.uint8)


			imageio.mimwrite(pathToReplaced+ i.replace(".tif","")+"_replaced.mp4", out_replaced, fps=30)
			
		else :
			df=pd.read_csv(pathIn+i)
			[df.to_csv(path+i,index_col=False) for path in [pathToMask , pathToTagged , pathToReplaced ] ]
		break
	print("Done")

