# coding: utf-8
from __future__ import print_function # Python 2/3 compatibility
from PIL import Image
import tifffile
import numpy
#import sys
import numpy as np
import cv2
import os
import pandas as pd

##----------------------------------------------------------------
#Change directories according to your goal
pathIn="../data/rawdata/"
pathOut="../data/tifvideo/"

##----------------------------------------------------------------

def extractImages(path):
	tifvid=[]
	count = 0
	vidcap = cv2.VideoCapture(path)
	success,image = vidcap.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	tifvid.append(gray)
	success = True
	while success:
		success,image = vidcap.read()
		if success:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			tifvid.append(gray)
			print('Read a new frame: ', success)
		count = count + 1
	return tifvid


if not os.path.exists(pathIn):
	print("Failed to acess the dir :"+str(pathIn	) )	

if not os.path.exists(pathOut):
	# Create a new directory because it does not exist 
	os.makedirs(pathOut)
	print("The ouput will be available in the following dir :"+str(pathOut) )


filenames = os.listdir(pathIn)
for file in filenames:
	if file.endswith(".mp4"):
		Images=extractImages(pathIn+file)
		Images=np.array(Images)
		with tifffile.TiffWriter(pathOut+file.replace(".mp4","")+".tif") as tiff:
		    for j in range(Images.shape[0]):
		        tiff.save(Images[j])
	else :
		df=pd.read_csv(pathIn+file)
		print(df)
		df.to_csv(pathOut+file)	

print("Done")
