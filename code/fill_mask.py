import os,cv2
from PIL import Image
import numpy as np
from skimage import data, util
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_fill_holes
import math
import skvideo.io,imageio
import pandas as pd
##--------------------------------------------------------
#change path to ur directories

pathtomask="../outputs/masks/"
pathOut = '../outputs/filled_masks/'
number_frame=300
img_shape=(256,256)

##--------------------------------------------------------
if not os.path.exists(pathOut):
		# Create a new directory because it does not exist 
		os.makedirs(pathOut)
		print("Filled Masks will be available in the following dir :"+str(pathOut) )
filenames = os.listdir(pathtomask)
for i in filenames :
	if i.endswith(".mp4"):
		mask  = skvideo.io.vread(pathtomask+i,as_grey=True) #loading masks
		mask = np.array(mask).reshape(number_frame,img_shape[0],img_shape[1])
		otsu_threshold, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY )#thresolding to smooth contours
		mask[np.where(mask==255)]=1 #transform to binary image
		mask = binary_fill_holes(mask,structure=np.ones((1,5,5))) #fill_holes using structuring element
		mask=mask*255
		mask = mask.astype(np.uint8)
		imageio.mimwrite( pathOut+i.replace(".mp4","") +"_filled.mp4", mask, fps=30)
	else :
		df=pd.read_csv(pathtomask+i,index_col=False)
		df.to_csv(pathOut+i) 
		
print("Done")

