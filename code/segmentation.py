import pandas as pd
import skvideo.io,imageio 
from skimage.measure import label, regionprops
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from read_frame import sub_sample_frames
from skimage.transform import resize
from baseline_aug import get_unet, input_shape

##---------------------------------------------------------------
pathtovid="../data/tifvideo/"
pathtofmask="../outputs/filled_masks/"
pathOut="../results/segmented_videos/"
##---------------------------------------------------------------



if not os.path.exists(pathOut):
		# Create a new directory because it does not exist 
		os.makedirs(pathOut)
		print("Segmented Videos will be available in the following dir :"+str(pathOut) )

df=pd.DataFrame()
comments=pd.DataFrame(columns=["video","frame"])
fn = os.listdir(pathtovid)

for file in fn:
	print(file)
	if file.endswith(".tif"):
		img = Image.open(pathtovid+file)
		list_frames = sub_sample_frames(img)
		imgs = [resize(x[:, :, np.newaxis]/255, input_shape) for x in list_frames]
		imgs=np.array(imgs)
		imgs=imgs[:,:,:,0]

		fmask  = skvideo.io.vread(pathtofmask+file.replace(".tif","")+"_mask_filled.mp4",as_grey=True)
		fmask = np.array(fmask)
		fmask = fmask[:,:,:,0]

		R=np.zeros(imgs.shape)
		radius=[]
		b=0
		res= True
		

		# label image regions
		label_image = label(fmask[-1])
		props = regionprops(label_image)
		area=[o.area for o in props]
		imax=np.argmax(np.array(area))
		r=round(props[imax].axis_major_length /2)
		print(r)#maximum 

		for p in range(imgs.shape[0]):
			#label image regions
			label_image = label(fmask[p])
			props = regionprops(label_image)

			area=[o.area for o in props]
			imax=np.argmax(np.array(area))

			x0,y0 = props[imax].centroid
			b =  props[imax].axis_major_length /2
			radius.append(b)
			x0,y0 = round(x0),round(y0)
			for i in range(-r,r):
				for j in range(-r,r):
					if (x0+i)<255 and (x0+j)>0 and (y0+j)<255 and (y0+j)>0 :
						if((i)*(i)+(j)*(j)<=(r*r)):#circle equation
							R[p,x0+i,y0+j]=imgs[p,x0+i,y0+j]
		
		#if there has been some problems with masks segmentation we try to segment using hough circles on original frames
		if np.std(radius)>b/40 or r>80:
			r=75
			R=np.zeros(imgs.shape)
			for k in range(imgs.shape[0]):
				img= imgs[k] + round(np.mean(imgs[k]))
				img =(img*255).astype(np.uint8)
				img = cv2.GaussianBlur(img,(5,5),0)
				circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp=2,minDist=1,param1=85,param2=40, minRadius=65,maxRadius=74)
				if circles is None :
					com= pd.DataFrame([(file.replace(".tif","","frame: "+str(k))],columns=["video","frame"])
					comments = pd.concat([comments,com])
					res=False
					print("Cannot segment following file:"+file)	
					break
				circles = circles[0,:,:]
				#print(circles.shape[0])
				CM=np.mean(circles[:,0:2],axis=0)
				x0,y0 = CM[0],CM[1]
				x0,y0 = round(x0),round(y0)
				for i in range(-r,r):
					for j in range(-r,r):
						if (y0+i)<255 and (y0+i)>=0 and (x0+j)<255 and (x0+j)>=0 :
							if((i)*(i)+(j)*(j)<=(r*r)):#circle equation
								R[k,y0+i,x0+j]=imgs[k,y0+i,x0+j]
		temp_df=pd.DataFrame(radius,columns={file.replace(".tif","")})
		df=pd.concat([df,temp_df],axis=1)
		R=(R*255).astype(np.uint8)
		if res == True :
			imageio.mimwrite(pathOut+file.replace(".tif","") +".mp4", R, fps=30)
	else:
		dataframe=pd.read_csv(pathtovid+file,index_col=False)
	break
comments=pd.merge(dataframe,comments,how="right",left_on ="nam",right_on ="video")
comments=comments[{"video","frame","class","t0"}]
comments.to_csv(pathOut+"failed.csv", sep=',',index=False)
dataframe.to_csv(pathOut+file)
print("Done")

