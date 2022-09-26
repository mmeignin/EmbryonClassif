# Segmentation for EmbryonClassif
Instruction for Segmentation.
You got to specify the data directory first in mp4_To_tif.py and then you can freely name the dir :
* #to create tif objects in order to do U-net mask prediction
- python3 mp4_To_tif.py 
* #to do unet prediction
- python3 unet_videos.py
* #to fill the unet mask
- python3 fill_mask.py
* #To finally do the segmentaiton
- python3 segmentation.py
 
