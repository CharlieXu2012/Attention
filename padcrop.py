import numpy as np
import scipy as sp
import os
import scipy.ndimage, scipy.misc
import cv2
""" 
Crop and pad depth images to 1920x1080.

NOT INCLUDED IN MAIN PIPELINE!!!
"""

# output folder
outpath = "" 

# input folder
path = "" 

imagelist = os.listdir(path)

for i in range(0, np.size(imagelist,0)):

    frame = scipy.misc.imread(path + imagelist[i],mode='L') # load image
    frame = scipy.misc.imresize(frame, 300) # resize image by 3.0 factor
    frame = np.delete(frame, range(0,96), axis=0) # crop top and bottom rows , frame[1080:frame.shape[0]-1,:]], 
    frame = np.delete(frame, range(1080,frame.shape[0]), axis=0)
    pad = np.zeros([1080,192])
    frame = np.concatenate((pad,frame),axis=1) # pad left and right
    frame = np.concatenate((frame,pad),axis=1) # pad left and right
    scipy.misc.imsave(outpath + imagelist[i],frame.astype('uint8')) # save image to output folder 