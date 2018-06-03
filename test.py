import numpy as np
import scipy as sp
import cv2 as cv

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

x, y = [600,600]
point = np.array([[[x-1, y-1], [x, y-1], [x+1, y-1]], 
                     [[x-1, y], [x, y], [x+1, y]],
                     [[x-1, y+1], [x, y+1], [x+1, y+1]]])

frame_n = 'E:\PANDORA_DEPTH_CROPPED\PAN01_cropped\PAN01_0001.png'
frame = cv.imread(frame_n)
point_i = np.array([[[frame[point[0,0,0],point[0,0,1]][0],frame[point[0,1,0],point[0,1,1]][0],frame[point[0,2,0],point[0,2,1]]][0]],
                    [[frame[point[1,0,0],point[1,0,1]][0],frame[point[1,1,0],point[1,1,1]][0],frame[point[1,2,0],point[1,2,1]]][0]],
                    [[frame[point[2,0,0],point[2,0,1]][0],frame[point[2,1,0],point[2,1,1]][0],frame[point[2,2,0],point[2,2,1]]][0]]])

depth_k = np.mean(point_i).astype(int)

minc = np.min(depth_k)
maxc = np.max(depth_k)
depth_k[:,i] = (depth_k - minc) / (maxc - minc)