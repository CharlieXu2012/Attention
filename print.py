from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import numpy as np
import scipy as sp
import os
from os.path import join, expanduser

labels = np.load(join(os.path.dirname(__file__),'models\\GRANADE_predictions2.npy'))
path = "E:\\GRANADE_out\\"
path_o = "E:\\GRANADE_out\\"
frame_files = os.listdir(path)
for idx in range(0,len(frame_files)):
    img = Image.open(path+frame_files[idx])
    draw = ImageDraw.Draw(img)
    c = labels[idx].astype(str)
    #font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype('arial.ttf', 40)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((1180, 50),c,(255,255,255),font=font)
    img.save(path_o + frame_files[idx])