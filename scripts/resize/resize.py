import PIL
import os
import os.path
from PIL import Image                   

path = r"/home/jrobador//Downloads/datasets/soybean/bugdetection-classification/test-train224/train/Healthy"
dirs = os.listdir( path )                                       


for file in dirs: 
    f_img = path+"/"+file
    img = Image.open(f_img)
    img = img.resize((224, 224)) #(width, height)
    img.save(f_img)