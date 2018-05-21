import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2 as cv
import os
from random import randint
import shutil

import img_augmentation as aug
import utils
from utils import ImageProcessing


images=[]
positions=[]
#load path
path='/home/oumaima/Pictures/'

# create a directory to hold training images
utils.create_dir("train_path1")

# create a directory to hold label images
utils.create_dir("label_path")

# load images
for pic in os.listdir(path):

   #read image
   img = cv.imread(path+pic)

   img_process = ImageProcessing(img)

   # Create a white image
   img_white = img_process.white_img(img)

   # get text caracteristics
   font, fontScale, fontColor, lineType = utils.text_caracteristics(img)
   # number of lines
   nb_lines = randint(1, 4)

    # name of image without the extension
   img_name = os.path.splitext(pic)[0]
    #write texte
   img, pt1, bt, bottom, end = img_process.draw_text(pic,img,img_white,nb_lines,font,fontScale,fontColor,lineType)

   images.append([img_name,img])
   #text region
   positions.append([pt1, bt, bottom, end])

#create a directory to hold training images
utils.create_dir("train_path")
shutil.rmtree('train_path1')

#instance of Augmentor
augmentor = aug.Augmentor()

#get augmented images
augmented_images=  augmentor.img_aug(images,positions)

for each_img in augmented_images:
    img_label=each_img[0]
    picture=each_img[1]
    # save image
    cv.imwrite("train_path/" + img_label + ".jpg", picture)


