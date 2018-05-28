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
transparents=[]
#load path
path='/home/oumaima/Pictures/'

resolution=(194, 259)

# create a directory to hold training images
utils.create_dir("train_path1")
# create a directory to hold label images
utils.create_dir("label_path")

# load images
for pic in os.listdir(path):

   #read image
   n_img = cv.imread(path+pic)
   #resize images with size inferior of resolution
   if (n_img.shape[:2]<resolution):
       n_img.resize(resolution)
   #crop image
   #size of cropped image equal to resolution
   i=randint(0,n_img.shape[0]-resolution[0])
   j= randint(0,n_img.shape[1]-resolution[1])
   img = n_img[i:i+resolution[0],j:j+resolution[1]]

   #probability of transparent text
   sometimes = randint(1, 10)

   img_process = ImageProcessing(img)

   # Create a white image
   img_white = utils.white_img(img)

   # get text caracteristics
   font, fontScale, fontColor, lineType = utils.text_caracteristics(img)

   # number of lines
   nb_lines = randint(1, 4)

    # name of image without the extension
   img_name = os.path.splitext(pic)[0]

   #write texte
   img, bt, left1, bottom, end_left, transparent = img_process.draw_text(pic,img,img_white,nb_lines,font,fontScale,fontColor,lineType,sometimes)

   images.append([img_name,img])
   transparents.append([sometimes,transparent])

   #text region
   positions.append([bt, left1, bottom, end_left])

#create a directory to hold training images
utils.create_dir("train_path")
#delete the
shutil.rmtree('train_path1')

#instance of Augmentor
augmentor = aug.Augmentor()
#apply blending to get transparent text
#for 10%  of images
transparency = augmentor.blending(images,transparents,positions)
#get augmented images
augmented_images=  augmentor.img_aug(images,positions)

#save the images on the train path
for each_img in augmented_images:
    img_label=each_img[0]
    picture=each_img[1]
    # save image
    cv.imwrite("train_path/" + img_label + ".jpg", picture)

