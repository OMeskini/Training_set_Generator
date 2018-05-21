import numpy as np
import cv2 as cv
import os
from random import randint, uniform, choice,gauss


def random_color():
    u = gauss(0, 1)
    v = gauss(0, 1)
    z = gauss(0, 1)

    R = abs(int(30 * u))
    G = abs(int(20 * v))
    B = abs(int(30 * z))
    return (R, G, B)

def lineOfText():
    # choose a text
    words = open('/etc/dictionaries-common/words').read().splitlines()

    line = ""
    while line == "":
        for k in range(1, randint(1, 4)):
            word = choice(words)
            line += word + " "

    return line

def create_dir(name_dir):
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)


def text_caracteristics(img):
 # list of fonts
        list = [cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_PLAIN, cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_DUPLEX,
                cv.FONT_HERSHEY_COMPLEX, cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_TRIPLEX,
                cv.FONT_HERSHEY_COMPLEX_SMALL, cv.FONT_HERSHEY_SCRIPT_SIMPLEX, cv.FONT_HERSHEY_SCRIPT_COMPLEX]

        # text caracteristics
        font = choice(list)
        fontScale = uniform(0.3, 6.5)
        lineType = randint(1, 5)

        # random color
        fontColor = random_color()


        # in case of little image apply an average fontscale
        if img.shape[0] < 200:
            fontScale = uniform(0.3, 3.5)

        # for visibility of text
        if (fontScale < 1):
           lineType = 1

        return font, fontScale, fontColor, lineType



class ImageProcessing:

    def __init__(self,img):
        self.img=img


    def white_img(self,img):
        # Create a white image
        img_white = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        img_white[:] = (255, 255, 255)
        return img_white

    def draw_text(self, pic, img, img_white, nb_lines, font, fontScale, fontColor, lineType):
        global text_bottom
        end_lines = 0
        height, weight = img.shape[:2]

        # foreach line
        for j in range(1, nb_lines+1):

            # load_text
            line = lineOfText()
            # get size of text
            textSize = cv.getTextSize(line, font, fontScale, lineType)
            w_text, h_text = textSize[0]


            if (j == 1):
              # Localization of text
              bottom = randint(h_text , height - h_text )
              while (w_text>weight):
                line=line[0:int(len(line)/2)]
                w_text=w_text/2
              left = randint(0, weight - w_text)
              text_bottom = bottom-h_text

            end_lines = max(left+w_text,end_lines)
            bottomLeftCornerOfText = (left, bottom)

            # write the text
            cv.putText(img, line,
                       bottomLeftCornerOfText,
                       font,
                       fontScale,
                       fontColor,
                       lineType)

            cv.putText(img_white, line,
                       bottomLeftCornerOfText,
                       font,
                       fontScale,
                       fontColor,
                       lineType)

            # save image
            cv.imwrite("train_path1/" + pic, img)

            cv.imwrite("label_path/" + pic, img_white)

            # for next line
            if (nb_lines > 1):
             # load image to write next line
             img = cv.imread("train_path1/" + pic)

             img_white = cv.imread("label_path/" + pic)

             # make space between lines
             bottom = bottom + h_text

             #bottomLeftCornerOfText[0] = bottom

        if(end_lines>weight):
            end_lines = weight

        return (img, text_bottom, left, bottom, end_lines)
