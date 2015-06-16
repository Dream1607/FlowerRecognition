# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from optparse import OptionParser

def draw(img,save_name = False):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.savefig(save_name)

def Grab_Cut(img, name = False):
    height, weight, rgb = img.shape

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # mask initialized to PR_BG
    mask = np.zeros(img.shape[:2],np.uint8)

    # the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
    rect = (int(0.15 * weight),int(0.15 * height),int(0.7 * weight),int(0.7 * height))
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    draw(img,save_name = "grab" + str("/") + name)

parser = OptionParser()
parser.add_option("-i","--iterate", dest = "iterate", default = "5")

(options, args) = parser.parse_args()

options.iterate = float(options.iterate)

for image_index, image_name in enumerate(os.listdir("jpg")):
        image_path = "jpg" + str("/") + image_name
        img =  cv2.imread(image_path)

        Grab_Cut(img,image_name)
        print image_name