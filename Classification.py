# -*- coding: utf-8 -*-
import cv2
import cPickle as pickle

import numpy as np

import os
import sys
import math
import datetime

from PIL import Image
from matplotlib import pyplot as plt

from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.measure import block_reduce

from sklearn import svm
from sklearn import datasets,metrics

def Color_Features_Extract(img_folder):
    print "Color_Features_Extract Start"
    starttime = datetime.datetime.now()

    image_num = len(os.listdir(seg_img_folder))

    Color_Features = []

    for index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") +image_name

        image = cv2.imread(image_path)
        rows, columns, bgr = image.shape

        # Make densely-sampling color features
        pixel_index = 0

        for x in range(rows):
            for y in range(columns):
                if pixel_index % 10 == 0:
                    Color_Features.append(image[x][y].tolist())
                pixel_index += 1
    
    # Get CodeBook of Color_Features
    Color_Features = np.float32(Color_Features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(Color_Features,200,None,criteria,10,flags)

    Image_Color_Features = [[0 for x in range(200)] for y in range(image_num)]

    color_index = 0

    for image_index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") +image_name
        img =  cv2.imread(image_path)
        
        pixel_index = 0

        for x in range(rows):
            for y in range(columns):
                if pixel_index % 10 == 0:
                    Image_Color_Features[image_index][labels[color_index]] += 1
                    color_index += 1
                pixel_index += 1

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "Color_Features_Extract End"

    return Image_Color_Features

def SIFT_Features_Extract(seg_img_folder):
    print "SIFT_Features_Extract Start"
    starttime = datetime.datetime.now()

    image_num = len(os.listdir(seg_img_folder))

    Image_KeyPoints_Num = [0 for y in range(image_num)]
    SIFT_Features = []

    for index, image_name in enumerate(os.listdir(seg_img_folder)):
        image_path = seg_img_folder + str("/") +image_name
        img =  cv2.imread(image_path)

        sift = cv2.xfeatures2d.SIFT_create(1000)
        keypoints,des = sift.detectAndCompute(img,None)

        k = 0

        for point in keypoints:
            SIFT_Features.append(des[k])
            k += 1

        Image_KeyPoints_Num[index] = k

    # Get CodeBook of SIFT_Features
    SIFT_Features = np.float32(SIFT_Features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(SIFT_Features,1000,None,criteria,10,flags)

    Image_SIFT_Features = [[0 for x in range(1000)] for y in range(image_num)]

    image_index = 0
    label_index = 0

    while image_index < image_num:
    	Image_SIFT_Features[image_index][labels[label_index]] += 1
        label_index += 1

        if label_index == sum(Image_KeyPoints_Num[0:image_index + 1]):
        	image_index += 1

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "SIFT_Features_Extract End"

    return Image_SIFT_Features

def Get_Features(seg_img_folder):
	Color_Features = Color_Features_Extract(seg_img_folder)
	SIFT_Features = SIFT_Features_Extract(seg_img_folder)
	
	Features = [ x + y for x,y in zip(Color_Features,SIFT_Features)]

	print np.array(Features).shape
	return Features

# main function
if __name__ == "__main__":
	seg_img_folder = 'seg_image'

	Get_Features(seg_img_folder)