import Segmentation

import cv2
import cPickle as pickle

import numpy as np
import math

from matplotlib import pyplot as plt

from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import block_reduce


def Center_Boundary(img):
    segments, segments_pixels, segments_label = Segmentation.Label_Super_Pixels(Segmentation.Super_Pixels(image),
                                                                                Segmentation.Grab_Cut(image))
    row,col = segments.shape
    Center_Boundary_Features = np.zeros((len(segments_label),2))

    # Check Center
    Center_Boundary_Features[segments[int(row/2)][int(col/2)]][0] = 1

    # Check Boundary
    for i in range(row):
        Center_Boundary_Features[segments[i][0]][1] = 1
        Center_Boundary_Features[segments[i][col-1]][1] = 1
    for i in range(col):
        Center_Boundary_Features[segments[0][i]][1] = 1
        Center_Boundary_Features[segments[row-1][i]][1] = 1

    return Center_Boundary_Features

if __name__ == '__main__':
    image = 'image.jpg'
    img = io.imread(image)

    Center_Boundary_Features = Center_Boundary(img)
