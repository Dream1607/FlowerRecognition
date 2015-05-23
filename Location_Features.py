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


def Segment_Mask(segments,label):
    # Make mask for each segment
    # mark 1 for particular label of segments
    # mark 0 for other pixels

    mask = [1 if i^label==0 else 0 for i in segments.flatten()]
    return np.array(mask).reshape(segments.shape)

def Super_Pixels_Location(img):
    segments, segments_pixels, segments_label = Segmentation.Label_Super_Pixels(Segmentation.Super_Pixels(image),
                                                                                Segmentation.Grab_Cut(image))
    row,col = segments.shape
    block_row = int(math.ceil(row/6))
    block_col = int(math.ceil(col/6))

    Location_Features = []
    for label in range(len(segments_label)):
        # Make mask for each segment
        seg_mask = Segment_Mask(segments, label)

        # Downsample to 6*6
        downsample = block_reduce(seg_mask, block_size=(block_row, block_col), cval = 0, func=np.max)

        # Convert to 36-D Location Features
        Location_Features.append(downsample.flatten().tolist())

    return Location_Features

if __name__ == '__main__':
    image = 'image.jpg'
    img = io.imread(image)
    
    Location_Features = Super_Pixels_Location(img)
