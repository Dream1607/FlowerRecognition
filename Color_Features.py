import Segmentation

import cv2
import cPickle as pickle

import numpy as np

from matplotlib import pyplot as plt

from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

def Superpixel_Color_Hist(img):
	b, g, r = cv2.split(img)
	segments,segments_pixels,segments_label = Segmentation.Label_Super_Pixels(Segmentation.Super_Pixels(image), Segmentation.Grab_Cut(image))

	Color_Hist_Features = [[] for y in range(len(segments_label))] 
	for label in range(len(segments_label)):
		mask = np.zeros(img.shape[:2], np.uint8)
		rows,columns,rgb = np.array(img).shape

		for i in range(rows):
			for j in range(columns):
				mask[i][j] = (255 if segments[i][j] == label else 0)

		hist_mask_b = cv2.calcHist([img],[0],mask,[256],[0,256])
		hist_mask_g = cv2.calcHist([img],[1],mask,[256],[0,256])
		hist_mask_r = cv2.calcHist([img],[2],mask,[256],[0,256])

		Color_Hist_Features[label].append(hist_mask_b)
		Color_Hist_Features[label].append(hist_mask_g)
		Color_Hist_Features[label].append(hist_mask_r)

	return np.array(Color_Hist_Features)

if __name__ == '__main__':
	image = 'image.jpg'
	img = cv2.imread(image)

	Color_Hist_Features = Superpixel_Color_Hist(img)