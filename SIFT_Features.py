import Segmentation

import cv2
import cPickle as pickle

import numpy as np

from matplotlib import pyplot as plt

from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

def pickle_keypoints(keypoints, descriptors): 
	i = 0 
	temp_array = []

	for point in keypoints:
		temp_array.append((point.pt, point.size, point.angle, point.response, point.octave, point.class_id, descriptors[i]))
		i = i + 1

	return np.array(temp_array)

def unpickle_keypoints(array): 
	keypoints = [] 
	descriptors = []

	for point in array: 
		temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5]) 
		temp_descriptor = point[6] 
		keypoints.append(temp_feature) 
		descriptors.append(temp_descriptor) 
	
	return keypoints, np.array(descriptors)

def SIFT_extract(img):
	sift = cv2.xfeatures2d.SIFT_create()

	kp,des = sift.detectAndCompute(img,None)
	temp = pickle_keypoints(kp, des)

	# img = cv2.drawKeypoints(img,kp,img)
	# plt.imshow(img),plt.show()
	return temp

def Superpixel_SIFT(image_SIFT):
	segments,segments_pixels,segments_label = Segmentation.Label_Super_Pixels(Segmentation.Super_Pixels(image), Segmentation.Grab_Cut(image))

	Super_Pixels_Features = [[] for y in range(len(segments_label))] 
	for index, input_vector in enumerate(image_SIFT):
		x = int(round(input_vector[0][1]))
		y = int(round(input_vector[0][0]))
		Super_Pixels_Features[segments[x][y]].append(input_vector[6])
	
	for index, vectors in enumerate(Super_Pixels_Features):
		len_v = 128
		mean_vector = np.zeros(len_v)
		if len(vectors) != 0:
			for i in range(len(vectors)):
				mean_vector += vectors[i]
			Super_Pixels_Features[index] = mean_vector/(len(vectors))
		else:
			Super_Pixels_Features[index] = mean_vector

	return np.array(Super_Pixels_Features)

# main function
if __name__ == "__main__":
	image = 'image.jpg'

	img = cv2.imread(image)

	image_SIFT = SIFT_extract(img)

	Super_Pixels_Features = Superpixel_SIFT(image_SIFT)